#!/usr/bin/env python
"""音声モデルのフレーム単位確率に、認識結果 BERT の確率を**発話末以降だけ**混ぜる。

後段で確率を平均する方式は「発話末 + 0.25 秒」の 1 点でしか測っていなかった
（audio_text_combine.py）。この研究の主張は早期確定による短縮なので、
停止規則の下で測らなければ実運用の値にならない。

因果性：BERT に与えるのは**発話全体の認識結果**なので、発話途中では手に入らない。
発話末より前の時刻は音声のみ、発話末以降だけ混ぜる。共同学習と同じ条件。

混合重みは dev（train_dev）で選び、eval で報告する。

  python local/turntaking/frame_fusion.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from frame_rules import load_frame, point, utt_lengths              # noqa: E402
import frame_rules                                                  # noqa: E402
from audio_text_combine import text_probs                           # noqa: E402


def fuse(P, Pt, grid, lens, w):
    """発話末以降のフレームだけ音声 w ・テキスト (1-w) で混ぜる。"""
    avail = grid[:, None] >= lens[None, :]              # (S, N)
    mixed = w * P + (1.0 - w) * Pt[None, :, :]
    return np.where(avail[:, :, None], mixed, P)


def fuse_conf(P, Pt, grid, lens, beta):
    """テキストの確信度に応じて混ぜる強さを変える。

    確信度は最大クラス確率。3 クラスなので一様分布が 1/3、確信が最大で 1。
    これを 0〜1 に伸ばして beta を掛けたものがテキスト側の重み。
    確信が無いときは音声のみに戻る。
    """
    lam = beta * np.clip((Pt.max(1) - 1 / 3) / (2 / 3), 0.0, 1.0)    # (N,)
    lam = lam[None, :, None]
    avail = grid[:, None] >= lens[None, :]
    mixed = (1.0 - lam) * P + lam * Pt[None, :, :]
    return np.where(avail[:, :, None], mixed, P)


def rules(P, lab, grid, cap, taus):
    """基準（区間検出まで待つ）と、それを崩さず最も早い完了 τ。"""
    S = len(grid)
    base = point(np.full(len(lab), S), P, lab, grid, cap, False)
    best = None
    for t in taus:
        ok = P[..., 1] >= t
        first = np.where(ok.any(0), ok.argmax(0), S)
        v = point(first, P, lab, grid, cap, True)
        if v[0] <= base[0] + 0.005 and v[1] <= base[1] + 0.005:
            if best is None or v[2] < best[1][2]:
                best = (t, v)
    return base, best


def _text_probs_maybe_ens(bert, dset, asr_text):
    """bert がカンマ区切りなら各モデルの確率を平均する（アンサンブル）。

    1 個だけ渡したときは従来と完全に同じ経路を通る。
    """
    dirs = [b for b in str(bert).split(",") if b]
    if len(dirs) == 1:
        return text_probs(dirs[0], dset, asr_text)
    acc = None
    for d in dirs:
        T = text_probs(d, dset, asr_text)
        if acc is None:
            acc = {u: np.array(q, dtype=np.float64) for u, q in T.items()}
        else:
            for u in list(acc):
                if u in T:
                    acc[u] += T[u]
                else:
                    del acc[u]
    n = float(len(dirs))
    print(f"  {dset}: BERT {len(dirs)} 個の確率を平均（{len(acc)} 区間）")
    return {u: q / n for u, q in acc.items()}


def prepare(stem, ckpt, dset, bert, asr_text, grid):
    P, lab, utts = load_frame(f"tt_preds/{stem}_{dset}_{ckpt}.tsv", grid)
    T = _text_probs_maybe_ens(bert, dset, asr_text)
    keep = [i for i, u in enumerate(utts) if u in T]
    if len(keep) < len(utts):
        print(f"  {dset}: テキストが無い {len(utts) - len(keep)} 区間を除外")
    P = P[:, keep, :]
    lab = lab[keep]
    utts = [utts[i] for i in keep]
    Pt = np.array([T[u] for u in utts])
    return P, Pt, lab, utts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--bert", default="exp/text_bert_asr",
                    help="カンマ区切りで複数渡すと確率を平均する（アンサンブル）")
    ap.add_argument("--bert_ref", default="exp/text_ceiling_bert",
                    help="正解書き起こしで学習した BERT（天井の測定用）")
    ap.add_argument("--asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/eval/text")
    ap.add_argument("--dev_asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/"
                    "train_dev_dec/text")
    ap.add_argument("--dev_set", default="train_dev",
                    help="重みを選ぶ側（新しい分割では dev_g）")
    ap.add_argument("--eval_set", default="eval",
                    help="報告する側（新しい分割では eval_g）")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_sec", type=float, default=3.0)
    args = ap.parse_args()

    grid = np.arange(args.step, args.max_sec + 1e-9, args.step)
    taus = np.round(np.arange(0.30, 1.00, 0.02), 2)
    ws = np.round(np.arange(0.0, 1.01, 0.05), 2)

    # ---- dev で重みを選ぶ ----
    print("dev（train_dev）を読み込み中…")
    Pd, Ptd, labd, uttsd = prepare(args.stem, args.ckpt, args.dev_set,
                                   args.bert, args.dev_asr_text, grid)
    lensd = utt_lengths(uttsd, args.dev_set)
    capd = np.clip(np.searchsorted(grid, lensd + args.max_wait, side="right") - 1,
                   0, len(grid) - 1)
    frame_rules.LENS = lensd
    devs = []
    for w in ws:
        base, _ = rules(fuse(Pd, Ptd, grid, lensd, w), labd, grid, capd, taus)
        devs.append((w, base[3]))
    w_best = max(devs, key=lambda x: x[1])[0]
    print(f"dev {len(uttsd)} 区間で選んだ重み: 音声 {w_best:.2f} "
          f"（dev macro-F1 {max(d[1] for d in devs):.4f}／"
          f"音声のみ {dict(devs)[1.0]:.4f}）")
    devs_b = []
    for b in ws:
        base, _ = rules(fuse_conf(Pd, Ptd, grid, lensd, b), labd, grid, capd, taus)
        devs_b.append((b, base[3]))
    b_best = max(devs_b, key=lambda x: x[1])[0]
    print(f"dev で選んだ確信度連動の強さ: β {b_best:.2f} "
          f"（dev macro-F1 {max(d[1] for d in devs_b):.4f}）")

    # 天井：正解書き起こし → 正解で学習した BERT。重みは同じ手続きで dev から選ぶ。
    _, Ptd_ref, _, _ = prepare(args.stem, args.ckpt, args.dev_set,
                               args.bert_ref, None, grid)
    devs_r = []
    for w in ws:
        base, _ = rules(fuse(Pd, Ptd_ref, grid, lensd, w), labd, grid, capd, taus)
        devs_r.append((w, base[3]))
    w_ref = max(devs_r, key=lambda x: x[1])[0]
    print(f"dev で選んだ重み（正解書き起こし）: 音声 {w_ref:.2f} "
          f"（dev macro-F1 {max(d[1] for d in devs_r):.4f}）\n")

    # ---- eval で報告する ----
    print("eval を読み込み中…")
    P, Pt, lab, utts = prepare(args.stem, args.ckpt, args.eval_set,
                               args.bert, args.asr_text, grid)
    lens = utt_lengths(utts, args.eval_set)
    cap = np.clip(np.searchsorted(grid, lens + args.max_wait, side="right") - 1,
                  0, len(grid) - 1)
    frame_rules.LENS = lens

    print(f"\neval {len(utts)} 区間 / 発話長 平均 {lens.mean():.2f} 秒 / "
          f"格子 {args.step} 秒 / 区間検出 {args.max_wait} 秒"
          f"【フレーム単位・テキストは発話末以降のみ】\n")
    print(f"{'条件':>28}{'規則':>16}{'応答ミス':>9}{'誤割り込み':>10}"
          f"{'完了の遅延':>11}{'macroF1':>9}{'短縮':>8}")

    _, Pt_ref, _, _ = prepare(args.stem, args.ckpt, args.eval_set,
                              args.bert_ref, None, grid)

    for name, w in (("音声のみ", 1.0), (f"融合（dev 選択・音声{w_best:.2f}）", w_best),
                    (f"確信度連動（dev 選択・β{b_best:.2f}）", None),
                    ("　参考 音声0.50", 0.50), ("　参考 音声0.30", 0.30),
                    (f"正解書き起こし（到達不能・音声{w_ref:.2f}）", "ref")):
        Pf = (fuse_conf(P, Pt, grid, lens, b_best) if w is None
              else fuse(P, Pt_ref, grid, lens, w_ref) if w == "ref"
              else fuse(P, Pt, grid, lens, w))
        base, best = rules(Pf, lab, grid, cap, taus)
        print(f"{name:>28}{'区間検出まで待つ':>16}{base[0]:9.3f}{base[1]:10.3f}"
              f"{base[2]:+11.3f}{base[3]:9.4f}{'—':>8}")
        if best:
            t, v = best
            print(f"{'':>28}{f'完了τ={t:.2f}':>16}{v[0]:9.3f}{v[1]:10.3f}"
                  f"{v[2]:+11.3f}{v[3]:9.4f}{base[2]-v[2]:8.3f}")
        else:
            print(f"{'':>28}{'（基準を超えられない）':>16}")
        for t in (0.70, 0.80):
            ok = Pf[..., 1] >= t
            first = np.where(ok.any(0), ok.argmax(0), len(grid))
            v = point(first, Pf, lab, grid, cap, True)
            print(f"{'':>28}{f'完了τ={t:.2f}':>16}{v[0]:9.3f}{v[1]:10.3f}"
                  f"{v[2]:+11.3f}{v[3]:9.4f}{base[2]-v[2]:8.3f}")
        print()

    # ---- 重みを振ったときの動作点曲線（応答ミスと誤割り込みの交換比率）----
    print("重みを振ったときの動作点【区間検出まで待つ／eval】\n")
    print(f"{'音声の重み':>10}{'応答ミス':>10}{'誤割り込み':>11}{'macroF1':>10}"
          f"{'継続F1':>9}{'完了F1':>9}{'相槌F1':>9}")
    S = len(grid)
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        Pf = fuse(P, Pt, grid, lens, w)
        base, _ = rules(Pf, lab, grid, cap, taus)
        pred = Pf.argmax(-1)[np.minimum(np.full(len(lab), S), cap),
                             np.arange(len(lab))]
        f1, mac, _, _ = prf(lab, pred)
        mark = " ←dev 選択" if abs(w - round(w_best, 1)) < 1e-9 else ""
        print(f"{w:10.2f}{base[0]:10.3f}{base[1]:11.3f}{base[3]:10.4f}"
              f"{f1[0]:9.3f}{f1[1]:9.3f}{f1[2]:9.3f}{mark}")


if __name__ == "__main__":
    main()
