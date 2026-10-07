#!/usr/bin/env python
"""N-best 認識結果を融合に使う。まずオラクルで上限を測る。

天井（正解書き起こし 0.7829）と実測（0.7429）の差 +0.040 は、認識結果が
意味的に間違っていることに起因する。門番（捨てる）の上限は +0.0025 しか
無かったので、残るのは「情報を増やす」方向だけ。

実装の前に上限を測る手順は門番のときと同じ。
  オラクル … 5 候補のうち**正解書き起こしに最も近いもの**を反則的に選ぶ。
              これが 0.7429 から動かなければ、N-best という筋に余地は無い。
  実運用   … ASR スコアの softmax を重みに 5 候補の BERT 確率を平均する。
              候補の選択はしない（BERT は誤った認識結果にも自信を持つため）。

因果性は frame_fusion.py と同じ。テキストは発話末以降のフレームでのみ使う。

  python local/turntaking/nbest_fusion.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from frame_rules import point, utt_lengths                          # noqa: E402
import frame_rules                                                  # noqa: E402
from frame_fusion import prepare, rules                             # noqa: E402
from audio_text_combine import text_probs                           # noqa: E402
from fusion_by_cer import cer, read_map                             # noqa: E402


def nbest_paths(decode_dir, dset, n):
    return f"{decode_dir}/{dset}/{n}best_recog"


def load_nbest(decode_dir, dset, N, bert):
    """N 候補それぞれのテキスト・スコア・BERT 確率。"""
    texts, scores, probs = [], [], []
    for n in range(1, N + 1):
        d = nbest_paths(decode_dir, dset, n)
        texts.append(read_map(f"{d}/text"))
        scores.append({k: float(v) for k, v in read_map(f"{d}/score").items()})
        probs.append(text_probs(bert, dset, f"{d}/text"))
    return texts, scores, probs


def stack(utts, per_utt, dim=3):
    """[{utt: vec}, ...] → (N, U, dim) と有効性マスク (N, U)。

    ビームが 5 本返さない発話がある（eval で 8〜20 件）。欠けた候補を確率 0 の
    ベクトルとして扱うと平均もオラクルも狂うので、マスクで除外する。
    """
    out = np.zeros((len(per_utt), len(utts), dim))
    ok = np.zeros((len(per_utt), len(utts)), dtype=bool)
    for i, d in enumerate(per_utt):
        for j, u in enumerate(utts):
            v = d.get(u)
            if v is not None:
                out[i, j] = v
                ok[i, j] = True
    return out, ok


def mbr_pick(texts, utts, ok, post):
    """最小ベイズリスク選択。他の候補に最も近い候補を選ぶ。

    risk(i) = Σ_j post_j · CER(候補i, 候補j)
    BERT は一切見ない（誤った認識結果にも自信を持つため）。認識側の合意だけで選ぶ。
    """
    N = len(texts)
    pick = np.zeros(len(utts), dtype=int)
    for j, u in enumerate(utts):
        cand = [texts[n].get(u, "") if ok[n, j] else None for n in range(N)]
        idx = [n for n in range(N) if cand[n] is not None]
        if len(idx) <= 1:
            continue
        risk = []
        for a in idx:
            r = sum(post[b, j] * cer(cand[b], cand[a]) for b in idx if b != a)
            risk.append((r, a))
        pick[j] = min(risk)[1]
    return pick


def mix_frames(P, Pt, grid, lens, w):
    avail = grid[:, None] >= lens[None, :]
    mixed = w * P + (1.0 - w) * Pt[None, :, :]
    return np.where(avail[:, :, None], mixed, P)


def report(name, Pf, lab, grid, cap, taus):
    base, best = rules(Pf, lab, grid, cap, taus)
    S = len(grid)
    pred = Pf.argmax(-1)[np.minimum(np.full(len(lab), S), cap), np.arange(len(lab))]
    f1, _, _, _ = prf(lab, pred)
    line = (f"{name:>34}{base[0]:9.3f}{base[1]:10.3f}{base[3]:10.4f}"
            f"{f1[0]:8.3f}{f1[1]:8.3f}{f1[2]:8.3f}")
    if best:
        line += f"{best[1][3]:10.4f}{base[2]-best[1][2]:8.3f}"
    print(line)
    return base[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--decode_dir", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_nbest5_asr_model_valid.loss.ave")
    ap.add_argument("--onebest", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/eval/text")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--w", type=float, default=0.45)
    ap.add_argument("--dev_onebest", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/"
                    "train_dev_dec/text")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_sec", type=float, default=3.0)
    args = ap.parse_args()

    grid = np.arange(args.step, args.max_sec + 1e-9, args.step)
    taus = np.round(np.arange(0.30, 1.00, 0.02), 2)

    # ---- dev で重み w と温度 tau を選ぶ（eval 上で掃引すると楽観的になる）----
    print("dev を読み込み中…")
    dP, dPt1, dlab, dutts = prepare(args.stem, args.ckpt, "train_dev",
                                    args.bert, args.dev_onebest, grid)
    dlens = utt_lengths(dutts, "train_dev")
    dcap = np.clip(np.searchsorted(grid, dlens + args.max_wait, side="right") - 1,
                   0, len(grid) - 1)
    dtexts, dscores, dprobs = load_nbest(args.decode_dir, "train_dev_dec",
                                         args.N, args.bert)
    dT, dokT = stack(dutts, dprobs)
    dSc, dokS = stack(dutts, [{k: [v] for k, v in x.items()} for x in dscores], dim=1)
    dSc = dSc[..., 0]
    dok = dokT & dokS
    dok[0] = True
    dT = np.where(dok[:, :, None], dT, dT[0][None, :, :])

    def dev_macro(Pt, w):
        frame_rules.LENS = dlens
        return rules(mix_frames(dP, Pt, grid, dlens, w), dlab, grid, dcap, taus)[0][3]

    best = (None, None, -1.0)
    for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
        z = np.where(dok, dSc / tau, -np.inf)
        z = z - z.max(0, keepdims=True)
        a = np.exp(z); a /= a.sum(0, keepdims=True)
        Ptd = (a[:, :, None] * dT).sum(0)
        for w in np.round(np.arange(0.25, 0.76, 0.05), 2):
            m = dev_macro(Ptd, w)
            if m > best[2]:
                best = (tau, w, m)
    tau_best, w_best, dev_best = best
    d1 = max(dev_macro(dPt1, w) for w in np.round(np.arange(0.25, 0.76, 0.05), 2))
    print(f"dev {len(dutts)} 区間で選択: 温度 {tau_best} / 音声の重み {w_best:.2f} "
          f"（dev macro-F1 {dev_best:.4f}／1-best 最良 {d1:.4f}）\n")

    print("eval を読み込み中…")
    P, Pt1, lab, utts = prepare(args.stem, args.ckpt, "eval",
                                args.bert, args.onebest, grid)
    lens = utt_lengths(utts, "eval")
    cap = np.clip(np.searchsorted(grid, lens + args.max_wait, side="right") - 1,
                  0, len(grid) - 1)
    frame_rules.LENS = lens

    print(f"N-best（{args.N} 候補）を読み込み中…")
    texts, scores, probs = load_nbest(args.decode_dir, "eval", args.N, args.bert)
    T, okT = stack(utts, probs)                              # (N,U,3)
    Sc, okS = stack(utts, [{k: [v] for k, v in s.items()} for s in scores], dim=1)
    Sc = Sc[..., 0]
    ok = okT & okS
    ok[0] = True                       # 1 位は必ずある
    T = np.where(ok[:, :, None], T, T[0][None, :, :])        # 欠けたら 1 位で埋める

    ref = read_map("data/eval/text")
    C = np.array([[cer(ref.get(u, ""), texts[n].get(u, "")) if texts[n].get(u) is not None
                   else np.inf for u in utts]
                  for n in range(args.N)])                   # (N,U)
    C = np.where(ok, C, np.inf)
    C[0] = np.where(np.isfinite(C[0]), C[0], 1.0)

    print(f"\neval {len(utts)} 区間 / 判定時点＝区間検出（発話末 + {args.max_wait} 秒）")
    print(f"1 位の CER 平均 {C[0].mean():.3f} / "
          f"N 候補の最小 CER 平均 {C.min(0).mean():.3f} "
          f"（オラクルで選べば CER がここまで下がる）")
    uniq = np.array([len({texts[n].get(u) for n in range(args.N)
                          if ok[n, j]}) for j, u in enumerate(utts)])
    print(f"有効な候補数 平均 {ok.sum(0).mean():.2f}")
    print(f"候補が実質何通りか（重複除く）平均 {uniq.mean():.2f} / "
          f"1 通りしかない割合 {(uniq == 1).mean():.3f}")
    better = (C.min(0) < C[0] - 1e-9)
    print(f"2 位以下に 1 位より良い候補がある割合 {better.mean():.3f}\n")

    print(f"{'条件':>34}{'応答ミス':>9}{'誤割り込み':>10}{'macroF1':>10}"
          f"{'継続F1':>8}{'完了F1':>8}{'相槌F1':>8}{'τ最良':>10}{'短縮':>8}")
    report("音声のみ", P, lab, grid, cap, taus)
    report(f"1-best 融合（音声{args.w:.2f}）",
           mix_frames(P, Pt1, grid, lens, args.w), lab, grid, cap, taus)
    frame_rules.LENS = lens

    # オラクル：正解に最も近い候補を選ぶ（到達不能）
    pick = C.argmin(0)
    Pt_or = T[pick, np.arange(len(utts))]
    report("オラクル N-best（到達不能）",
           mix_frames(P, Pt_or, grid, lens, args.w), lab, grid, cap, taus)

    # 実運用：dev で選んだ温度と重み
    z = np.where(ok, Sc / tau_best, -np.inf)
    z = z - z.max(0, keepdims=True)
    a = np.exp(z); a /= a.sum(0, keepdims=True)
    report(f"スコア重み付き平均（dev 選択・温度{tau_best}・音声{w_best:.2f}）",
           mix_frames(P, (a[:, :, None] * T).sum(0), grid, lens, w_best),
           lab, grid, cap, taus)

    # MBR 選択（認識側の合意だけで 1 本選ぶ）
    z = np.where(ok, Sc / tau_best, -np.inf)
    z = z - z.max(0, keepdims=True)
    post = np.exp(z); post /= post.sum(0, keepdims=True)
    pk = mbr_pick(texts, utts, ok, post)
    report(f"MBR 選択（温度{tau_best}・音声{w_best:.2f}）",
           mix_frames(P, T[pk, np.arange(len(utts))], grid, lens, w_best),
           lab, grid, cap, taus)
    print(f"{'':>34}（1 位と違う候補を選んだ割合 {(pk != 0).mean():.3f}）")

    # 参考：温度を振る（eval 上の掃引なので参考値）
    for tau in (0.5, 1.0, 2.0):
        z = np.where(ok, Sc / tau, -np.inf)
        z = z - z.max(0, keepdims=True)
        a = np.exp(z); a /= a.sum(0, keepdims=True)
        Pt_avg = (a[:, :, None] * T).sum(0)
        report(f"スコア重み付き平均（温度{tau:.1f}）",
               mix_frames(P, Pt_avg, grid, lens, args.w), lab, grid, cap, taus)

    # 参考：単純平均
    m = ok[:, :, None]
    report("N-best 単純平均",
           mix_frames(P, (T * m).sum(0) / m.sum(0), grid, lens, args.w),
           lab, grid, cap, taus)


if __name__ == "__main__":
    main()
