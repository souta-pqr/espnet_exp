#!/usr/bin/env python
"""音声モデルと認識結果 BERT を、**学習する結合器**でまとめる。

単一の重みで混ぜる方式（frame_fusion.py）は、クラスごとに最適な重みが違うのに
1 つで妥協している。クラス別内訳を見ると、テキストは
  継続   … 大きく効く（F1 0.531 → 0.568）
  完了   … むしろ悪化（0.740 → 0.723）
  相槌   … 無関係（0.93 で横ばい）
と振る舞いが真逆なので、共通の重みでは取りこぼす。

両者の対数確率 6 次元を入力とする多項ロジスティック回帰を dev で学習する。
これは線形混合・幾何平均（対数線形プーリング）・クラス別重み・温度校正を
すべて特殊ケースとして含む。パラメータ 21 個に対し dev は 14,039 件。

因果性は frame_fusion.py と同じ。テキストは発話末以降のフレームでのみ使う。

  python local/turntaking/frame_stack.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from frame_rules import point, utt_lengths                          # noqa: E402
import frame_rules                                                  # noqa: E402
from frame_fusion import fuse, prepare, rules                       # noqa: E402

EPS = 1e-8


def fit(Xd, yd, l2=1e-3, steps=400, balanced=True):
    """多項ロジスティック回帰を L-BFGS で解く。

    balanced=True でクラス頻度の逆数を重みにする。評価指標 macro-F1 はクラスを
    対等に扱うのに、素の交差エントロピーは事前分布を再現して多数派に寄るため。
    """
    X = torch.tensor(Xd, dtype=torch.float64)
    y = torch.tensor(yd, dtype=torch.long)
    if balanced:
        cnt = np.bincount(yd, minlength=3).astype(np.float64)
        cw = torch.tensor(len(yd) / (3.0 * np.maximum(cnt, 1)), dtype=torch.float64)
    else:
        cw = None
    W = torch.zeros(X.shape[1], 3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y, weight=cw) \
            + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def tune_offsets(z, lab, lo=-2.0, hi=2.0, step=0.1):
    """クラスごとのロジット補正を dev で macro-F1 に対して直接合わせる。

    結合器は対数尤度で学習するので、指標（macro-F1＝事前分布を無視する量）とは
    動作点が噛み合わない。情報の統合と動作点の選択を分ける。
    自由度は 2（3 クラスの補正は定数だけ冗長なので最後を 0 に固定）。
    """
    best = (None, -1.0)
    rng = np.arange(lo, hi + 1e-9, step)
    for o0 in rng:
        for o1 in rng:
            off = np.array([o0, o1, 0.0])
            mac = prf(lab, (z + off).argmax(1))[1]
            if mac > best[1]:
                best = (off, mac)
    return best


def apply_stack(P, Pt, grid, lens, W, b, off=None):
    """発話末以降のフレームだけ結合器の出力に差し替える。"""
    S, N, _ = P.shape
    ls = np.log(np.clip(P, EPS, 1.0))                       # (S,N,3)
    lt = np.log(np.clip(Pt, EPS, 1.0))                      # (N,3)
    X = np.concatenate([ls, np.broadcast_to(lt, (S, N, 3))], axis=2)   # (S,N,6)
    z = X @ W + b
    if off is not None:
        z = z + off
    z -= z.max(axis=2, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=2, keepdims=True)
    avail = grid[:, None] >= lens[None, :]
    return np.where(avail[:, :, None], q, P)


def show(name, Pf, lab, grid, cap, taus):
    base, best = rules(Pf, lab, grid, cap, taus)
    S = len(grid)
    pred = Pf.argmax(-1)[np.minimum(np.full(len(lab), S), cap), np.arange(len(lab))]
    f1, _, _, _ = prf(lab, pred)
    print(f"{name:>30}{'区間検出まで待つ':>16}{base[0]:9.3f}{base[1]:10.3f}"
          f"{base[2]:+11.3f}{base[3]:9.4f}{'—':>8}"
          f"{f1[0]:8.3f}{f1[1]:8.3f}{f1[2]:8.3f}")
    if best:
        t, v = best
        print(f"{'':>30}{f'完了τ={t:.2f}':>16}{v[0]:9.3f}{v[1]:10.3f}"
              f"{v[2]:+11.3f}{v[3]:9.4f}{base[2]-v[2]:8.3f}")
    for t in (0.70, 0.80):
        ok = Pf[..., 1] >= t
        first = np.where(ok.any(0), ok.argmax(0), S)
        v = point(first, Pf, lab, grid, cap, True)
        print(f"{'':>30}{f'完了τ={t:.2f}':>16}{v[0]:9.3f}{v[1]:10.3f}"
              f"{v[2]:+11.3f}{v[3]:9.4f}{base[2]-v[2]:8.3f}")
    print()
    return base[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/eval/text")
    ap.add_argument("--dev_asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/"
                    "train_dev_dec/text")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_sec", type=float, default=3.0)
    args = ap.parse_args()

    grid = np.arange(args.step, args.max_sec + 1e-9, args.step)
    taus = np.round(np.arange(0.30, 1.00, 0.02), 2)
    ws = np.round(np.arange(0.0, 1.01, 0.05), 2)

    print("dev（train_dev）を読み込み中…")
    Pd, Ptd, labd, uttsd = prepare(args.stem, args.ckpt, "train_dev",
                                   args.bert, args.dev_asr_text, grid)
    lensd = utt_lengths(uttsd, "train_dev")
    capd = np.clip(np.searchsorted(grid, lensd + args.max_wait, side="right") - 1,
                   0, len(grid) - 1)
    frame_rules.LENS = lensd

    # 比較用：単一重みも同じ dev で選ぶ
    devs = [(w, rules(fuse(Pd, Ptd, grid, lensd, w), labd, grid, capd, taus)[0][3])
            for w in ws]
    w_best = max(devs, key=lambda x: x[1])[0]

    # 結合器は「区間検出まで待つ」判定時点の事例で学習する
    n = len(labd)
    ls = np.log(np.clip(Pd[capd, np.arange(n), :], EPS, 1.0))
    lt = np.log(np.clip(Ptd, EPS, 1.0))
    Xd = np.concatenate([ls, lt], axis=1)
    W, b = fit(Xd, labd, balanced=True)
    W0, b0 = fit(Xd, labd, balanced=False)
    cnt = np.bincount(labd, minlength=3)
    print(f"dev {n} 区間で学習（継続 {cnt[0]} / 完了 {cnt[1]} / 相槌 {cnt[2]}）。"
          f"単一重みの選択は音声 {w_best:.2f}")
    off, mac_off = tune_offsets(Xd @ W + b, labd)
    off0, mac_off0 = tune_offsets(Xd @ W0 + b0, labd)
    print(f"dev で選んだ動作点の補正: 継続 {off[0]:+.1f} / 完了 {off[1]:+.1f} "
          f"（dev macro-F1 {prf(labd, (Xd @ W + b).argmax(1))[1]:.4f} → {mac_off:.4f}）")
    print(f"　　重み無しの場合: 継続 {off0[0]:+.1f} / 完了 {off0[1]:+.1f} "
          f"（dev macro-F1 {mac_off0:.4f}）\n")

    print("結合器の係数（行＝入力、列＝出力クラス）")
    names = ["音声 継続", "音声 完了", "音声 相槌", "テキスト 継続",
             "テキスト 完了", "テキスト 相槌"]
    print(f"{'':>14}{'→継続':>9}{'→完了':>9}{'→相槌':>9}")
    for nm, row in zip(names, W):
        print(f"{nm:>14}{row[0]:9.3f}{row[1]:9.3f}{row[2]:9.3f}")
    print(f"{'切片':>14}{b[0]:9.3f}{b[1]:9.3f}{b[2]:9.3f}\n")

    print("eval を読み込み中…")
    P, Pt, lab, utts = prepare(args.stem, args.ckpt, "eval",
                               args.bert, args.asr_text, grid)
    lens = utt_lengths(utts, "eval")
    cap = np.clip(np.searchsorted(grid, lens + args.max_wait, side="right") - 1,
                  0, len(grid) - 1)
    frame_rules.LENS = lens

    print(f"\neval {len(utts)} 区間 / 格子 {args.step} 秒 / 区間検出 {args.max_wait} 秒"
          f"【テキストは発話末以降のみ】\n")
    print(f"{'条件':>30}{'規則':>16}{'応答ミス':>9}{'誤割り込み':>10}"
          f"{'完了の遅延':>11}{'macroF1':>9}{'短縮':>8}"
          f"{'継続F1':>8}{'完了F1':>8}{'相槌F1':>8}")
    show("音声のみ", P, lab, grid, cap, taus)
    show(f"単一重み（音声{w_best:.2f}）", fuse(P, Pt, grid, lens, w_best),
         lab, grid, cap, taus)
    show("結合器（クラス重み付き）", apply_stack(P, Pt, grid, lens, W, b),
         lab, grid, cap, taus)
    show("結合器＋動作点調整", apply_stack(P, Pt, grid, lens, W, b, off),
         lab, grid, cap, taus)
    show("結合器（重み無し）＋動作点調整",
         apply_stack(P, Pt, grid, lens, W0, b0, off0), lab, grid, cap, taus)


if __name__ == "__main__":
    main()
