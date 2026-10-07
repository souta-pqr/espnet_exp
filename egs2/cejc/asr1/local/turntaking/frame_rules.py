#!/usr/bin/env python
"""フレーム単位ダンプの上で停止ルールを比べる（モデル間の公平な比較）。

フレーム単位ダンプは 1 区間あたり約 90 時点の確率を持つ。
これを共通の時刻格子に載せ直し、各モデルに同じ停止ルールを適用する。
指標は response_tradeoff.py と同じ「応答ミス／誤割り込み／完了の遅延」。

  応答ミス   = P(完了と判定しない | 真＝完了)   … システムが応答しない
  誤割り込み = P(完了と判定する   | 真≠完了)   … 話の途中で遮る
  完了の遅延 = 真に完了だった区間での（確定時刻 − 発話末）

どの規則も「音声区間検出が発話末 + max_wait 秒で発火する」制約の下で測る。

  python local/turntaking/frame_rules.py --stems frame_attn-trunc-new_same_n5 frame_attn-frame_same_n5
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402


def load_frame(path, grid):
    """フレーム単位ダンプを共通格子に載せる → (S,N,3), ラベル (N,), utt 一覧。"""
    el = collections.defaultdict(list)
    pr = collections.defaultdict(list)
    lab = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            u = p[0]; lab[u] = int(p[1])
            el[u].append(float(p[3]))
            pr[u].append((float(p[4]), float(p[5]), float(p[6])))
    utts = sorted(el)
    P = np.zeros((len(grid), len(utts), 3))
    for n, u in enumerate(utts):
        e = np.asarray(el[u]); q = np.asarray(pr[u])
        # 時刻 t では「t 以下で最も新しいフレーム」を使う（先読みしない）
        idx = np.searchsorted(e, grid, side="right") - 1
        idx = np.clip(idx, 0, len(e) - 1)
        P[:, n, :] = q[idx]
    return P, np.array([lab[u] for u in utts]), utts


def utt_lengths(utts, dset):
    seg = {}
    for line in open(f"data/{dset}/segments"):
        u, s, b, e = line.split(); seg[u] = float(e) - float(b)
    return np.array([seg[u] for u in utts])


def point(first, P, lab, grid, cap, force_end):
    """規則の動作点 → (応答ミス, 誤割り込み, 完了の遅延, macro-F1, 完了F1)。"""
    S = len(grid)
    fired = first < S
    idx = np.minimum(np.where(fired, first, cap), cap)
    pr = P.argmax(-1)[idx, np.arange(len(lab))]
    if force_end:
        pr = np.where(fired & (first <= cap), 1, pr)
    lat = grid[idx] - LENS
    f1, mac, uar, acc = prf(lab, pr)
    m1 = lab == 1
    return ((pr[m1] != 1).mean(),
            ((pr == 1) & ~m1).sum() / max((~m1).sum(), 1),
            lat[m1].mean(), mac, f1[1])


def main():
    global LENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--dset", default="eval")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05, help="共通格子の刻み（秒）")
    ap.add_argument("--max_sec", type=float, default=3.0)
    ap.add_argument("--grid_points", default="",
                    help="共通格子を明示指定（既存グリッドと同じ時刻で比べるとき）")
    args = ap.parse_args()

    grid = (np.array([float(x) for x in args.grid_points.split()])
            if args.grid_points
            else np.arange(args.step, args.max_sec + 1e-9, args.step))
    per = {}
    for stem in args.stems:
        path = f"tt_preds/{stem}_{args.dset}_{args.ckpt}.tsv"
        if not Path(path).exists():
            print(f"！ 無い: {path}"); continue
        P, lab, utts = load_frame(path, grid)
        per[stem] = (P, lab, utts)

    if not per:
        sys.exit("ダンプが読めない")
    base_utts = next(iter(per.values()))[2]
    LENS = utt_lengths(base_utts, args.dset)
    cap = np.clip(np.searchsorted(grid, LENS + args.max_wait, side="right") - 1,
                  0, len(grid) - 1)

    print(f"{args.dset} {len(base_utts)} 区間 / 発話長 平均 {LENS.mean():.2f} 秒 / "
          f"格子 {args.step} 秒 / 区間検出 {args.max_wait} 秒【フレーム単位・無音込み】\n")

    taus = np.round(np.arange(0.30, 1.00, 0.02), 2)
    print(f"{'モデル':>30}{'規則':>16}{'応答ミス':>9}{'誤割り込み':>10}"
          f"{'完了の遅延':>11}{'macroF1':>9}{'短縮':>8}")
    for stem, (P, lab, utts) in per.items():
        S = len(grid)
        base = point(np.full(len(lab), S), P, lab, grid, cap, False)
        print(f"{stem:>30}{'区間検出まで待つ':>16}{base[0]:9.3f}{base[1]:10.3f}"
              f"{base[2]:+11.3f}{base[3]:9.4f}{'—':>8}")
        # どちらの失敗も基準以下に保ったまま、最も早く応答できる完了 τ
        best = None
        for t in taus:
            ok = P[..., 1] >= t
            first = np.where(ok.any(0), ok.argmax(0), S)
            v = point(first, P, lab, grid, cap, True)
            if v[0] <= base[0] + 0.005 and v[1] <= base[1] + 0.005:
                if best is None or v[2] < best[1][2]:
                    best = (t, v)
        if best:
            t, v = best
            print(f"{'':>30}{f'完了τ={t:.2f}':>16}{v[0]:9.3f}{v[1]:10.3f}"
                  f"{v[2]:+11.3f}{v[3]:9.4f}{base[2]-v[2]:8.3f}")
        else:
            print(f"{'':>30}{'（基準を超えられない）':>16}")
        # 参考：固定 τ での動作点
        for t in (0.70, 0.80):
            ok = P[..., 1] >= t
            first = np.where(ok.any(0), ok.argmax(0), S)
            v = point(first, P, lab, grid, cap, True)
            print(f"{'':>30}{f'完了τ={t:.2f}':>16}{v[0]:9.3f}{v[1]:10.3f}"
                  f"{v[2]:+11.3f}{v[3]:9.4f}{base[2]-v[2]:8.3f}")
        print()


if __name__ == "__main__":
    main()
