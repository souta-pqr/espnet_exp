#!/usr/bin/env python
"""先読みヘッド（tt_horizons）による早期確定を、前向きグリッドのダンプから評価する。

forward_commit.py と同じ前向きグリッド（発話開始からの経過時間・無音込み）を読み、
「p(h 秒以内に完了で終わる) ≥ θ になった最初の格子点で <完了> を確定」という規則を、
従来の単一 τ（最大確率）・完了確率 τ・経過時間ゲートと同じ土俵で比べる。
どの規則も「区間検出が発話末 + max_wait 秒で発火する」制約の下で測る（11 節と同じ）。

指標（forward_commit.py と同じ定義）:
  遅延平均    … 確定時刻 − 発話末（全区間）
  完了の遅延  … 真に完了だった区間での同上（応答の速さそのもの）
  取りこぼし  … 真に完了なのに完了と判定しなかった割合
  誤割り込み  … 完了でないのに完了と判定した割合
  早切り      … 真に完了の区間で発話末より 0.5 秒以上手前に確定した割合
  macro-F1 / 完了 F1 … 確定時点の 3 クラス判定

  python local/turntaking/anticipate_commit.py --stem pool_attn-tail_same_n5 --sil --max_wait 0.25
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                     # noqa: E402

GRID = ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0", "2.5", "3.0"]


def load_tsv_h(path):
    """utt / ラベル / 3 クラス確率 / 先読み確率 (N,K) / ホライズン名。"""
    utt, lab, prob, ph = [], [], [], []
    with open(path, encoding="utf-8") as f:
        head = next(f).rstrip("\n").split("\t")
        hnames = [float(c[3:]) for c in head[7:] if c.startswith("p_h")]
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            utt.append(p[0]); lab.append(int(p[1]))
            prob.append([float(p[3]), float(p[4]), float(p[5])])
            ph.append([float(x) for x in p[7:7 + len(hnames)]])
    return utt, np.array(lab), np.array(prob), np.array(ph), hnames


def load_forward(stem, ckpt, grid, dev=False, sil=True):
    sfx = "_train_dev" if dev else ""
    ss = "s" if sil else ""
    P, H, base, lab, hn = [], [], None, None, None
    for e in grid:
        u, y, p, h, hnames = load_tsv_h(f"tt_preds/{stem}{sfx}_e{e}{ss}_{ckpt}.tsv")
        if base is None:
            base, lab, hn = u, y, hnames
            order = {x: i for i, x in enumerate(u)}
        else:
            assert len(u) == len(base), f"区間数が不一致: {e}"
            idx = np.array([order[x] for x in u])
            q = np.zeros_like(p); q[idx] = p; p = q
            qh = np.zeros_like(h); qh[idx] = h; h = qh
        P.append(p); H.append(h)
    return np.stack(P), np.stack(H), lab, base, hn


def utt_lengths(utts, dset):
    seg = {}
    for line in open(f"data/{dset}/segments"):
        u, s, b, e = line.split(); seg[u] = float(e) - float(b)
    return np.array([seg[u] for u in utts])


def cap_index(ev, length, max_wait):
    """区間検出が発話末 + max_wait で発火 → それ以下の最後の格子点で強制確定。"""
    idx = np.searchsorted(ev, length + max_wait, side="right") - 1
    return np.clip(idx, 0, len(ev) - 1)


def evaluate(name, first, pred_at_first, P, lab, length, ev, cap, rows):
    """first: 規則が発火した格子点 (N,)（発火しなければ len(ev)）。cap で強制確定。"""
    S = len(ev)
    fired = first < S
    idx = np.where(fired, np.minimum(first, cap), cap)
    j = np.arange(len(lab))
    pr = P.argmax(-1)[idx, j]
    if pred_at_first is not None:                     # 規則発火時の予測クラスを上書き
        pr = np.where(fired & (first <= cap), pred_at_first, pr)
    lat = ev[idx] - length
    f1, mac, uar, acc = prf(lab, pr)
    m1 = lab == 1
    rows.append((name, lat.mean(), lat[m1].mean(), (pr[m1] != 1).mean(),
                 ((pr == 1) & ~m1).sum() / max((~m1).sum(), 1),
                 (lat[m1] < -0.5).mean(), mac, f1[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-tail_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--grid", default=" ".join(GRID))
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--sil", action="store_true", default=True)
    ap.add_argument("--nosil", dest="sil", action="store_false")
    ap.add_argument("--dev", action="store_true", help="eval ではなく train_dev で表を出す")
    ap.add_argument("--thetas", default="0.5 0.6 0.7 0.8 0.9 0.95")
    args = ap.parse_args()
    grid = args.grid.split()
    ev = np.array([float(e) for e in grid])
    dset = "train_dev" if args.dev else "eval"
    P, H, lab, utts, hn = load_forward(args.stem, args.ckpt, grid, dev=args.dev, sil=args.sil)
    L = utt_lengths(utts, dset)
    cap = cap_index(ev, L, args.max_wait)
    S = len(ev)
    thetas = [float(t) for t in args.thetas.split()]
    rows = []

    # 基準：区間検出まで待つ（規則なし）
    evaluate("区間検出まで待つ", np.full(len(lab), S), None, P, lab, L, ev, cap, rows)
    # 従来：最大確率 τ
    mx = P.max(-1)
    for tau in thetas:
        ok = mx >= tau
        evaluate(f"単一τ={tau:.2f}", np.where(ok.any(0), ok.argmax(0), S), None,
                 P, lab, L, ev, cap, rows)
    # 従来：完了確率 τ（完了だけ早く確定・他は区間検出まで待つ）
    for tau in thetas:
        ok = P[..., 1] >= tau
        evaluate(f"完了τ={tau:.2f}", np.where(ok.any(0), ok.argmax(0), S),
                 np.ones(len(lab), dtype=int), P, lab, L, ev, cap, rows)
    # 従来：単一 τ ＋ 経過時間ゲート
    for gate in [0.5, 0.75, 1.0]:
        for tau in [0.8, 0.9, 0.95]:
            ok = (mx >= tau) & (ev[:, None] >= gate)
            evaluate(f"τ={tau:.2f}+ゲート{gate}", np.where(ok.any(0), ok.argmax(0), S), None,
                     P, lab, L, ev, cap, rows)
    # 提案：先読み p(h) ≥ θ → 完了で確定
    for k, h in enumerate(hn):
        for tau in thetas:
            ok = H[..., k] >= tau
            evaluate(f"先読み h={h:g} θ={tau:.2f}", np.where(ok.any(0), ok.argmax(0), S),
                     np.ones(len(lab), dtype=int), P, lab, L, ev, cap, rows)

    print(f"モデル {args.stem} / {dset} {len(lab)} 区間 / 発話長 平均 {L.mean():.2f} 秒 / "
          f"区間検出 {args.max_wait} 秒" + ("  【無音込み】" if args.sil else "  【無音なし】"))
    print("応答遅延 = 確定時刻 − 発話末（正なら待ちすぎ、負なら発話中に確定）\n")
    print(f"{'停止ルール':>22}{'遅延平均':>9}{'完了の遅延':>10}{'取りこぼし':>10}"
          f"{'誤割り込み':>10}{'早切り':>8}{'macroF1':>9}{'完了F1':>8}")
    for nm, m_, le, ms, fa, ea, mac, f1 in rows:
        print(f"{nm:>22}{m_:9.3f}{le:10.3f}{ms:10.3f}{fa:10.3f}{ea:8.3f}{mac:9.4f}{f1:8.3f}")


if __name__ == "__main__":
    main()
