#!/usr/bin/env python
"""早期確定：クラス別の信頼度閾値 τ=(τ継続, τ完了, τ相槌) をオフライン評価する。

early_commit.py の単一 τ を拡張した実験。動機は 2 つ。
  1. 運用上、早く確定したいのは「完了」（応答開始）と「相槌」で、
     「継続」を早く確定しても行動は変わらない（聞き続けるだけ）。
  2. 早く切るほど相槌を過剰予測するバイアスがあるため、
     相槌だけ閾値を上げる余地がある。

規則: 各時点で argmax クラス c の確率が τ_c を超えたら確定。
τ_c は時刻に依存しないので単一 τ と同じく因果的に適用できる。

選択と評価を分ける:
  - τ の組は train_dev で選ぶ（目標短縮量以上の中で macro-F1 最大）
  - 数字は eval で報告し、同じ選択規則で選んだ単一 τ と比較する
  - さらに eval の単一 τ パレート曲線を同短縮量で線形補間した値も併記

  python local/turntaking/early_commit_classwise.py --stem pool_attn-trunc03_same_n5
  python local/turntaking/early_commit_classwise.py --stem pool_attn-trunc-new_same_n5 --min_elapsed 0.7
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

from early_commit import load_tsv, prf

NAMES = ["継続", "完了", "相槌"]


def load_snapshots(stem, ckpt, times):
    """時刻グリッドのダンプを (S,N,3)/(S,N) に積む。utt id で整列する。"""
    P, NS, base_utt, lab = [], [], None, None
    for t in times:
        sfx = "" if t == 0.0 else f"_t{t:g}"
        f = Path("tt_preds") / f"{stem}{sfx}_{ckpt}.tsv"
        u, y, p, ns = load_tsv(f)
        if base_utt is None:
            base_utt, lab = u, y
            order = {x: i for i, x in enumerate(u)}
        else:
            assert len(u) == len(base_utt), \
                f"{f}: 区間数 {len(u)} が基準 {len(base_utt)} と不一致（ダンプ取り直しが必要）"
            idx = np.array([order[x] for x in u])
            q = np.zeros_like(p); q[idx] = p; p = q
            q2 = np.zeros_like(ns); q2[idx] = ns; ns = q2
        P.append(p); NS.append(ns)
    return np.stack(P), np.stack(NS), np.array(lab)


def apply_rule(P, NS, lab, times_arr, tau_vec, gate_samp):
    """クラス別閾値 tau_vec=(3,) の停止ルールを適用して指標を返す。"""
    S, N, _ = P.shape
    pred = P.argmax(-1)                                   # (S,N)
    maxp = P.max(-1)                                      # (S,N)
    ok = maxp >= tau_vec[pred]
    if gate_samp > 0:
        ok &= NS >= gate_samp
    first = np.where(ok.any(0), ok.argmax(0), S - 1)
    j = np.arange(N)
    pr = pred[first, j]
    saved = times_arr[first]
    f1, mac, uar, acc = prf(lab, pr)
    # 真クラス別の平均短縮（完了の短縮＝応答遅延の短縮そのもの）
    saved_c = [float(saved[lab == c].mean()) for c in range(3)]
    return dict(saved=float(saved.mean()), saved_c=saved_c,
                f1=f1, mac=mac, uar=uar, acc=acc,
                early=float((first < S - 1).mean()))


def pareto_interp(points, x):
    """単一 τ の (saved, macroF1) 点群から「短縮量 >= x で得られる最大 macro-F1」を返す。"""
    arr = np.array(sorted(points))
    sv, mf = arr[:, 0], arr[:, 1]
    best_from = np.maximum.accumulate(mf[::-1])[::-1]     # saved>=sv[i] で得られる最大 F1
    i = np.searchsorted(sv, x, side="left")
    if i >= len(sv):
        return None
    return float(best_from[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-trunc03_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--times", default="0.6,0.4,0.3,0.2,0.1,0.0")
    ap.add_argument("--min_elapsed", type=float, default=0.0,
                    help="経過時間ゲート（秒）。因果的に適用可能")
    ap.add_argument("--targets", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5")
    args = ap.parse_args()

    times = [float(t) for t in args.times.split(",")]
    times_arr = np.array(times)
    gate = int(args.min_elapsed * 16000)
    targets = [float(t) for t in args.targets.split(",")]

    dev_P, dev_NS, dev_lab = load_snapshots(f"{args.stem}_train_dev", args.ckpt, times)
    ev_P, ev_NS, ev_lab = load_snapshots(args.stem, args.ckpt, times)
    print(f"モデル {args.stem} / dev {dev_P.shape[1]} 区間 / eval {ev_P.shape[1]} 区間"
          + (f" / 経過時間ゲート {args.min_elapsed}秒" if gate else ""))
    print(f"スナップショット（早い順）: {times}\n")

    # --- 候補を dev で総当たり ---------------------------------------------
    grid_c = [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
              0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
              0.96, 0.97, 0.98, 0.99, 0.995,
              1.01]   # 1.01 = そのクラスでは確定しない。高 τ 側は運用点が細かいので刻みも細かく
    grid_s = np.arange(0.34, 1.0001, 0.005).tolist() + [1.01]

    dev_cls = {}
    for tau in itertools.product(grid_c, repeat=3):
        r = apply_rule(dev_P, dev_NS, dev_lab, times_arr, np.array(tau), gate)
        dev_cls[tau] = (r["saved"], r["mac"])
    dev_sgl = {}
    for t in grid_s:
        tau = (t, t, t)
        r = apply_rule(dev_P, dev_NS, dev_lab, times_arr, np.array(tau), gate)
        dev_sgl[tau] = (r["saved"], r["mac"])

    # eval 側の単一 τ パレート曲線（同短縮量での補間比較に使う）
    ev_sgl_pts = []
    for t in grid_s:
        r = apply_rule(ev_P, ev_NS, ev_lab, times_arr, np.array([t] * 3), gate)
        ev_sgl_pts.append((r["saved"], r["mac"]))

    # --- 目標短縮量ごとに dev で選び eval で報告 ---------------------------
    hdr = (f"{'目標':>5} {'τ(継続,完了,相槌)':>22} {'短縮':>6} {'継続短縮':>7} {'完了短縮':>7}"
           f" {'相槌短縮':>7} {'継続F1':>7} {'完了F1':>7} {'相槌F1':>7} {'macroF1':>8}"
           f" {'単一τ同短縮':>10} {'差':>7}")
    print(hdr)
    for tgt in targets:
        # dev: 短縮量 >= 目標の中で macro-F1 最大
        cand = [(mac, tau) for tau, (sv, mac) in dev_cls.items() if sv >= tgt]
        if not cand:
            continue
        _, tau = max(cand)
        r = apply_rule(ev_P, ev_NS, ev_lab, times_arr, np.array(tau), gate)
        base = pareto_interp(ev_sgl_pts, r["saved"])
        diff = "" if base is None else f"{r['mac'] - base:+7.4f}"
        base_s = "" if base is None else f"{base:10.4f}"
        tau_s = ",".join("--" if t > 1 else f"{t:.2f}" for t in tau)
        print(f"{tgt:5.2f} {tau_s:>22} {r['saved']:6.3f} {r['saved_c'][0]:7.3f}"
              f" {r['saved_c'][1]:7.3f} {r['saved_c'][2]:7.3f} {r['f1'][0]:7.3f}"
              f" {r['f1'][1]:7.3f} {r['f1'][2]:7.3f} {r['mac']:8.4f} {base_s} {diff}")

    # --- 参考: 単一 τ を同じ選択規則（dev 選択→eval 報告）で通した場合 -----
    print("\n参考: 単一 τ（dev 選択 → eval 報告）")
    print(f"{'目標':>5} {'τ':>6} {'短縮':>6} {'macroF1':>8}")
    for tgt in targets:
        cand = [(mac, tau) for tau, (sv, mac) in dev_sgl.items() if sv >= tgt]
        if not cand:
            continue
        _, tau = max(cand)
        r = apply_rule(ev_P, ev_NS, ev_lab, times_arr, np.array(tau), gate)
        print(f"{tgt:5.2f} {tau[0]:6.3f} {r['saved']:6.3f} {r['mac']:8.4f}")


if __name__ == "__main__":
    main()
