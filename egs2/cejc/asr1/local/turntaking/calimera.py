#!/usr/bin/env python
"""Calimera 型（将来コストを回帰で推定する非近視眼的停止ルール）を評価する。

Economy（economy.py）は状態を離散化して表で将来コストを持つ。Calimera 系は
同じ枠組みで**将来コストを回帰器で推定**する。ECTS のベンチマーク
（Renault+, TMLR 2025）では両者とも上位だが、同論文は
「Calimera は将来コストを回帰推定するため較正が崩れると大きく劣化、
Proba Threshold は閾値 1 個なので影響が軽微」とも述べている。

本リポジトリでは別に、**dev で当てはめた決定規則は eval に転移しない**
（クラス比補正でも救えない ＝ 条件付き分布のずれ）ことが分かっている。
そこから素直に立つ予測は「当てはめの自由度が高い Calimera は Economy より転移が悪い」。
この予測を検証するのがこのスクリプトの目的で、**当たっても外れても情報がある**。

  python local/turntaking/calimera.py --stem pool_attn-trunc-new_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                      # noqa: E402
from early_commit_classwise import (load_snapshots, apply_rule,    # noqa: E402
                                    pareto_interp)


def features(P, NS):
    """回帰器の入力。事後確率（2 自由度）＋ margin ＋ 経過時間。すべて因果的に既知。"""
    srt = np.sort(P, -1)
    el = NS / 16000.0
    return np.stack([P[..., 0], P[..., 1], srt[..., 2] - srt[..., 1],
                     el, np.log1p(el)], axis=-1)


def fit_regressors(Pd, NSd, labd, C, n_leaves):
    """各スナップショット i について「i で止めた場合の誤分類コスト」を回帰で当てる。

    Economy が状態ごとの表引きなのに対し、こちらは連続特徴からの回帰。
    自由度が高いぶん dev に当てはまりやすく、その分だけ転移が悪くなるかを見る。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    S = Pd.shape[0]
    X = features(Pd, NSd)
    pred = Pd.argmax(-1)
    regs = []
    for i in range(S):
        y = C[labd, pred[i]]                    # i で止めたときの実際の誤分類コスト
        r = HistGradientBoostingRegressor(max_leaf_nodes=n_leaves, max_iter=200,
                                          learning_rate=0.08, random_state=0)
        r.fit(X[i], y)
        regs.append(r)
    return regs


def decide(Pe, NSe, times, regs, alpha):
    """各時点で「今止める推定コスト <= 将来のどこかで止める推定コストの最小」なら確定。

    将来コストは、現時点の特徴から各将来スナップショット用の回帰器で予測する
    （＝現在の状態から将来を見通す。非近視眼的）。
    """
    S = Pe.shape[0]
    X = features(Pe, NSe)
    idx = np.arange(S) / max(S - 1, 1)
    delay = alpha * idx
    est = np.stack([regs[j].predict(X[i]) for i in range(S) for j in range(S)])
    est = est.reshape(S, S, -1)                 # est[i, j] = 状態 i から見た「j で止める推定コスト」
    N = Pe.shape[1]
    first = np.full(N, S - 1)          # どこでも止まらなければ最終時点
    done = np.zeros(N, bool)
    for i in range(S - 1):
        now = est[i, i] + delay[i]
        fut = np.min(est[i, i + 1:] + delay[i + 1:, None], axis=0)
        hit = (~done) & (now <= fut)
        first[hit] = i; done |= hit
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-trunc-new_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--times", default="0.6,0.4,0.3,0.2,0.1,0.0")
    ap.add_argument("--leaves", type=int, default=31,
                    help="回帰器の複雑さ。小さいほど Economy の表引きに近づく")
    args = ap.parse_args()
    times = [float(x) for x in args.times.split(",")]
    ta = np.array(times)

    Pe, NSe, labe = load_snapshots(args.stem, args.ckpt, times)
    Pd, NSd, labd = load_snapshots(f"{args.stem}_train_dev", args.ckpt, times)
    C = 1.0 - np.eye(3)
    regs = fit_regressors(Pd, NSd, labd, C, args.leaves)

    base = [(r["saved"], r["mac"]) for r in
            (apply_rule(Pe, NSe, labe, ta, np.array([x] * 3), 0)
             for x in list(np.arange(0.34, 0.9801, 0.002)) + [0.99, 0.995, 0.999])]
    pred = Pe.argmax(-1); j = np.arange(len(labe))
    print(f"モデル {args.stem} / 区間 {len(labe)} / 回帰器 max_leaf_nodes={args.leaves}")
    print(f"{'alpha':>7}{'平均短縮':>10}{'継続F1':>8}{'完了F1':>8}{'相槌F1':>8}"
          f"{'macroF1':>9}{'単一τ同短縮':>12}{'差':>9}")
    for alpha in [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]:
        first = decide(Pe, NSe, times, regs, alpha)
        f1, mac, uar, acc = prf(labe, pred[first, j])
        sv = ta[first].mean()
        b = pareto_interp(base, sv)
        d = "" if b is None else f"{mac - b:+9.4f}"
        bs = "" if b is None else f"{b:12.4f}"
        print(f"{alpha:7.2f}{sv:10.3f}{f1[0]:8.3f}{f1[1]:8.3f}{f1[2]:8.3f}"
              f"{mac:9.4f}{bs}{d}")


if __name__ == "__main__":
    main()
