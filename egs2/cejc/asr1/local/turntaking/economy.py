#!/usr/bin/env python
"""Economy 型（非近視眼的・コスト考慮）の停止ルールを、ダンプ済み確率から評価する。

Dachraoui et al. (2015) / Achenchabe et al. (2021) の定式化。設計は cif/method_B_ECONOMY.md。
早期分類のベンチマーク（Renault+, TMLR 2025）が最上位とする「非近視眼的 × コスト考慮」の
象限にあたる。本リポジトリの既存の停止ルール（信頼度閾値・TEASER・クラス別τ・ゲート）は
すべて近視眼的・コスト非考慮なので、その外側を初めて試すことになる。

  τ = min{ i : E[L | 今 i で決める, s_i] <= min_{i'>i} E[L | i' で決める, s_i] }

状態 s は (予測クラス, margin ビン, 経過時間ビン) で離散化する。**発話末からの距離は
使わない** —— 実行時に未知であり、時刻ごとの較正はそれで因果性を失っていたため。
統計量は「状態 g にいた系列が時刻 j で示す (真ラベル, 予測) の同時分布」という
**コスト非依存**の形で train_dev から推定し、C_mis と alpha は推論時に与える。
これによりコストを掃引しても再推定が要らない（Achenchabe の cost-agnostic 推定）。

  python local/turntaking/economy.py --stem pool_attn-trunc-new_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                      # noqa: E402
from early_commit_classwise import (load_snapshots, apply_rule,    # noqa: E402
                                    pareto_interp)

MARGIN_EDGES = np.array([0.0, .1, .2, .35, .5, .7, .85, 1.01])
ELAPSED_EDGES = np.array([0, .5, .75, 1.0, 1.25, 1.5, 2.0, 3.0, 1e9])
MIN_GROUP = 20


def states(P, NS):
    """(S,N) の状態 id と状態数。予測クラス × margin ビン × 経過時間ビン。"""
    pred = P.argmax(-1)
    srt = np.sort(P, -1)
    mb = np.clip(np.digitize(srt[..., 2] - srt[..., 1], MARGIN_EDGES) - 1,
                 0, len(MARGIN_EDGES) - 2)
    eb = np.clip(np.digitize(NS / 16000.0, ELAPSED_EDGES) - 1,
                 0, len(ELAPSED_EDGES) - 2)
    nm, ne = len(MARGIN_EDGES) - 1, len(ELAPSED_EDGES) - 1
    return (pred * nm + mb) * ne + eb, 3 * nm * ne


def fit_tables(Pd, NSd, labd):
    """M[i,g,j] = 状態(i,g)にいた系列の、時刻 j における (真ラベル, 予測) 同時分布。

    コストを含まないので C_mis / alpha を後から自由に振れる。
    n[i,g] は状態の標本数（薄い状態を捨てるのに使う）。
    """
    sd, G = states(Pd, NSd)
    S = Pd.shape[0]
    predj = Pd.argmax(-1)
    M = np.zeros((S, G, S, 3, 3))
    n = np.zeros((S, G))
    for i in range(S):
        for g in np.unique(sd[i]):
            m = sd[i] == g
            cnt = int(m.sum())
            n[i, g] = cnt
            y = labd[m]
            for j in range(i, S):
                np.add.at(M[i, g, j], (y, predj[j, m]), 1.0)
            M[i, g, i:] /= cnt
    return M, n, G


def decide(Pe, NSe, times, M, n, G, C, alpha):
    """状態ごとにトリガ表を作り、各区間の停止時刻を決める。"""
    S = Pe.shape[0]
    se, _ = states(Pe, NSe)
    idx = np.arange(S) / max(S - 1, 1)
    delay = alpha * idx                       # 遅く決めるほど高い（i が大きい＝発話全体寄り）
    ecost = np.einsum("igjab,ab->igj", M, C) + delay[None, None, :]
    ok = np.zeros((S, G), dtype=bool)
    for i in range(S):
        for g in range(G):
            if n[i, g] < MIN_GROUP:
                continue
            fut = [ecost[i, g, j] for j in range(i + 1, S) if n[i, g] >= MIN_GROUP]
            ok[i, g] = ecost[i, g, i] <= (min(fut) if fut else np.inf)
    trig = ok[np.arange(S)[:, None], se]      # (S,N)
    first = np.where(trig.any(0), trig.argmax(0), S - 1)
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-trunc-new_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--times", default="0.6,0.4,0.3,0.2,0.1,0.0")
    ap.add_argument("--asym", action="store_true",
                    help="完了の取りこぼし（応答遅れ）を重くした非対称コストを使う")
    args = ap.parse_args()
    times = [float(x) for x in args.times.split(",")]

    Pe, NSe, labe = load_snapshots(args.stem, args.ckpt, times)
    Pd, NSd, labd = load_snapshots(f"{args.stem}_train_dev", args.ckpt, times)
    M, n, G = fit_tables(Pd, NSd, labd)

    ta = np.array(times)
    base = [(r["saved"], r["mac"]) for r in
            (apply_rule(Pe, NSe, labe, ta, np.array([x] * 3), 0)
             for x in list(np.arange(0.34, 0.9801, 0.002)) + [0.99, 0.995, 0.999])]

    C = 1.0 - np.eye(3)
    if args.asym:                              # 真=完了 を取り逃す誤りを 2 倍に
        C[1, :] *= 2.0
        C[1, 1] = 0.0
    pred = Pe.argmax(-1)
    print(f"モデル {args.stem} / 区間 {Pe.shape[1]} / 状態数 {G} / "
          f"コスト {'非対称' if args.asym else '対称'}")
    print(f"{'alpha':>7}{'平均短縮':>10}{'早期率':>8}{'継続F1':>8}{'完了F1':>8}{'相槌F1':>8}"
          f"{'macroF1':>9}{'単一τ同短縮':>12}{'差':>9}")
    for alpha in [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]:
        first = decide(Pe, NSe, times, M, n, G, C, alpha)
        j = np.arange(len(labe))
        f1, mac, uar, acc = prf(labe, pred[first, j])
        sv = ta[first].mean()
        b = pareto_interp(base, sv)
        d = "" if b is None else f"{mac - b:+9.4f}"
        bs = "" if b is None else f"{b:12.4f}"
        print(f"{alpha:7.2f}{sv:10.3f}{(first < len(times)-1).mean():8.3f}"
              f"{f1[0]:8.3f}{f1[1]:8.3f}{f1[2]:8.3f}{mac:9.4f}{bs}{d}")


if __name__ == "__main__":
    main()
