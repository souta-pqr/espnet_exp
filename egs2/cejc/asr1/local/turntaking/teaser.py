#!/usr/bin/env python
"""TEASER 型の二段停止ルールを、ダンプ済み確率からオフライン評価する。

Schäfer & Leser, "TEASER: early and accurate time series classification",
Data Mining and Knowledge Discovery 34, 2020.

  slave  … 各スナップショット長でクラス確率を出す分類器 ＝ 既存の学習済みモデル
  master … その予測を信用してよいかを判定する one-class SVM（スナップショットごと）
           train_dev で「slave が正解した事例の確率ベクトル」だけを内側として学習する
  さらに … 同じ予測が v 回連続で accept されるまで確定しない

信頼度閾値（early_commit.py）との違いは、master が「確信度がいくつ以上か」ではなく
**この確率ベクトルは正解時の分布に似ているか**を学習で判定する点。確率が全体的に
低めでも正解時の形をしていれば accept でき、逆に高くても異常な形なら待てる。

master の学習に eval は一切使わない（train_dev のみ）。

  python local/turntaking/teaser.py --stem pool_max-trunc_same_n5
"""
import argparse
from pathlib import Path

import numpy as np

NAMES = ["継続", "完了", "相槌"]


def load_tsv(path):
    utt, lab, prob = [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            utt.append(p[0]); lab.append(int(p[1]))
            prob.append([float(p[3]), float(p[4]), float(p[5])])
    return utt, np.array(lab), np.array(prob)


def stack(stem, ckpt, times, dev):
    """(S, N, 3) の確率と (N,) のラベル。utt id で整列する。"""
    P, lab, order = [], None, None
    for t in times:
        tsfx = "" if t == 0.0 else f"_t{t:g}"
        dsfx = "_train_dev" if dev else ""
        f = Path("tt_preds") / f"{stem}{dsfx}{tsfx}_{ckpt}.tsv"
        u, y, p = load_tsv(f)
        if lab is None:
            lab, order = y, {x: i for i, x in enumerate(u)}
        else:
            assert len(u) == len(lab), \
                f"{f}: 区間数 {len(u)} が基準 {len(lab)} と不一致（ダンプ取り直しが必要）"
            idx = np.array([order[x] for x in u])
            q = np.zeros_like(p); q[idx] = p; p = q
        P.append(p)
    return np.stack(P), lab


def prf(lab, pred):
    conf = np.zeros((3, 3), dtype=int)
    for y, p in zip(lab, pred):
        conf[y, p] += 1
    rec = [conf[c, c] / max(conf[c].sum(), 1) for c in range(3)]
    pre = [conf[c, c] / max(conf[:, c].sum(), 1) for c in range(3)]
    f1 = [2 * pre[c] * rec[c] / max(pre[c] + rec[c], 1e-8) for c in range(3)]
    return f1, float(np.mean(f1)), float(np.mean(rec)), float(np.trace(conf) / max(conf.sum(), 1))


def commit(accept, pred, v, times):
    """v 回連続で accept かつ同一予測になった時点で確定。無ければ最終時点。"""
    S, N = accept.shape
    first = np.full(N, S - 1)
    run = np.zeros(N, dtype=int)
    done = np.zeros(N, dtype=bool)
    prev = np.full(N, -1)
    for s in range(S):
        same = (pred[s] == prev) & (s > 0)
        # accept かつ予測が前回と同じなら連続数を伸ばし、予測が変わったら 1 から数え直す
        # （変わった時点も accept されていれば新しい連続の 1 回目に数える）
        run = np.where(accept[s], np.where(same, run + 1, 1), 0)
        prev = pred[s]
        hit = (~done) & (run >= v)
        first[hit] = s
        done |= hit
    j = np.arange(N)
    return pred[first, j], np.array(times)[first], first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_max-trunc_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--times", default="0.6,0.4,0.3,0.2,0.1,0.0", help="早い順（降順）")
    ap.add_argument("--nu", type=float, default=0.05, help="one-class SVM の nu")
    ap.add_argument("--fit_n", type=int, default=4000, help="master 学習に使う事例数の上限")
    args = ap.parse_args()

    from sklearn.svm import OneClassSVM
    times = [float(t) for t in args.times.split(",")]

    Pd, yd = stack(args.stem, args.ckpt, times, dev=True)      # master 学習用
    Pe, ye = stack(args.stem, args.ckpt, times, dev=False)     # 評価用
    S = len(times)
    rng = np.random.default_rng(0)

    # スナップショットごとに master を学習：slave が正解した事例の確率ベクトルを内側とする
    masters = []
    for s in range(S):
        ok = Pd[s].argmax(1) == yd
        X = Pd[s][ok]
        if len(X) > args.fit_n:
            X = X[rng.choice(len(X), args.fit_n, replace=False)]
        m = OneClassSVM(nu=args.nu, gamma="scale").fit(X)
        masters.append(m)
        print(f"  master s={s} (t={times[s]}): 学習 {len(X)} 件 / dev 正解率 {ok.mean():.3f}",
              flush=True)

    inlier = np.stack([masters[s].predict(Pe[s]) > 0 for s in range(S)])   # (S,N)
    srt = np.sort(Pe, -1)
    margin = srt[..., 2] - srt[..., 1]
    pred = Pe.argmax(-1)

    print(f"\nTEASER  モデル {args.stem} / 区間 {len(ye)} / nu={args.nu}")
    print(f"スナップショット（早い順）: {times}\n")
    print(f"{'v':>3}{'θ':>7}{'平均短縮':>10}{'早期率':>8}{'継続F1':>8}{'完了F1':>8}"
          f"{'相槌F1':>8}{'macroF1':>9}{'UAR':>7}{'acc':>7}")
    base = None
    for v in (1, 2, 3):
        for th in (0.0, 0.05, 0.10, 0.20, 0.30):
            acc_mask = inlier & (margin >= th)
            pr, saved, first = commit(acc_mask, pred, v, times)
            f1, mac, uar, acc = prf(ye, pr)
            early = float((first < S - 1).mean())
            if base is None:
                base = mac
            print(f"{v:>3}{th:>7.2f}{saved.mean():10.3f}{early:8.3f}{f1[0]:8.3f}{f1[1]:8.3f}"
                  f"{f1[2]:8.3f}{mac:9.3f}{uar:7.3f}{acc:7.3f}")

    print("\n参考: 全体を聞いた場合")
    f1, mac, uar, acc = prf(ye, pred[-1])
    print(f"{'':>10}{0.0:10.3f}{0.0:8.3f}{f1[0]:8.3f}{f1[1]:8.3f}{f1[2]:8.3f}"
          f"{mac:9.3f}{uar:7.3f}{acc:7.3f}")


if __name__ == "__main__":
    main()
