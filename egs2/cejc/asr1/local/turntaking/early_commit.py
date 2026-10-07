#!/usr/bin/env python
"""早期確定：信頼度閾値による停止ルールを、ダンプ済み確率からオフライン評価する。

発話を途中で切った確率ダンプ（tt_preds/..._t<t>_*.tsv）を時系列順に並べ、
「信頼度が τ を超えたらその予測で確定して以降は聞かない」という規則を適用する。
GPU は使わない。

  時系列の並び: t が大きい = 発話末から遠い = 早い時点
    t=0.6 → 0.4 → 0.3 → 0.2 → 0.1 → 0.0（発話全体）

τ は時刻に依存しないので、この規則は**実行時に発話末を知らなくても適用できる**
（システムは自分の伸びていく音声を順に見て、閾値を超えた時点で打ち切るだけ）。
一方、評価上の「何秒早く確定できたか」は発話末を基準に測る。

  python local/turntaking/early_commit.py --stem pool_max_same_n5
  python local/turntaking/early_commit.py --stem pool_max_same_n5 --conf margin
"""
import argparse
from pathlib import Path

import numpy as np

NAMES = ["継続", "完了", "相槌"]


def load_tsv(path):
    """utt / ラベル / 確率 / n_samp（その時点までに聞いた長さ＝経過時間）"""
    utt, lab, prob, ns = [], [], [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            utt.append(p[0]); lab.append(int(p[1]))
            prob.append([float(p[3]), float(p[4]), float(p[5])])
            ns.append(int(p[6]) if len(p) > 6 else 0)
    return utt, np.array(lab), np.array(prob), np.array(ns)


def prf(lab, pred):
    conf = np.zeros((3, 3), dtype=int)
    for y, p in zip(lab, pred):
        conf[y, p] += 1
    rec = [conf[c, c] / max(conf[c].sum(), 1) for c in range(3)]
    pre = [conf[c, c] / max(conf[:, c].sum(), 1) for c in range(3)]
    f1 = [2 * pre[c] * rec[c] / max(pre[c] + rec[c], 1e-8) for c in range(3)]
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    return f1, float(np.mean(f1)), float(np.mean(rec)), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_max_same_n5", help="tt_preds のファイル幹")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--times", default="0.6,0.4,0.3,0.2,0.1,0.0",
                    help="早い順（発話末からの秒数、降順）")
    ap.add_argument("--conf", choices=["max", "margin"], default="max",
                    help="信頼度: max=最大確率 / margin=1位と2位の差")
    ap.add_argument("--min_elapsed", type=float, default=0.0,
                    help="発話開始から最低これだけ聞くまで確定しない（秒）。因果的に適用可能")
    args = ap.parse_args()

    times = [float(t) for t in args.times.split(",")]
    P, NS, base_utt, lab = [], [], None, None
    for t in times:
        sfx = "" if t == 0.0 else f"_t{t:g}"
        f = Path("tt_preds") / f"{args.stem}{sfx}_{args.ckpt}.tsv"
        u, y, p, ns = load_tsv(f)
        if base_utt is None:
            base_utt, lab = u, y
            order = {x: i for i, x in enumerate(u)}
        else:                                    # utt id で整列（順序が違っても合わせる）
            assert len(u) == len(base_utt), \
                f"{f}: 区間数 {len(u)} が基準 {len(base_utt)} と不一致（ダンプ取り直しが必要）"
            idx = np.array([order[x] for x in u])
            q = np.zeros_like(p); q[idx] = p; p = q
            q2 = np.zeros_like(ns); q2[idx] = ns; ns = q2
        P.append(p); NS.append(ns)
    P = np.stack(P); NS = np.stack(NS)                              # (S, N, 3) 早い順
    S, N, _ = P.shape
    conf = P.max(-1) if args.conf == "max" else (np.sort(P, -1)[..., 2] - np.sort(P, -1)[..., 1])
    pred = P.argmax(-1)
    # 経過時間ゲート: 発話開始から min_elapsed 秒聞くまでは確定させない。
    # 発話末の位置は実行時に分からないが、経過時間は分かるので因果的に適用できる。
    gate = NS >= args.min_elapsed * 16000

    print(f"モデル {args.stem} / 信頼度 {args.conf} / 区間 {N}"
          + (f" / 経過時間ゲート {args.min_elapsed}秒" if args.min_elapsed > 0 else ""))
    print(f"スナップショット（早い順）: {times}\n")
    print(f"{'τ':>6}{'平均短縮(秒)':>13}{'早期確定率':>11}{'継続F1':>8}{'完了F1':>8}"
          f"{'相槌F1':>8}{'macroF1':>9}{'UAR':>7}{'acc':>7}")
    for tau in [0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]:
        ok = (conf >= tau) & gate                 # (S, N)
        # 最初に閾値を超えたスナップショット。どこも超えなければ最後（発話全体）
        first = np.where(ok.any(0), ok.argmax(0), S - 1)
        j = np.arange(N)
        pr = pred[first, j]
        saved = np.array(times)[first]            # 発話末より何秒早く確定したか
        f1, mac, uar, acc = prf(lab, pr)
        early = float((first < S - 1).mean())
        lab_t = "τ なし" if tau > 1 else f"{tau:.2f}"
        print(f"{lab_t:>6}{saved.mean():13.3f}{early:11.3f}{f1[0]:8.3f}{f1[1]:8.3f}"
              f"{f1[2]:8.3f}{mac:9.3f}{uar:7.3f}{acc:7.3f}")

    print("\n参考: 各スナップショット単独で全区間を確定した場合")
    print(f"{'手前(秒)':>8}{'継続F1':>8}{'完了F1':>8}{'相槌F1':>8}{'macroF1':>9}{'UAR':>7}{'acc':>7}")
    for k, t in enumerate(times):
        f1, mac, uar, acc = prf(lab, pred[k])
        print(f"{t:>8}{f1[0]:8.3f}{f1[1]:8.3f}{f1[2]:8.3f}{mac:9.3f}{uar:7.3f}{acc:7.3f}")


if __name__ == "__main__":
    main()
