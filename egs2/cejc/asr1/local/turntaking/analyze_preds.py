#!/usr/bin/env python
"""tt_preds/*.tsv（区間ごとの確率ダンプ）から、閾値に依らない指標と較正後 F1 を出す。

背景: argmax の 3 クラス F1 は「ある 1 点での性能」しか測らない。実測では、同じモデルの
中でチェックポイントを変えただけで二値分離が 0.639→0.677 と動き、手法間の差 0.634〜0.667
より大きかった。つまり手法比較が動作点の偶然に汚染されている。そこで

  AUC        … 閾値に依らないランキングの良さ。「情報を持っているか」
  較正後 F1  … train_dev でクラス別バイアスを合わせた上での 3 クラス F1。
               「公平な動作点でどれだけ当てられるか」

の 2 つを出す。GPU は使わない（ダンプ済み TSV を読むだけ）。

  python local/turntaking/analyze_preds.py tt_preds/pool_attn_same_n5_valid.loss.ave.tsv
  python local/turntaking/analyze_preds.py tt_preds/pool_*_same_n5_valid.loss.ave.tsv
  python local/turntaking/analyze_preds.py --dev tt_preds/pool_attn_same_n5_train_dev_*.tsv \
      tt_preds/pool_attn_same_n5_valid.loss.ave.tsv
"""
import argparse
from pathlib import Path

import numpy as np

NAMES = ["継続", "終了", "相槌"]


def load(path):
    """TSV -> (labels (n,), probs (n,3))"""
    lab, prob = [], []
    with open(path, encoding="utf-8") as f:
        next(f)                                   # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            lab.append(int(p[1]))
            prob.append([float(p[3]), float(p[4]), float(p[5])])
    return np.array(lab), np.array(prob)


def auc(score, pos):
    """順位和による AUC。score:(n,) pos:(n,) bool。同点は平均順位で扱う。"""
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    s = np.sort(score)
    i = 0                                          # 同点をまとめて平均順位に
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def prf(lab, pred):
    """3 クラスの prec/recall/F1・macro-F1・継続/終了 二値分離 acc。"""
    conf = np.zeros((3, 3), dtype=int)
    for y, p in zip(lab, pred):
        conf[y, p] += 1
    rec = [conf[c, c] / max(conf[c].sum(), 1) for c in range(3)]
    pre = [conf[c, c] / max(conf[:, c].sum(), 1) for c in range(3)]
    f1 = [2 * pre[c] * rec[c] / max(pre[c] + rec[c], 1e-8) for c in range(3)]
    m = np.isin(lab, [0, 1])
    sep = float((pred[m] == lab[m]).mean()) if m.any() else float("nan")
    return pre, rec, f1, float(np.mean(f1)), float(np.trace(conf) / max(conf.sum(), 1)), sep


def fit_bias(lab, prob, grid=np.arange(-3.0, 3.01, 0.1)):
    """dev の macro-F1 を最大にするクラス別バイアス b を座標上昇で求める（b[1]=0 固定）。

    argmax(log p + b) で決める。b は「継続/相槌をどれだけ出しやすくするか」に相当し、
    dev と eval のクラス比のずれや重み付き CE によるキャリブレーションのずれを吸収する。
    """
    logp = np.log(np.clip(prob, 1e-12, None))
    b = np.zeros(3)
    best = prf(lab, (logp + b).argmax(1))[3]
    for _ in range(3):                              # 収束は速いので 3 巡で十分
        for c in (0, 2):
            cur = b[c]
            for v in grid:
                b[c] = v
                s = prf(lab, (logp + b).argmax(1))[3]
                if s > best:
                    best, cur = s, v
            b[c] = cur
    return b, best


def show(tag, lab, prob, bias=None):
    logp = np.log(np.clip(prob, 1e-12, None))
    pred = (logp + (bias if bias is not None else 0)).argmax(1)
    pre, rec, f1, macro, acc, sep = prf(lab, pred)

    # one-vs-rest AUC（クラス別 F1 と 1 対 1 で対応する閾値なし版）
    ovr = [auc(prob[:, c], lab == c) for c in range(3)]
    # 継続 vs 終了 AUC（既存の「二値分離 acc」の閾値なし版）
    m = np.isin(lab, [0, 1])
    ce = auc(prob[m, 0] - prob[m, 1], lab[m] == 0)

    head = f"{tag}" + ("" if bias is None else f"  [較正 b={np.round(bias, 2).tolist()}]")
    print(f"\n=== {head}")
    print(f"  n={len(lab)}  acc={acc:.3f}  macro-F1={macro:.3f}  "
          f"継続/終了 二値分離={sep:.3f}")
    print(f"  {'class':>6} | {'prec':>6} {'recall':>6} {'F1':>6} | {'AUC(1vR)':>9}")
    for c in range(3):
        print(f"  {NAMES[c]:>6} | {pre[c]:6.3f} {rec[c]:6.3f} {f1[c]:6.3f} | {ovr[c]:9.3f}")
    print(f"  macro-AUC = {np.nanmean(ovr):.3f}   継続 vs 終了 AUC = {ce:.3f}")
    return dict(tag=tag, macro=macro, sep=sep, ce_auc=ce,
                macro_auc=float(np.nanmean(ovr)), f1=f1, ovr=ovr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+", help="eval の予測ダンプ")
    ap.add_argument("--dev", default="", help="同じモデルの train_dev ダンプ（較正に使う）")
    args = ap.parse_args()

    bias = None
    if args.dev:
        dl, dp = load(args.dev)
        bias, dmacro = fit_bias(dl, dp)
        print(f"[較正] dev={Path(args.dev).name}  b={np.round(bias, 2).tolist()}  "
              f"dev macro-F1={dmacro:.3f}")

    rows = []
    for t in args.tsv:
        lab, prob = load(t)
        rows.append(show(Path(t).stem, lab, prob))
        if bias is not None:
            show(Path(t).stem, lab, prob, bias)

    if len(rows) > 1:
        print("\n=== 継続 vs 終了 AUC 順（閾値に依らない判別能力）")
        print(f"  {'AUC':>6} {'macroAUC':>9} {'macroF1':>8} {'二値分離':>8}  手法")
        for r in sorted(rows, key=lambda x: -x["ce_auc"]):
            print(f"  {r['ce_auc']:6.3f} {r['macro_auc']:9.3f} {r['macro']:8.3f} "
                  f"{r['sep']:8.3f}  {r['tag']}")


if __name__ == "__main__":
    main()
