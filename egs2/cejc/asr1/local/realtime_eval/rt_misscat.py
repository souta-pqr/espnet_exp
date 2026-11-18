#!/usr/bin/env python
"""完了の区間（次まで 2 秒以上）を，区間末の後 0.5 秒までに 0.4 秒以上の無音が始まるかで分け，応答しなかった割合を方式ごとに出す（2026-11-13）
  python rt_misscat.py <data> <silero> <書き出し> <決め方> <rt_eval の出力>..."""
import sys, glob, os
import numpy as np
from rt_rule import Data, real_events, HOP, OFF, FREE
data, sdir, ddir, rule = sys.argv[1:5]
D = Data(data, sdir, ddir)
# 区間ごと：区間末 −0.3〜+0.5 秒に始まり 0.4 秒以上続く無音があるか
has = np.zeros(len(D.b), bool); late = np.full(len(D.b), np.nan)
ST, EN = [], []
for k, rec in enumerate(D.recs):
    d = np.load(f"{ddir}/{rec}.npz"); sil = d["p"] < 0.5
    st = np.flatnonzero(sil & ~np.r_[False, sil[:-1]]) * HOP; en = np.flatnonzero(~sil & np.r_[False, sil[:-1]]) * HOP
    if len(en) < len(st): en = np.r_[en, len(sil) * HOP]
    L = en[:len(st)] - st; ok = L >= 0.4
    ST.append(st[ok] + k * OFF); EN.append(en[:len(st)][ok] + k * OFF)
ST = np.concatenate(ST)
i = np.searchsorted(ST, D.e - 0.3)
has = (i < len(ST)) & (ST[np.minimum(i, len(ST) - 1)] <= D.e + 0.5)
def per_seg(ev):
    t, se, cond = ev; o = np.argsort(t); t, se = t[o], se[o]
    b, e, hor = D.b, D.e, D.hor
    i1 = np.searchsorted(t, e, "left") - 1
    mid = (i1 >= 0) & (t[np.maximum(i1, 0)] >= b) & (se[np.maximum(i1, 0)] >= b)
    i2 = np.maximum(np.searchsorted(t, e, "left"), np.searchsorted(se, b, "left"))
    ok = i2 < len(t); i2c = np.minimum(i2, len(t) - 1); ok &= t[i2c] <= hor
    return mid, np.where(ok, t[i2c] - e, np.inf)
st = D.heard & (D.lab == 1) & (D.g >= FREE)
def show(name, ev):
    mid, lt = per_seg(ev); m = st & ~mid
    for nm, c in (("区間末に無音あり", has), ("区間末に無音なし", ~has)):
        x = lt[m & c]
        print(f"{name:28s} {nm}：{len(x):6d}  応答しなかった {np.mean(~np.isfinite(x)):.1%}  latency の中央値 {np.median(x):.2f}")
r = tuple(None if v in ("", "-") else float(v) for v in rule.split(","))
show("後から当てはめた " + rule, D.events(r))
for o in sys.argv[5:]: show(o, real_events(D, o))
