#!/usr/bin/env python
"""オフラインのスコア（区間の頭から入れたもの）で，完了の区間（次の区間まで 2 秒以上）の区間末の後 2 秒までのスコアの最大を調べる（2026-11-13）"""
import sys, os
from collections import defaultdict
import numpy as np, pandas as pd
A = os.path.expanduser("~/2025/espnet/egs2/cejc/asr1"); d, tsv = sys.argv[1:3]
seg = defaultdict(list); dur = {}
for l in open(f"{A}/data/{d}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u)); dur[u] = float(e) - float(b)
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{A}/data/{d}/tag")}
gap = {}
for rec, lst in seg.items():
    lst.sort()
    for j, (b, e, u) in enumerate(lst): gap[u] = lst[j + 1][0] - e if j + 1 < len(lst) else 9.0
df = pd.read_csv(tsv, sep="\t")
df["d"] = df.utt.map(dur); df = df[(df.elapsed > df.d) & (df.elapsed <= df.d + 2.0)]
mx = df.groupby("utt").p_end.max()
us = [u for u in mx.index if tag.get(u) == 1 and gap[u] >= 2]
v = mx[us].values; D = np.array([dur[u] for u in us])
print(f"完了の区間 {len(v)}：スコア最大が 0.82 未満 {np.mean(v < 0.82):.1%}・0.5 未満 {np.mean(v < 0.5):.1%}・中央値 {np.median(v):.2f}")
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 99)):
    m = (D >= lo) & (D < hi)
    print(f"  区間の長さ {lo}〜{hi}：{m.sum():6d}  {np.mean(v[m] < 0.82):.1%}")
