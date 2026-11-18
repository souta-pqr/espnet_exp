#!/usr/bin/env python
"""後から当てはめた採点で，完了の区間（次まで 2 秒以上）に応答しなかったものの内訳（2026-11-18）．
  python rt_miss_soma.py <data> <silero> <書き出し> g1,th1,g2,th2,S,phi"""
import sys
import numpy as np
from rt_rule import Data, FREE
D = Data(*sys.argv[1:4]); r = tuple(float(v) for v in sys.argv[4].split(","))
ev = D.events(r); t, se, cond = ev; o = np.argsort(t); t, se = t[o], se[o]
b, e, hor = D.b, D.e, D.hor
i1 = np.searchsorted(t, e, "left") - 1
mid = (i1 >= 0) & (t[np.maximum(i1, 0)] >= b) & (se[np.maximum(i1, 0)] >= b)
i2 = np.maximum(np.searchsorted(t, e, "left"), np.searchsorted(se, b, "left")); i2c = np.minimum(i2, len(t) - 1)
ok = (i2 < len(t)) & (t[i2c] <= hor)
st = D.heard & (D.lab == 1) & (D.g >= FREE) & ~mid
miss = st & ~ok
print(f"数える区間 {st.sum()}  応答しなかった {miss.sum()} ({miss.sum()/st.sum():.1%})")
# 区間末から窓の終わりまでの無音フレーム
L = np.searchsorted(D.t, e); R = np.searchsorted(D.t, hor, "right")
def feat(i):
    s = slice(L[i], R[i]); run, sc, pb, rid = D.run[s], D.sc[s], D.pb[s], D.rid[s]
    k = D.se_all[rid] >= b[i]          # 区間の中から始まった無音だけ
    run, sc, pb = run[k], sc[k], pb[k]
    if not len(run): return 0.0, -1.0, 0.0
    return run.max(), sc.max(), pb[run >= 0.4].max(initial=0)
F = np.array([feat(i) for i in np.flatnonzero(st)]); m = miss[st]
dur = (e - b)[st]
print("== 応答しなかった区間の内訳")
mr, ms, mp = F[m].T
a = mr < r[0]; print(f"区間末の後に {r[0]} 秒以上の無音が無い\t{a.sum()}\t{a.mean():.1%}")
for lo, hi in ((-1, 0.4), (0.4, 0.6), (0.6, r[1])):
    x = ~a & (ms >= lo) & (ms < hi); print(f"無音はあり，スコアの最大 {lo}〜{hi}\t{x.sum()}\t{x.mean():.1%}")
x = ~a & (ms >= r[1]); print(f"無音はあり，スコアの最大 {r[1]} 以上（無音が {r[0]} 秒になる前だけ）\t{x.sum()}\t{x.mean():.1%}")
print(f"無音ありの中で，相槌の確率の最大の中央値 {np.median(mp[~a]):.2f}  0.5 以上 {np.mean(mp[~a] >= 0.5):.1%}")
print("== 区間の長さごとの応答しなかった割合")
for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 99)):
    x = (dur >= lo) & (dur < hi); print(f"{lo}〜{hi} 秒\t{x.sum()}\t{m[x].mean():.1%}")
print("== 後に区間が無い（対話の最後）", int((D.g[st] == np.inf).sum()), f"{m[D.g[st] == np.inf].mean():.1%}")
if len(sys.argv) > 5:   # 数えた区間ごとに，応答しなかったか・スコアの最大・相槌の確率の最大を書き出す
    from collections import defaultdict
    seg = defaultdict(list); tag = {l.split()[0] for l in open(f"{sys.argv[1]}/tag")}
    for l in open(f"{sys.argv[1]}/segments"):
        u, rec, b0, e0 = l.split(); seg[rec].append((float(b0), float(e0), u))
    U = [u for rec in D.recs for _, _, u in sorted(seg[rec]) if u in tag]
    with open(sys.argv[5], "w") as fo:
        for j, i in enumerate(np.flatnonzero(st)): fo.write(f"{U[i]}\t{int(m[j])}\t{F[j,1]:.3f}\t{F[j,2]:.3f}\n")
