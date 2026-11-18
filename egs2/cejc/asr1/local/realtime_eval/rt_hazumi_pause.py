#!/usr/bin/env python
"""Hazumi で，ユーザの発話の途中の間ごとに「その間の中で応答してしまうか」を数える（2026-11-16）．rt_hazumi.py と同じ引数．
間＝ユーザの持ち時間の中で，声と声の間の無音（0.4 秒以上）．比べる相手として，最後の声の後（言い終わり）に同じ長さの時間内に応答する割合も出す．"""
import csv, os, sys, tempfile
from collections import defaultdict
import numpy as np
from rt_rule import Data, HOP, OFF

tsv, ddir = sys.argv[1:3]
ex = defaultdict(list)
for r in csv.DictReader(open(tsv), delimiter="\t"):
    ex[r["rec"]].append((float(r["sys_end"]), float(r["ex_end"])))
tmp = tempfile.mkdtemp(); os.makedirs(tmp + "/sil")
with open(tmp + "/segments", "w") as fs, open(tmp + "/tag", "w") as ft:
    for rec in sorted(ex):
        if not os.path.exists(f"{ddir}/{rec}.npz"): continue
        np.save(f"{tmp}/sil/{rec}.npy", np.load(f"{ddir}/{rec}.npz")["p"])
        for i, (a, c) in enumerate(ex[rec]):
            fs.write(f"{rec}_{i:04d} {rec} {a:.3f} {c:.3f}\n"); ft.write(f"{rec}_{i:04d} 1\n")
D = Data(tmp, tmp + "/sil", ddir)
BINS = [(0.4, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
P, E = [], []                                    # 途中の間（始まり，終わり），言い終わり（L，操作者の応答 c）
for k, rec in enumerate(D.recs):
    v = np.load(f"{ddir}/{rec}.npz")["p"] >= 0.5
    for a, c in ex[rec]:
        i0, i1 = int(a / HOP) + 3, int(c / HOP)
        vi = np.flatnonzero(v[i0:i1]) + i0
        if len(vi) < 10: continue
        vt = (vi + 1) * HOP + k * OFF
        j = np.flatnonzero(np.diff(vt) >= 0.4)
        for q in j:
            if (vt > vt[q]).sum() >= 10 and (vt <= vt[q]).sum() >= 10: P.append((vt[q], vt[q + 1]))
        E.append((vt[-1], k * OFF + c))
P = np.array(P); E = np.array(E); gp = P[:, 1] - P[:, 0]
print("途中の間", len(P), " 言い終わり", len(E))
print("名前\t" + "\t".join(f"間 {lo}〜{hi} 秒（{((gp >= lo) & (gp < hi)).sum()}）" for lo, hi in BINS) + "\t言い終わりの後 1 秒以内に応答\t2 秒以内")
for spec in sys.argv[3:]:
    name, r = spec.split("=")
    if name.startswith("sil"):
        x = float(r); f = np.minimum.reduceat(np.where(D.run >= x, np.arange(len(D.run)), len(D.run)), D.first); t = D.t[f[f < len(D.run)]]
    else:
        t = D.events(tuple(None if v in ("", "-") else float(v) for v in r.split(",")))[0]
    t = np.sort(t)
    n = lambda lo, hi: np.searchsorted(t, hi) - np.searchsorted(t, lo)      # lo〜hi の応答の数
    hit = n(P[:, 0], P[:, 1]) > 0
    out = [f"{hit[(gp >= lo) & (gp < hi)].mean():.1%}" for lo, hi in BINS]
    for w in (1.0, 2.0):
        u = E[:, 1] - E[:, 0] >= w                                         # 操作者がそれより早く応答したやりとりは除く
        out.append(f"{(n(E[u, 0], E[u, 0] + w) > 0).mean():.1%}（{u.sum()}）")
    print(name + " " + r + "\t" + "\t".join(out))
