#!/usr/bin/env python
"""応答しなかった割合の上限を変えて，開発データで latency が最小の決め方を並べる（2026-11-13）．rt_rule.py と同じ場所で動かす．
  python rt_front.py <data> <silero> <書き出し> <評価 data> <silero> <書き出し>"""
import itertools, sys
import numpy as np
from rt_rule import Data, fmt
D = Data(*sys.argv[1:4]); E = Data(*sys.argv[4:7])
G1 = (0.4, 0.5, 0.6); TH1 = np.round(np.arange(0.70, 0.981, 0.02), 2)
G2 = (None, 1.5, 2.0, 2.5); TH2 = (0.3, 0.4, 0.5, 0.6)
S3 = [(S, p) for S in (2.0, 2.5, 3.0) for p in (0.1, 0.3, 0.5, 0.7, 0.9, 2.0)]
rows = []
for g1, th1, g2, (S, phi) in itertools.product(G1, TH1, G2, S3):
    for th2 in (TH2 if g2 is not None else (None,)):
        if g2 is not None and (g2 <= g1 or th2 >= th1): continue
        r = (g1, float(th1), g2, th2, S, phi)
        rows.append((r, D.score(D.events(r))))
print("候補", len(rows))
for cut in (0.06, 0.07, 0.08):
    for miss in (0.10, 0.12, 0.15, 0.18, 1.0):
        for bc in (0.10, 0.20, 1.0):
            ok = [x for x in rows if x[1]["cut"] <= cut and x[1]["miss"] <= miss and x[1]["bc"] <= bc]
            if not ok: print(f"== cut-in ≤ {cut:.0%}・応答しなかった ≤ {miss:.0%}・相槌の後 ≤ {bc:.0%}：なし"); continue
            r, m = min(ok, key=lambda x: (x[1]["lat"], x[1]["miss"]))
            print(f"== cut-in ≤ {cut:.0%}・応答しなかった ≤ {miss:.0%}・相槌の後 ≤ {bc:.0%}：{r}")
            print("   開発：", fmt(m)); print("   評価：", fmt(E.score(E.events(r))))
