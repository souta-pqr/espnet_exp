#!/usr/bin/env python
"""cut-in rate と latency の曲線（Raux & Eskenazi 2009 の比べ方．2026-11-14）．rt_rule.py と同じ場所で動かす．
cut-in rate の上限を少しずつ変え，開発データで latency が最小の決め方を選び，評価データに当てはめる．
応答しなかった割合の上限は全部の方式で同じ値にそろえる．
  python rt_curve.py <開発 data> <silero> <書き出し> <評価 data> <silero> <書き出し> [応答しなかった割合の上限=0.12]"""
import itertools, sys
import numpy as np
from rt_rule import Data
D = Data(*sys.argv[1:4]); E = Data(*sys.argv[4:7])
MISS = float(sys.argv[7]) if len(sys.argv) > 7 else 0.12
G1 = (0.4, 0.5, 0.6); TH1 = np.round(np.arange(0.70, 0.981, 0.02), 2)
G2 = (None, 1.5, 2.0, 2.5); TH2 = (0.3, 0.4, 0.5, 0.6)
S3 = [(S, p) for S in (2.0, 2.5, 3.0) for p in (0.1, 0.3, 0.5, 0.7, 0.9, 2.0)]
rows = []
for g1, th1, g2, (S, phi) in itertools.product(G1, TH1, G2, S3):
    for th2 in (TH2 if g2 is not None else (None,)):
        if g2 is not None and (g2 <= g1 or th2 >= th1): continue
        r = (g1, float(th1), g2, th2, S, phi)
        rows.append((r, D.score(D.events(r))))
print(f"# 候補 {len(rows)}・応答しなかった ≤ {MISS:.0%}")
print("上限\t決め方\t開発cut\t開発lat\t評価cut\t評価lat\t評価lat応答分\t評価平均応答分\t評価90%点\t評価応答なし\t評価相槌の後\t評価待てた")
for cut in np.round(np.arange(0.04, 0.1001, 0.005), 3):
    ok = [x for x in rows if x[1]["cut"] <= cut and x[1]["miss"] <= MISS]
    if not ok: print(f"{cut:.1%}\tなし"); continue
    r, m = min(ok, key=lambda x: (x[1]["lat"], x[1]["miss"]))
    e = E.score(E.events(r))
    print(f"{cut:.1%}\t{r}\t{m['cut']:.1%}\t{m['lat']:.2f}\t{e['cut']:.1%}\t{e['lat']:.2f}\t{e['lat_ok']:.2f}\t{e['mean_ok']:.2f}\t{e['p90']:.2f}\t{e['miss']:.1%}\t{e['bc']:.1%}\t{e['wait']:.1%}")
