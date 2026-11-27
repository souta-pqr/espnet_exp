#!/usr/bin/env python
"""silero-vad のみ（無音が g 秒続いたら応答）の決め方を，開発データで cut-in rate の上限ごとに選び，評価データに当てはめる（2026-11-18）．
silero-vad の確率は提案手法の書き出しの中のもの（どの方式でも同じ）．rt_rule.py と同じ場所で動かす．
  python rt_silero.py <開発 data> <silero> <書き出し> <評価 data> <silero> <書き出し>"""
import sys
import numpy as np
from rt_rule import Data, fmt
D = Data(*sys.argv[1:4]); E = Data(*sys.argv[4:7])
rows = []
for g in np.round(np.arange(0.3, 3.01, 0.05), 2):
    r = (float(g), -9.0, None, None, None, None)
    rows.append((r, D.score(D.events(r))))
for g_, m in rows[::4]:
    print(f"開発  無音 {g_[0]:.2f} 秒\t{fmt(m)}")
for cut in (0.06, 0.07, 0.08):
    ok = [x for x in rows if x[1]["cut"] <= cut]
    if not ok:
        print(f"== cut-in ≤ {cut:.0%}：なし"); continue
    r, m = min(ok, key=lambda x: x[1]["lat"])
    print(f"== cut-in ≤ {cut:.0%}：無音 {r[0]:.2f} 秒\n  開発\t{fmt(m)}\n  評価\t{fmt(E.score(E.events(r)))}")
