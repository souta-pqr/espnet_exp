#!/usr/bin/env python
"""rt_pick.py と同じ選び方を，無音が CAP 秒を超えた後のスコアを CAP 秒の時点の値に固定して行う（2026-11-27）．
最後のフレームで判定するモデルが実時間で悪いのは，無音が長くなった後（エンコーダの区切り 約 1.4 秒を超えた後）の
崩れたスコアで応答してしまうからか，を確かめる．崩れたスコアを使わなければ，全フレームの平均に近づくはず．
  cd <rt_rule.py のある場所>
  CAP=1.2 python rt_pick_cap.py <data> <silero> <書き出し> <評価 data> <silero> <書き出し>"""
import itertools, os, sys
import numpy as np
from rt_rule import Data, fmt

CAP = float(os.environ["CAP"])


def cap(D):
    """無音の中で run > CAP の時刻のスコアを，同じ無音の run <= CAP の最後の時刻のスコアにする"""
    pos = np.arange(len(D.run))
    last = np.maximum.accumulate(np.where(D.run <= CAP, pos, -1))
    ok = (last >= 0) & (D.rid[np.maximum(last, 0)] == D.rid)      # 同じ無音の中に run <= CAP の時刻がある
    j = np.where(ok, last, pos)
    D.sc, D.pb = D.sc[j], D.pb[j]
    return D


D = cap(Data(*sys.argv[1:4])); E = cap(Data(*sys.argv[4:7]))
G1 = (0.4, 0.5, 0.6); TH1 = np.round(np.arange(0.50, 0.981, 0.02), 2)
G2 = (None, 1.5, 2.0, 2.5); TH2 = (0.3, 0.4, 0.5, 0.6)
S3 = [(S, p) for S in (2.0, 2.5, 3.0) for p in (0.1, 0.3, 0.5, 0.7, 0.9, 2.0)]
rows = []
for g1, th1, g2, (S, phi) in itertools.product(G1, TH1, G2, S3):
    for th2 in (TH2 if g2 is not None else (None,)):
        if g2 is not None and (g2 <= g1 or th2 >= th1): continue
        r = (g1, float(th1), g2, th2, S, phi)
        rows.append((r, D.score(D.events(r))))
print(f"CAP={CAP}")
for cut, miss, bc in ((0.08, 0.10, 0.10), (0.07, 0.12, 0.10), (0.06, 0.12, 0.20), (0.06, 0.15, 0.10), (0.06, 1.0, 0.10)):
    ok = [x for x in rows if x[1]["cut"] <= cut and x[1]["miss"] <= miss and x[1]["bc"] <= bc]
    print(f"== cut-in ≤ {cut:.0%}・応答しなかった ≤ {miss:.0%}・相槌の後 ≤ {bc:.0%}：", end="")
    if not ok: print("なし"); continue
    r, m = min(ok, key=lambda x: (x[1]["lat"], x[1]["miss"]))
    print(r); print("   開発：", fmt(m)); print("   評価：", fmt(E.score(E.events(r))))
