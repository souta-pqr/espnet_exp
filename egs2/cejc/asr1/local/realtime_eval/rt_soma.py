#!/usr/bin/env python
"""waseda_soma（人のターンのラベル）で応答の決め方を採点する（2026-11-18）．rt_rule.py と同じ場所で動かす．数え方は CEJC と同じ（rt_rule.Data.score）．
silero の確率は書き出しの中のものを使う．無音だけの方式は sil=1.0 のように書く．
  python rt_soma.py <data> <書き出し> <名前=g1,th1,g2,th2,S,phi> …"""
import os, sys, tempfile
import numpy as np
from rt_rule import Data, fmt
data, ddir = sys.argv[1:3]
tmp = tempfile.mkdtemp()
for f in os.listdir(ddir):
    if f.endswith(".npz"): np.save(f"{tmp}/{f[:-4]}.npy", np.load(f"{ddir}/{f}")["p"])
D = Data(data, tmp, ddir)
n = {k: int(((D.lab == k) & D.heard).sum()) for k in (0, 1, 2)}
print(f"録音 {len(D.recs)}・区間 継続 {n[0]}・完了 {n[1]}（後に 2 秒以上の間 {int(((D.lab == 1) & D.heard & (D.g >= 2)).sum())}）・相槌 {n[2]}")
for spec in sys.argv[3:]:
    name, r = spec.split("=")
    r = (float(r), -9.0, None, None, None, None) if name.startswith("sil") else tuple(None if v in ("", "-") else float(v) for v in r.split(","))
    print(f"{name}\t{fmt(D.score(D.events(r)))}")
