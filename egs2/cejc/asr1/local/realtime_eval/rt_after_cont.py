#!/usr/bin/env python
"""latency の区間（次まで 2 秒以上空いた完了）を，直前の同じ話者の区間で分けて採点する（2026-11-11）．
  詰まった後：直前の区間が継続で，その間が 0.4〜2.0 秒（実時間では前の区間とつながって 1 つの発話になりやすい）
  それ以外
  python rt_after_cont.py <data> <silero の出力> <名前=rt_eval の出力:モデル> ...
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
data, sdir = sys.argv[1], sys.argv[2]
HOP, W = 512 / 16000, 5.0
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
for spec in sys.argv[3:]:
    name, rest = spec.split("="); rdir, model = rest.split(":")
    out = defaultdict(list)
    for rec in sorted(seg):
        p = f"{rdir}/{rec}.tsv"
        if not os.path.exists(p): continue
        ev = [(float(r["t"]), float(r["speech_end"])) for r in csv.DictReader(open(p), delimiter="\t") if r["model"] == model and r["cond"] != "0"]
        x = np.load(f"{sdir}/{rec}.npy"); lst = sorted(seg[rec])
        for k, (b, e, u) in enumerate(lst):
            if tag.get(u) != 1 or k + 1 >= len(lst) or lst[k + 1][0] - e < 2.0: continue
            if x[int(b / HOP):int(e / HOP) + 1].max(initial=0) < 0.5: continue
            if any(b <= se and b <= t < e for t, se in ev): continue
            ts = [t for t, se in ev if b <= se and e <= t <= e + W]
            lat = min(ts) - e if ts else np.inf
            pb, pe, pu = lst[k - 1] if k else (0, -9, None)
            kind = "詰まった後" if tag.get(pu) == 0 and 0.4 <= b - pe <= 2.0 else "それ以外"
            out[kind].append(lat)
    for kind in ("詰まった後", "それ以外"):
        a = np.array(out[kind])
        print(f"{name}\t{kind}\t{len(a)}\t遅れの中央値 {np.median(a):.2f}\t1 秒以内 {np.mean(a <= 1.0):.1%}\t応答せず {np.mean(~np.isfinite(a)):.1%}")
