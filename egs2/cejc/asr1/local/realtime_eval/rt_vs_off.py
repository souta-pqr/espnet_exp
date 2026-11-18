"""silero-vad のみ（無音 1.575 秒）の Smooth の区間で，オフラインと実時間の応答を区間ごとに突き合わせる（2026-11-03）"""
import csv, glob, os, sys
from collections import defaultdict, Counter
import numpy as np
data, rdir, sdir = sys.argv[1:4]
G, HOP = 1.575, None
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
C = Counter(); ex = []
for p in sorted(glob.glob(f"{rdir}/*.tsv")):
    rec = os.path.basename(p)[:-4]
    ev = [(float(r["t"]), float(r["utt_start"]), float(r["speech_end"])) for r in csv.DictReader(open(p), delimiter="\t") if r["model"] == "silero" and r["cond"] != "0"]
    x = np.load(f"{sdir}/{rec}.npy").astype(np.float32); hop = 0.032   # SIL_HOP
    lst = sorted(seg[rec])
    for k, (b, e, u) in enumerate(lst):
        if tag.get(u) != 1: continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        if g < 2: continue
        hor = e + min(g, 5.0)
        # オフライン：区間の頭から，無音（0.35 未満で数え始め 0.5 以上で戻す）が G 秒続いた最初の時刻
        i0, i1 = int(b / hop), min(len(x), int(hor / hop) + 1)
        start, off = -1, None
        for j in range(i0, i1):
            v = x[j]
            if v >= 0.5: start = -1
            elif start < 0 and v < 0.35: start = j
            if start >= 0 and (j - start + 1) * hop >= G - 1e-9:
                off = (j + 1) * hop; break
        off_ok = off is not None and off >= e
        rt_ok = any(b <= se and e <= t <= hor for t, us, se in ev)
        C[(off_ok, rt_ok)] += 1
        if off_ok and not rt_ok:
            C["  うち区間中に silero が 0.5 以上にならない"] += int(x[int(b / hop):int(e / hop) + 1].max(initial=0) < 0.5)
        if off_ok and not rt_ok and len(ex) < 8: ex.append((rec, round(b, 2), round(e, 2), round(off, 2)))
tot = sum(C.values())
tot = sum(v for k, v in C.items() if isinstance(k, tuple))
for k, v in C.items():
    print((f"オフライン{'応答' if k[0] else '無し'}・実時間{'応答' if k[1] else '無し'}" if isinstance(k, tuple) else k) + f"\t{v}\t{v / tot:.1%}")
print(ex)
