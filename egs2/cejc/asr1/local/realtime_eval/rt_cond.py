"""実時間評価で，応答がどの条件で出たか（モデルの判断＝条件1・2，長い無音＝条件3）を区間の種類ごとに数える（2026-11-03）"""
import csv, glob, os, sys
from collections import defaultdict, Counter
import numpy as np
data, rdir, sdir = sys.argv[1:4]
HOP = 512 / 16000
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
C = defaultdict(Counter)
for p in glob.glob(f"{rdir}/*.tsv"):
    rec = os.path.basename(p)[:-4]
    x = np.load(f"{sdir}/{rec}.npy")
    ev = defaultdict(list)
    for r in csv.DictReader(open(p), delimiter="\t"):
        if r["cond"] != "0": ev[r["model"]].append((float(r["t"]), float(r["speech_end"]), r["cond"], float(r["run"])))
    lst = sorted(seg[rec])
    for k, (b, e, u) in enumerate(lst):
        if u not in tag or x[int(b / HOP):int(e / HOP) + 1].max(initial=0) < 0.5: continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        hor = e + min(g, 5.0)
        kind = {1: "完了（次まで2秒以上）" if g >= 2 else None, 0: ("継続（間 0.4〜1 秒）" if 0.4 <= g < 1 else "継続（間 1〜2 秒）" if 1 <= g < 2 else "継続（間 2〜3 秒）" if 2 <= g < 3 else None)}.get(tag[u])
        if kind is None: continue
        for m, E in ev.items():
            hit = sorted((t, c, run) for t, se, c, run in E if b <= se and b <= t <= hor)
            C[(m, kind)]["応答なし" if not hit else ("条件" + hit[0][1] + ("（無音 %.1f 秒）" % hit[0][2] if hit[0][1] == "3" else ""))] += 1
for (m, kind), c in sorted(C.items()):
    n = sum(c.values())
    print(m, kind, n, "  ".join(f"{k} {v / n:.1%}" for k, v in sorted(c.items())))
