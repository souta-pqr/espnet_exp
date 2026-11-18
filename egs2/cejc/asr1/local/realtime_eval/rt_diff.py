"""完了の区間（次まで 2 秒以上）で，区切る前は応答し，区切った後は応答しなくなった区間を調べる（2026-11-04）"""
import csv, os, sys
from collections import defaultdict, Counter
import numpy as np
data, r0, r1, sdir = sys.argv[1:5]
HOP = 512 / 16000
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
def load(p):
    return [(float(r["t"]), float(r["utt_start"]), float(r["speech_end"]), r["cond"]) for r in csv.DictReader(open(p), delimiter="\t") if r["model"] == "proposed"]
C = Counter(); ex = []
for rec in seg:
    if not os.path.exists(f"{r0}/{rec}.tsv"): continue
    x = np.load(f"{sdir}/{rec}.npy")
    E0, E1 = load(f"{r0}/{rec}.tsv"), load(f"{r1}/{rec}.tsv")
    lst = sorted(seg[rec])
    for k, (b, e, u) in enumerate(lst):
        if tag.get(u) != 1 or x[int(b / HOP):int(e / HOP) + 1].max(initial=0) < 0.5: continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        if g < 2: continue
        hor = e + min(g, 5.0)
        h0 = [ev for ev in E0 if ev[3] != "0" and b <= ev[2] and b <= ev[0] <= hor]
        h1 = [ev for ev in E1 if ev[3] != "0" and b <= ev[2] and b <= ev[0] <= hor]
        if h0 and not h1:
            # 区切った後で，この区間を含む発話はどう終わったか
            near = [ev for ev in E1 if ev[0] >= b][:1]
            if not near: C["後に応答がない"] += 1; continue
            t, us, se, c = near[0]
            why = ("頭が区間の後（区間末より後に区切り直した）" if us > e else "頭が区間の中") + ("・5 秒無音で打ち切り" if c == "0" else "・区間末＋5 秒より後に応答")
            C[why] += 1
            if len(ex) < 8: ex.append((rec, b, e, us, se, t, c))
print(sum(C.values()), C.most_common())
for z in ex: print(*[z[0]] + ["%.2f" % v for v in z[1:6]] + [z[6]])
