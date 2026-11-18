"""実時間評価で Smooth（完了・次まで 2 秒以上）に応答しなかった区間の内訳（2026-11-03）"""
import csv, glob, os, sys
from collections import defaultdict, Counter
import numpy as np
data, rdir, model = sys.argv[1], sys.argv[2], sys.argv[3]
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
C = Counter(); late = []
for p in glob.glob(f"{rdir}/*.tsv"):
    rec = os.path.basename(p)[:-4]
    ev = [(float(r["t"]), float(r["utt_start"]), float(r["speech_end"]), r["cond"]) for r in csv.DictReader(open(p), delimiter="\t") if r["model"] == model]
    lst = sorted(seg[rec])
    for k, (b, e, u) in enumerate(lst):
        if tag.get(u) != 1: continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        if g < 2: continue
        hor = e + min(g, 5.0)
        if any(b <= se and b <= t <= hor and c != "0" for t, us, se, c in ev): continue
        # 応答しなかった：何があったか
        after = [(t, us, se, c) for t, us, se, c in ev if t >= b]
        nxt = after[0] if after else None
        if nxt is None: C["以後に出来事なし"] += 1; continue
        t, us, se, c = nxt
        if us > hor: C["区間の間に発話の頭がない（silero が声を取らない）"] += 1
        elif c == "0": C["無音 5 秒で打ち切り（条件を満たさず）"] += 1
        elif t > hor: C["応答が窓より後（声ありが続いた）"] += 1; late.append(se - e)
        elif se < b: C["前の発話への応答の中"] += 1
        else: C["その他"] += 1
tot = sum(C.values())
for k, v in C.most_common(): print(f"{k}\t{v}\t{v / tot:.0%}")
if late: print("窓より後の応答：声が止んだ時刻 − 区間末 の中央値", np.median(late).round(2))
