"""実時間評価で，継続の区間（次の自分の区間まで 1〜3 秒）に応答した例を出す（2026-11-03）。
  - 間のあいだに同じ会話の他の人（他の IC のチャネル）が話したか
  - 区間の文・同じ人の次の区間の文・応答したときのスコア
  python rt_cutin_examples.py <data> <rt_eval の出力> <vad_silero> <方式> <出力 tsv>
"""
import csv, glob, os, re, sys, random
from collections import defaultdict, Counter
import numpy as np
data, rdir, sdir, model, out = sys.argv[1:6]
HOP = 512 / 16000
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(f"{data}/text"))
norm = lambda t: re.sub(r"\([A-Z?]+ ?", "", re.sub(r"<(mask|unclear)>", "＊", t)).replace(")", "").replace(" ", "")
sess = defaultdict(list)                         # 会話 → [(rec, b, e)]
for rec, L in seg.items():
    for b, e, u in L: sess[rec.rsplit("_", 1)[0]].append((rec, b, e))
rows, C = [], Counter()
for p in sorted(glob.glob(f"{rdir}/*.tsv")):
    rec = os.path.basename(p)[:-4]
    x = np.load(f"{sdir}/{rec}.npy")
    ev = [(float(r["t"]), float(r["speech_end"]), r["cond"], r["score"]) for r in csv.DictReader(open(p), delimiter="\t") if r["model"] == model and r["cond"] != "0"]
    lst = sorted(seg[rec])
    for k, (b, e, u) in enumerate(lst[:-1]):
        if tag.get(u) != 0 or x[int(b / HOP):int(e / HOP) + 1].max(initial=0) < 0.5: continue
        nb = lst[k + 1][0]; g = nb - e
        if not 1 <= g < 3: continue
        hit = [(t, c, s) for t, se, c, s in ev if b <= se and b <= t <= nb]
        other = sum(max(0, min(oe, nb) - max(ob, e)) for orec, ob, oe in sess[rec.rsplit("_", 1)[0]] if orec != rec)
        o = "他の人が話した" if other >= 0.3 else "誰も話さない"
        C[(o, bool(hit))] += 1
        if hit:
            rows.append((rec, f"{b:.2f}", f"{e:.2f}", f"{g:.2f}", o, f"{hit[0][0] - e:.2f}", hit[0][2], norm(txt[u]), norm(txt[lst[k + 1][2]])))
for o in ("誰も話さない", "他の人が話した"):
    n = C[(o, True)] + C[(o, False)]
    print(f"{o}：{n} 区間のうち応答 {C[(o, True)]}（{C[(o, True)] / max(n, 1):.1%}）")
random.Random(0).shuffle(rows)
with open(out, "w") as f:
    f.write("rec\tb\te\t間\t間の他の人\t区間末から応答まで\tスコア\t区間の文\t次の自分の区間の文\n")
    for r in rows: f.write("\t".join(r) + "\n")
print(len(rows), "件 →", out)
