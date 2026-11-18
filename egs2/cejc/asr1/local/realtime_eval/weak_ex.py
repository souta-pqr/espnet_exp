import pickle, re, sys, numpy as np
from collections import defaultdict
A = "/home/kobori/2025/espnet/egs2/cejc/asr1/data/eval_g"
P = pickle.load(open("weak_espm.pkl", "rb")); us = P["us"]; S5, CL, g = P["S5"], P["CL"], P["g"]
txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(A + "/text"))
seg = defaultdict(list)
for l in open(A + "/segments"):
    u, r, b, e = l.split(); seg[r].append((float(b), u))
nxt = {}
for r, v in seg.items():
    v.sort()
    for (b, u), (b2, u2) in zip(v, v[1:]): nxt[u] = u2
cl = lambda s: re.sub(r"\([A-Z?]+ ?", "(", s).replace(" ", "")
for kind in sys.argv[1:]:
    idx = np.flatnonzero(P["weak"] & P["tk"] & (P["K"] == kind)); print("##", kind, len(idx))
    for i in np.random.default_rng(0).choice(idx, min(26, len(idx)), replace=False):
        u = us[i]; print(f"{S5[i]:.2f} かたまり{CL[i]:.1f}s 間{g[i]:.1f}s｜{cl(txt[u])[-40:]} ／→ {cl(txt.get(nxt.get(u), ''))[:25]}")
