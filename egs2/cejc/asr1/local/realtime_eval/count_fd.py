"""フィラー・言い直し・語尾の伸ばしで終わる区間のラベルと，次の声までの間を数える（2026-11-15）"""
import sys, re, collections
import numpy as np
A = "/home/kobori/2025/espnet/egs2/cejc/asr1/data/" + sys.argv[1]
tag = dict(l.split() for l in open(A + "/tag"))
txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(A + "/text"))
seg = [l.split() for l in open(A + "/segments")]
bs = collections.defaultdict(list)
for u, r, b, e in seg: bs[r].append(float(b))
bs = {k: np.array(sorted(set(v))) for k, v in bs.items()}
def kind(t):
    t = t.strip()
    if re.search(r"\(F[^()]*\)$", t): return "フィラーで終わる"
    if re.search(r"\(D[^()]*\)$", t): return "言い直しで終わる"
    if t.endswith("ー"): return "語尾を伸ばす"
    return "それ以外"
N = collections.defaultdict(lambda: np.zeros(3, int)); G = collections.defaultdict(list)
for u, r, b, e in seg:
    k = kind(txt[u]); y = int(tag[u]); N[k][y] += 1
    a = bs[r]; i = np.searchsorted(a, float(e) - 0.05)
    G[k, y].append(a[i] - float(e) if i < len(a) else 99.0)
print("終わり方\t区間の数\tまだ続く\t言い終わり\t相槌\t言い終わりのうち 2 秒以内に次の声\t同 1 秒以内")
for k in ("フィラーで終わる", "言い直しで終わる", "語尾を伸ばす", "それ以外"):
    n = N[k]; g = np.array(G[k, 1])
    print(f"{k}\t{n.sum()}\t{n[0]/n.sum():.1%}\t{n[1]/n.sum():.1%}\t{n[2]/n.sum():.1%}\t{(g<2).mean():.1%}\t{(g<1).mean():.1%}")
