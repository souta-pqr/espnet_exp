#!/usr/bin/env python
"""「まだ続く」区間の後の間で応答してしまう割合を，区間の終わり方・間の長さ・区間の長さで分ける（2026-11-15）．rt_rule.py と同じ場所で動かす．
  python rt_hes.py <data> <silero> <提案手法の書き出し> <決め方> [比べる実走の出力 …]"""
import re, sys
from collections import defaultdict
import numpy as np
from rt_rule import Data, real_events
data, sdir, ddir, rule = sys.argv[1:5]
D = Data(data, sdir, ddir)
# Data と同じ順に区間の名前を並べる
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0] for l in open(f"{data}/tag")}
txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(f"{data}/text"))
us = [u for rec in D.recs for _, _, u in sorted(seg[rec]) if u in tag]
assert len(us) == len(D.b)

def took(ev):
    t, se, _ = ev; o = np.argsort(t); t, se = t[o], se[o]
    i1 = np.searchsorted(t, D.e, "left") - 1
    mid = (i1 >= 0) & (t[np.maximum(i1, 0)] >= D.b) & (se[np.maximum(i1, 0)] >= D.b)
    i2 = np.maximum(np.searchsorted(t, D.e, "left"), np.searchsorted(se, D.b, "left"))
    ok = i2 < len(t); i2c = np.minimum(i2, len(t) - 1); ok &= t[i2c] <= D.hor
    return mid | ok

def kind(u):
    raw = txt[u]
    if re.search(r"\(F[^)]*\)\s*$", raw): return "フィラーで終わる"
    s = re.sub(r"\([A-Z?]+ ?", "", raw).replace(")", "").replace(" ", "")
    s = re.sub(r"<[a-z]+>", "", s)
    if re.search(r"(まあ|まー|えー|えーと|えっと|あの|あのー|その|そのー|なんか|うーん|んー|そうだな|そうですね|こう)ー?$", s): return "フィラーで終わる"
    if s.endswith("ー"): return "語尾を伸ばして終わる"
    if re.search(r"(けど|けれど|て|で|が|から|し|と|は|も|に|を|の|とか|って|ので|たら|ば)$", s): return "助詞で終わる"
    return "それ以外"

sysm = [("提案手法", took(D.events(tuple(None if v in ("", "-") else float(v) for v in rule.split(",")))))]
for d in sys.argv[5:]: sysm.append((d, took(real_events(D, d))))
cont = D.heard & (D.lab == 0) & (D.g >= 0.4)
K = np.array([kind(u) for u in us]); L = D.e - D.b
def show(title, groups):
    print(f"\n## {title}\n" + "分け方\t区間の数\t" + "\t".join(n for n, _ in sysm))
    for name, m in groups:
        m = m & cont
        print(f"{name}\t{m.sum()}\t" + "\t".join(f"{tk[m].mean():.1%}" for _, tk in sysm))
print("「まだ続く」で後に 0.4 秒以上の間がある区間", cont.sum(), "（フィラーの印 (F があるか：", any("(F" in txt[u] for u in us[:5000]), "）")
show("区間の終わり方", [(k, K == k) for k in ("フィラーで終わる", "語尾を伸ばして終わる", "助詞で終わる", "それ以外")] + [("全部", np.ones(len(K), bool))])
show("後の間の長さ", [(f"{a}〜{b} 秒", (D.g >= a) & (D.g < b)) for a, b in ((0.4, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99))])
show("区間の長さ", [(f"{a}〜{b} 秒", (L >= a) & (L < b)) for a, b in ((0, 1), (1, 2), (2, 4), (4, 8), (8, 99))])
show("フィラーで終わる・語尾を伸ばす区間の，後の間の長さ", [(f"{a}〜{b} 秒", (D.g >= a) & (D.g < b) & ((K == "フィラーで終わる") | (K == "語尾を伸ばして終わる"))) for a, b in ((0.4, 1.0), (1.0, 2.0), (2.0, 99))])
