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
# 11/16：モデルの弱点を探る．2 つの LLM が一致して「まだ続く」で，後の間が 1〜3 秒の区間
print("\n\n######## 弱点の分析")
q = dict(l.split() for l in open(f"{data}/tag.qwen")); Q = np.array([int(q[u]) for u in us])
tk = sysm[0][1]
# 各区間の末の無音を探す
first = D.first; se = D.se_all[D.rid[first]]; last = np.r_[first[1:] - 1, len(D.t) - 1]
dur = D.run[last]                                    # 無音の長さ（上限あり）
r = np.clip(np.searchsorted(se, D.e - 0.3), 0, len(se) - 1)
has = (se[r] >= D.e - 0.3) & (se[r] <= D.e + 0.5)
# 今の声のかたまりの長さ：0.4 秒以上の無音で区切る
cs = np.zeros(len(se)); start = None
for i in range(len(se)):
    if i == 0 or se[i] - D.t[last[i - 1]] < 0 or se[i] - se[i - 1] > 60 or dur[i - 1] >= 0.4: start = D.t[last[i - 1]] if i and se[i] - se[i - 1] <= 60 else se[i] - 30
    cs[i] = se[i] - start
CL = np.where(has, np.minimum(cs[r], 12), np.nan)
def sc_at(x):                                        # 無音が x 秒のときのスコア（そこまで無音が続かなければ nan）
    out = np.full(len(r), np.nan)
    for j, i in enumerate(r):
        if not has[j]: continue
        a, b = first[i], last[i] + 1
        k = a + np.searchsorted(D.run[a:b], x)
        if k < b: out[j] = max(D.sc[k], 0.0)
    return out
S5, S10, S15 = sc_at(0.5), sc_at(1.0), sc_at(1.5)
weak = cont & (Q == 0) & (D.g >= 1.0) & (D.g < 3.0) & has
done = D.heard & (D.lab == 1) & (Q == 1) & (D.g >= 1.0) & has
print("弱点の区間", weak.sum(), "応答してしまう", f"{tk[weak].mean():.1%}", " 比べる相手（一致して言い終わり・間 1 秒以上）", done.sum(), "応答する", f"{tk[done].mean():.1%}")
ev = D.events(tuple(None if v in ("", "-") else float(v) for v in rule.split(",")))
t, sev, cond = ev; o = np.argsort(t); t, sev, cond = t[o], sev[o], cond[o]
i2 = np.minimum(np.maximum(np.searchsorted(t, D.e, "left"), np.searchsorted(sev, D.b, "left")), len(t) - 1)
m = weak & tk
print("応答した条件：", {int(c): int((cond[i2[m]] == c).sum()) for c in (1, 2, 3)}, " 応答までの無音の中央値", np.median((t[i2] - sev[i2])[m]).round(2))
def dist(name, x, m):
    v = x[m]; v = v[~np.isnan(v)]
    print(f"{name}\t{len(v)}\t" + "\t".join(f"{np.quantile(v, p):.2f}" for p in (0.1, 0.25, 0.5, 0.75, 0.9)) + f"\t{(v >= 0.84).mean():.1%}\t{(v >= 0.6).mean():.1%}")
print("\nスコアの分布\t数\t10%\t25%\t50%\t75%\t90%\t0.84 以上\t0.60 以上")
for nm, x in (("無音 0.5 秒", S5), ("無音 1.0 秒", S10), ("無音 1.5 秒", S15)):
    dist(nm + "・弱点", x, weak); dist(nm + "・言い終わり", x, done)
from sklearn.metrics import roc_auc_score
for nm, x in (("無音 0.5 秒", S5), ("無音 1.0 秒", S10), ("無音 1.5 秒", S15)):
    ok = (weak | done) & ~np.isnan(x); print(nm, "のスコアで弱点と言い終わりを分ける AUC", round(roc_auc_score(done[ok], x[ok]), 3))
def tab(title, groups):
    print(f"\n## {title}\n分け方\t区間の数\t応答してしまう\t無音 0.5 秒のスコア中央値")
    for name, mm in groups:
        mm = mm & weak
        if mm.sum(): print(f"{name}\t{mm.sum()}\t{tk[mm].mean():.1%}\t{np.nanmedian(S5[mm]):.2f}")
tab("今の声のかたまりの長さ", [(f"{a}〜{b} 秒", (CL >= a) & (CL < b)) for a, b in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 13))])
tab("区間の終わり方", [(k, K == k) for k in ("フィラーで終わる", "語尾を伸ばして終わる", "助詞で終わる", "それ以外")])
tab("区間の長さ", [(f"{a}〜{b} 秒", (L >= a) & (L < b)) for a, b in ((0, 1), (1, 2), (2, 4), (4, 99))])
import pickle; pickle.dump(dict(us=us, weak=weak, tk=tk, S5=S5, CL=CL, g=D.g, K=K), open("weak_espm.pkl", "wb"))
print("\n## 今の声のかたまりの長さ別：弱点と言い終わりの分けやすさ（無音 0.5 秒のスコア）\n長さ\t弱点の数\t弱点で 0.84 以上\t言い終わりの数\t言い終わりで 0.84 以上\tAUC")
for a, b in ((0, 1), (1, 2), (2, 4), (4, 13)):
    mm = (CL >= a) & (CL < b) & ~np.isnan(S5); w = weak & mm; d = done & mm
    print(f"{a}〜{b} 秒\t{w.sum()}\t{(S5[w] >= 0.84).mean():.1%}\t{d.sum()}\t{(S5[d] >= 0.84).mean():.1%}\t{roc_auc_score(np.r_[np.zeros(w.sum()), np.ones(d.sum())], np.r_[S5[w], S5[d]]):.3f}")
allc = cont & (D.g >= 1.0) & (D.g < 3.0) & has
print("\n（参考）ラベルが割れている区間も含めた「まだ続く」・間 1〜3 秒：", allc.sum(), f"{tk[allc].mean():.1%}")
