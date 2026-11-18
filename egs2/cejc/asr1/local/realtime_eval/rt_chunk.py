#!/usr/bin/env python
"""完了の区間（次の区間まで 2 秒以上）について，区間末の無音で出たスコアの最大と，そのときモデルに入った声のかたまりの長さを調べる（2026-11-13）
  python rt_chunk.py <data> <書き出し>"""
import glob, os, sys
from collections import defaultdict
import numpy as np
HOP = 512 / 16000
data, ddir = sys.argv[1:3]
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
rows = []   # (区間の長さ, かたまりの長さ, スコア最大, そのときの P(相槌), 無音が見つかったか)
for path in sorted(glob.glob(f"{ddir}/*.npz")):
    rec = os.path.basename(path)[:-4]; d = np.load(path)
    sil = d["p"] < 0.5
    st = np.flatnonzero(sil & ~np.r_[False, sil[:-1]]) * HOP          # 無音の始まり
    en = np.flatnonzero(~sil & np.r_[False, sil[:-1]]) * HOP          # 無音の終わり（声が戻った）
    if len(en) < len(st): en = np.r_[en, len(sil) * HOP]
    if sil[0]: pass
    L = en[:len(st)] - st
    # かたまりの頭：直前の 0.4 秒以上の無音の終わり − 0.2
    cs = np.zeros(len(st)); cur = 0.0
    for k in range(len(st)):
        cs[k] = cur
        if L[k] >= 0.4: cur = max(0.0, en[k] - 0.2)
    lst = sorted(seg[rec])
    for j, (b, e, u) in enumerate(lst):
        if tag.get(u) != 1: continue
        g = lst[j + 1][0] - e if j + 1 < len(lst) else 9.0
        if g < 2: continue
        k = np.searchsorted(st, e + 0.5) - 1                         # 区間末 +0.5 秒までに始まった最後の無音
        if k < 0 or st[k] < b or L[k] < 0.4: rows.append((e - b, -1, -1, -1, 0)); continue
        m = (d["st"] > st[k]) & (d["st"] <= min(en[k], e + min(g, 5.0)))
        if not m.any(): rows.append((e - b, st[k] - cs[k], 0, 0, 1)); continue
        i = np.argmax(d["ss"][m])
        rows.append((e - b, st[k] - cs[k], d["ss"][m][i], d["sb"][m][i], 1))
R = np.array(rows); ok = R[:, 4] == 1
print(f"完了の区間 {len(R)}，区間末に 0.4 秒以上の無音が無い {np.mean(~ok):.1%}")
R = R[ok]
def tab(name, v, bins):
    print(f"-- {name}：件数／スコア最大が 0.82 未満の割合／スコア最大の中央値／P(相槌) の中央値")
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum(): print(f"  {lo:5.1f}〜{hi:5.1f}：{m.sum():6d}  {np.mean(R[m, 2] < 0.82):6.1%}  {np.median(R[m, 2]):.2f}  {np.median(R[m, 3]):.2f}")
tab("声のかたまりの長さ（秒）", R[:, 1], [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 1e9])
tab("区間の長さ（秒）", R[:, 0], [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 1e9])
d = R[:, 1] - R[:, 0]
tab("かたまりの長さ − 区間の長さ（秒）", d, [-1e9, -3, -1, -0.3, 0.5, 1, 3, 8, 1e9])
