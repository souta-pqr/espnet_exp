#!/usr/bin/env python
"""Hazumi（人が WoZ のエージェントに話す雑談）で，応答の決め方を採点する（2026-11-16）．rt_rule.py と同じ場所で動かす．
正解は LLM のラベルではなく，実際の会話の流れ：やりとりごとに，システムの発話の終わり a から，操作者が次の応答を出した時刻 c までがユーザの持ち時間．
その中で silero が最後に声を拾った時刻 L を「言い終わり」とする．
  cut-in：L より前に応答し，その後にユーザの声が 0.3 秒以上続いた（まだ話すつもりだった）
  latency：cut-in が無いやりとりで，L から最初の応答までの時間．c までに応答しなければ「応答しなかった」
  python rt_hazumi.py <やりとりの表 tsv> <書き出し> <名前=決め方> …   決め方は g1,th1,g2,th2,S,phi（無音だけの方式は sil=1.0 のように書く）"""
import csv, os, sys, tempfile
from collections import defaultdict
import numpy as np
import rt_rule
from rt_rule import Data, HOP, OFF

tsv, ddir = sys.argv[1:3]
ex = defaultdict(list)
for r in csv.DictReader(open(tsv), delimiter="\t"):
    ex[r["rec"]].append((float(r["sys_end"]), float(r["ex_end"]), int(r["nbar"])))
# Data をそのまま使うための仮のデータ（区間＝ユーザの持ち時間，ラベルは全部 1）
tmp = tempfile.mkdtemp(); os.makedirs(tmp + "/sil")
with open(tmp + "/segments", "w") as fs, open(tmp + "/tag", "w") as ft:
    for rec in sorted(ex):
        if not os.path.exists(f"{ddir}/{rec}.npz"): continue
        np.save(f"{tmp}/sil/{rec}.npy", np.load(f"{ddir}/{rec}.npz")["p"])
        for i, (a, c, _) in enumerate(ex[rec]):
            fs.write(f"{rec}_{i:04d} {rec} {a:.3f} {c:.3f}\n"); ft.write(f"{rec}_{i:04d} 1\n")
D = Data(tmp, tmp + "/sil", ddir)
# やりとりごとの，声のある時刻
W = []
for k, rec in enumerate(D.recs):
    p = np.load(f"{ddir}/{rec}.npz")["p"]; v = p >= 0.5
    for a, c, nb in ex[rec]:
        i0, i1 = int(a / HOP) + 3, int(c / HOP)              # システムの声の消し残りを避けて，頭の約 0.1 秒は見ない
        vi = np.flatnonzero(v[i0:i1]) + i0
        if len(vi) < 10: continue                           # 声が 0.3 秒に満たないやりとりは数えない
        vt = (vi + 1) * HOP
        gaps = np.diff(vt); longest = gaps.max() if len(gaps) else 0.0
        W.append((k * OFF + a, k * OFF + c, k * OFF + vt[-1], k * OFF + vt, longest, nb))
print(f"録音 {len(D.recs)}・数えるやりとり {len(W)}・操作者が応答するまで（言い終わりから）中央値 {np.median([w[1] - w[2] for w in W]):.2f} 秒・途中に 1 秒以上の間があるやりとり {sum(w[4] >= 1.0 for w in W)}・2 秒以上 {sum(w[4] >= 2.0 for w in W)}")

def score(t):
    t = np.sort(t); cut, lat, pc1, pc2 = [], [], [], []
    for a, c, L, vt, lg, nb in W:
        r = t[(t > a) & (t < c)]
        ci = any(((vt > x).sum() >= 10) for x in r[r < L])  # 応答の後に声が 0.3 秒以上
        cut.append(ci)
        if lg >= 1.0: pc1.append(ci)
        if lg >= 2.0: pc2.append(ci)
        if not ci:
            after = r[r >= L - 1e-6]; lat.append(after[0] - L if len(after) else np.inf)
    lat = np.array(lat); ok = np.isfinite(lat)
    return f"cut-in rate {np.mean(cut):.1%}\tlatency {np.median(lat):.2f}\t応答しなかった {1 - ok.mean():.1%}\t応答した分 {np.median(lat[ok]):.2f}\t途中に 1 秒以上の間：cut-in {np.mean(pc1):.1%}\t2 秒以上：{np.mean(pc2):.1%}"

for spec in sys.argv[3:]:
    name, r = spec.split("=")
    if name.startswith("sil"):
        x = float(r); f = np.minimum.reduceat(np.where(D.run >= x, np.arange(len(D.run)), len(D.run)), D.first); t = D.t[f[f < len(D.run)]]
    else:
        t = D.events(tuple(None if v in ("", "-") else float(v) for v in r.split(",")))[0]
    print(f"{name}\t{r}\t{score(t)}")
