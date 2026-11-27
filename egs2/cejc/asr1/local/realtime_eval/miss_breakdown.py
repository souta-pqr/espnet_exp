"""実時間の評価で応答しなかった完了の区間を，音声区間と書き起こしの区間のずれの種類で分ける（2026-11-19）．
rt_score.py と同じ数え方（次の区間まで 2 秒以上空いた完了の区間，silero が声を拾ったもの，途中で応答していないもの）．
  python miss_breakdown.py <data> <rt_eval の出力> <silero の出力>
"""
import csv, glob, os, sys
from collections import defaultdict, Counter
import numpy as np

data, rdir, sdir = sys.argv[1:4]
H = 512 / 16000
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, r, b, e = l.split(); seg[r].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
cnt = {"応答した": Counter(), "応答しなかった": Counter()}
lat = defaultdict(list)
for f in sorted(glob.glob(f"{rdir}/*.tsv")):
    rec = os.path.basename(f)[:-4]
    ev = [(float(r["t"]), float(r["speech_end"])) for r in csv.DictReader(open(f), delimiter="\t") if r["cond"] != "0"]
    p = np.load(f"{sdir}/{rec}.npy").astype(np.float32); sp = p >= 0.5
    ed = np.diff(np.r_[0, (~sp).astype(int), 0]); s0, s1 = np.where(ed == 1)[0], np.where(ed == -1)[0]
    lg = (s1 - s0) * H >= 0.4; ls0, ls1 = s0[lg] * H, s1[lg] * H
    on = np.where(np.diff(np.r_[0, sp.astype(int)]) == 1)[0]
    heads = np.r_[on[:1] * H, ls1] - 0.2
    lst = sorted(seg[rec])
    for k, (b, e, u) in enumerate(lst):
        if tag.get(u) != 1:
            continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        if g < 2.0 or not sp[int(b / H):int(e / H) + 1].any():
            continue
        if any(b <= se and b <= t < e for t, se in ev):
            continue
        ts = [t for t, se in ev if b <= se and e <= t <= e + min(g, 5.0)]
        key = "応答した" if ts else "応答しなかった"
        j = np.where((ls0 >= e - 0.3) & (ls0 <= e + 5))[0]
        if len(j) == 0:
            c = "1 区間末の後 5 秒以内に 0.4 秒の無音が来ない"
        elif ls0[j[0]] - e >= 1.0:
            c = "2 無音の始まりが区間末より 1 秒以上遅い"
        else:
            hh = heads[heads < ls0[j[0]]]; h = hh[-1] if len(hh) else 0.0
            dur = e - b
            if h - b <= -0.5:
                c = "3 音声区間が区間の頭より 0.5 秒以上前から"
            elif h - b >= 0.5:
                c = "4 音声区間が区間の途中から"
            elif dur < 1.0:
                c = "5 ずれなし・区間が 1 秒未満"
            else:
                c = "6 ずれなし・区間が 1 秒以上"
        cnt[key][c] += 1
for key in cnt:
    n = sum(cnt[key].values())
    print(f"== {key}（{n} 区間）")
    for c in sorted(set(cnt["応答した"]) | set(cnt["応答しなかった"])):
        print(f"  {c}\t{cnt[key][c]}\t{cnt[key][c] / n:.1%}")
print("== 種類ごとの応答しなかった割合")
for c in sorted(set(cnt["応答した"]) | set(cnt["応答しなかった"])):
    a, m = cnt["応答した"][c], cnt["応答しなかった"][c]
    print(f"  {c}\t{m}/{a + m}\t{m / max(1, a + m):.1%}")
