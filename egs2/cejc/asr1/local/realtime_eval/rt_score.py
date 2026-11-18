#!/usr/bin/env python
"""rt_eval.py の応答の時刻を CEJC のラベルで採点する（2026-11-02）．
各応答を，応答した時刻で正解の区間に割り当てる（オフラインの評価と同じ．直前に声が止んだ時刻はその区間の頭より後のものだけ）：
  区間の終わり 〜 次の区間の頭（最大 5 秒）に出た応答 → その区間の後の応答
  区間の途中（頭 〜 終わり）に出た応答 → 区間の途中の間での応答
（11/03：はじめは声が止んだ時刻が区間末 ＋ 0.5 秒までのものだけを数えたが，録音には他の人の声が入って silero が区間末の後も
  声ありとすることが多く，応答していても数えない区間が 27% も出たので，時刻で割り当てるように直した）
数え方はオフラインの評価（eot_eval.point_stats）と Full-Duplex-Bench（表 8，fdb_style.py）に合わせる：
  cut-in rate：（途中で応答した完了の区間 ＋ 応答した継続の区間）/（完了＋継続）
  latency：次の区間まで 2 秒以上空いた完了の区間で，区間の後の最初の応答まで（応答しなければ inf として中央値）
  Pause Handling：継続の区間で次の区間までの間が 0.4〜1.0 秒のもののうち，応答した割合
  Backchannel：相槌の区間のうち，応答した割合
  Smooth Turn-Taking：次の区間まで 2 秒以上空いた完了の区間．応答したものだけの遅れ（平均・中央値），応答しなかった割合

「silero が声を拾った区間だけ」も出す：区間の間に silero（オフラインの評価と同じ出力 tt_analysis/vad_silero）が一度も 0.5 以上に
ならない区間は，実時間では発話が始まらず応答のしようがない（オフラインでは区間の頭から無音を数えるので応答したことになる）。
Smooth の区間の 8.7% がこれに当たり，実時間とオフラインの「応答しなかった」の差のほとんどだった（11/03）。

  python rt_score.py <data（segments・tag）> <rt_eval の出力> <silero の出力（vad_silero/<data>）>
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

W, FREE = 5.0, 2.0
data, rdir, sdir = sys.argv[1], sys.argv[2], sys.argv[3]
HOP = 512 / 16000
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split()
    seg[rec].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{rdir}/*.tsv"))
models = None
res = defaultdict(list)              # model → [(ラベル, 次の区間まで, 途中で応答, 区間の後の最初の応答までの時間 or inf)]
for rec in recs:
    ev = defaultdict(list)
    for r in csv.DictReader(open(f"{rdir}/{rec}.tsv"), delimiter="\t"):
        if r["cond"] != "0":
            ev[r["model"]].append((float(r["t"]), float(r["speech_end"])))
    models = models or list(ev)
    lst = sorted(seg[rec])
    x = np.load(f"{sdir}/{rec}.npy")
    for k, (b, e, u) in enumerate(lst):
        if u not in tag:
            continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        g = max(0.0, g)
        hor = e + min(g, W)
        for m in models:
            mid = any(b <= se and b <= t < e for t, se in ev[m])
            ts = [t for t, se in ev[m] if b <= se and e <= t <= hor]
            heard = x[int(b / HOP):int(e / HOP) + 1].max(initial=0) >= 0.5
            res[m].append((tag[u], g, mid, min(ts) - e if ts else np.inf, heard))

def report(sel):
    print(f"録音 {len(recs)}")
    print("方式\tcut-in rate\tlatency の中央値\tPause Handling\tBackchannel\tSmooth：遅れの平均\tSmooth：遅れの中央値\tSmooth：応答しなかった\tSmooth：途中で応答")
    for m in models:
        a = [r for r in res[m] if sel(r)]
        lab = np.array([x[0] for x in a]); g = np.array([x[1] for x in a]); mid = np.array([x[2] for x in a]); lt = np.array([x[3] for x in a])
        took = mid | np.isfinite(lt)
        end, cont, bc = lab == 1, lab == 0, lab == 2
        cut = ((end & mid).sum() + (cont & took).sum()) / (end.sum() + cont.sum())
        st = end & (g >= FREE)
        lat_all = np.where(mid, np.inf, lt)[st & ~mid]
        ph = cont & (g >= 0.4) & (g <= 1.0)
        ok = st & ~mid & np.isfinite(lt)
        print(f"{m}\t{cut:.1%}\t{np.median(lat_all):.2f}\t{took[ph].mean():.1%}（{ph.sum()}）\t{took[bc].mean():.1%}（{bc.sum()}）"
              f"\t{lt[ok].mean():.2f}\t{np.median(lt[ok]):.2f}\t{(st & ~mid & ~np.isfinite(lt)).sum() / st.sum():.1%}\t{(st & mid).sum() / st.sum():.1%}（{st.sum()}）")


report(lambda r: True)
print("silero が声を拾った区間だけ")
report(lambda r: r[4])
