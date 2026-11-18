# gm+silero+fb の選ばれた点（無音0.8s θ=0.75 無音3s φ=1.01）で，完了区間の遅れの内訳を見る（開発データ）
import sys, numpy as np
sys.path.insert(0, "local/turntaking")
import eot_eval as E
data, dump_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
import os
G, TH = float(os.environ.get("G", 0.8)), float(os.environ.get("TH", 0.75))   # 既定は gm の点．gt などは環境変数で（2026-10-26）
S, LA, W = 3.0, 0.2, 5.0
seg, gap, tag = E.load_segments(data)
dump = E.load_dump(dump_p)
utts = [u for u in seg if u in dump]
runs = E.silence_runs(f"tt_analysis/vad_silero/{data.split('/')[-1]}", utts, seg, W)
text = dict(l.rstrip("\n").split(" ", 1) if " " in l else (l.strip(), "") for l in open(f"{data}/text"))
rows = []
for u in utts:
    if tag[u] != 1: continue
    rec, b, e = seg[u]; dur = e - b
    hor = dur + min(gap[u], W)
    t, run = runs[u]
    el, pe, pb = dump[u]
    k = np.searchsorted(el + LA, t, side="right") - 1
    p = np.where(k >= 0, pe[np.clip(k, 0, len(pe) - 1)], -1.0)
    hit = np.where((run >= G) & (p >= TH))[0]
    t_m = t[hit[0]] if len(hit) else np.inf
    hf = np.where(run >= S)[0]
    t_f = t[hf[0]] if len(hf) else np.inf
    tr = min(t_m, t_f)
    if tr > hor: tr = np.inf
    if tr < dur: continue                                   # 遮り
    lat = tr - dur
    after = t >= dur
    # 区間末の後の情報
    sil_first = t[after & (run >= G)]
    t_sil = sil_first[0] - dur if len(sil_first) else np.inf     # 無音が 0.8 秒続いた時刻（区間末から）
    win = after & (t <= hor)
    pmax = p[win].max() if win.any() else np.nan                # 区間末から範囲の終わりまでの p(完了) の最大
    tp = t[win & (p >= TH)]; t_p = tp[0] - dur if len(tp) else np.inf   # p(完了) が θ に達した時刻（区間末から）
    rmax = run[win].max() if win.any() else 0.0                 # 範囲内で続いた無音の最長
    src = "none" if not np.isfinite(lat) else ("model" if t_m <= t_f else "fallback")
    rows.append((u, lat, src, t_sil, pmax, min(gap[u], 99), text.get(u, ""), t_p, rmax))
rows.sort(key=lambda r: -r[1] if np.isfinite(r[1]) else -1e9)
lat = np.array([r[1] for r in rows])
print("完了区間（遮らなかったもの）", len(rows), " 中央値", np.quantile(lat, .5), " 90%点", np.quantile(lat, .9))
for name, m in [("≤1s", lat <= 1), ("1-2s", (lat > 1) & (lat <= 2)), ("2-3s", (lat > 2) & (lat <= 3)), (">3s", (lat > 3) & np.isfinite(lat)), ("応答なし", ~np.isfinite(lat))]:
    print(name, m.sum(), f"{m.mean():.3f}")
long = [r for r in rows if r[1] > 2]
print("--- 遅れ 2 秒超", len(long))
from collections import Counter
print("応答のきっかけ", Counter(r[2] for r in long))
ts = np.array([r[3] for r in long]); pm = np.array([r[4] for r in long])
print("無音 0.8 秒に達した時刻（区間末から）：", "なし", np.isinf(ts).sum(), " ≤1s", (ts <= 1).sum(), " 1-2s", ((ts > 1) & (ts <= 2)).sum(), " >2s", ((ts > 2) & np.isfinite(ts)).sum())
print("区間末以降の p(完了) の最大：", " <0.3", (pm < .3).sum(), " 0.3-0.5", ((pm >= .3) & (pm < .5)).sum(), " 0.5-0.75", ((pm >= .5) & (pm < .75)).sum(), " ≥0.75", (pm >= .75).sum())
with open(out, "w") as f:
    f.write("utt\t遅れ\tきっかけ\t無音0.8sの時刻\tp完了の最大\t次の発話まで\tpがθに達した時刻\t無音の最長\ttext\n")
    for r in rows:
        f.write("\t".join([r[0], f"{r[1]:.2f}", r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.2f}", f"{r[7]:.2f}", f"{r[8]:.2f}", r[6]]) + "\n")
