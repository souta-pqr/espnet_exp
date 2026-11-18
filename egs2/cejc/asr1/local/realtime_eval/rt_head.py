"""話を取った継続の区間で，実時間の発話の頭が正解の区間の頭よりどれだけ前か（2026-11-03）"""
import csv, sys
import numpy as np
cut, off, rt = sys.argv[1:4]
offm = {(r["rec"], r["b"]): float(r["off_max"]) for r in csv.DictReader(open(off), delimiter="\t")}
res = []
for r in csv.DictReader(open(cut), delimiter="\t"):
    b, e = float(r["b"]), float(r["e"]); t = e + float(r["区間末から応答まで"])
    ev = [x for x in csv.DictReader(open(f"{rt}/{r['rec']}.tsv"), delimiter="\t") if x["model"] == "proposed" and abs(float(x["t"]) - t) < 0.02]
    us = float(ev[0]["utt_start"])
    res.append((r["間の他の人"], offm[(r["rec"], r["b"])] >= 0.84, b - us, e - b))
for o in ("誰も話さない", "他の人が話した"):
    for sh in (True, False):
        d = np.array([x[2] for x in res if x[0] == o and x[1] == sh])
        print(o, "オフラインでも" if sh else "実時間だけ", len(d), "件：頭のずれ（正解−実時間）",
              "中央値 %.2f 秒" % np.median(d), "| 0.5 秒以内 %d | 0.5〜3 秒 %d | 3 秒より前 %d" % ((abs(d) <= 0.5).sum(), ((d > 0.5) & (d <= 3)).sum(), (d > 3).sum()))
