#!/usr/bin/env python
"""学習データ train_gxv を作る（2026-11-14）：train_gx の各区間の頭を，実時間で入れるのと同じ「今の声のかたまりの頭」に置き換える．
声のかたまりの切り方は epd/models/encac.py の ctx="p0" と同じ：silero の無音（確率 0.35 未満で数え始め，0.5 以上で戻る）が
0.4 秒以上続いたあと声が戻った時刻の 0.2 秒前が，新しいかたまりの頭．
区間末より前にかたまりの頭が無い区間（silero が声を拾っていない）は元の頭のまま．tag・text・発話の名前は変えない．
  python make_train_gxv.py <asr1> <silero の出力の場所>"""
import os, sys
import numpy as np
A, V = sys.argv[1], sys.argv[2]
HOP, PAUSE, PRE = 512 / 16000, 0.4, 0.2
src, dst = f"{A}/data/train_gx", f"{A}/data/train_gxv"
os.makedirs(dst, exist_ok=True)
starts = {}
def chunk_starts(rec):
    p = np.load(f"{V}/{rec}.npy")
    out, s = [], 0          # s：無音を数え始めたフレーム（録音の頭は無音の扱い），-1＝声
    for j, v in enumerate(p):
        if v >= 0.5:
            if s >= 0 and (j - s + 1) * HOP >= PAUSE: out.append(max(0.0, j * HOP - PRE))
            s = -1
        elif s < 0 and v < 0.35:
            s = j
    return np.array(out)
n = keep = 0; d0, d1 = [], []
with open(f"{dst}/segments", "w") as f:
    for l in open(f"{src}/segments"):
        u, rec, b, e = l.split(); b, e = float(b), float(e)
        if rec not in starts: starts[rec] = chunk_starts(rec)
        c = starts[rec]; i = np.searchsorted(c, e - 0.1, "right") - 1
        if i < 0: nb = b; keep += 1
        else: nb = float(c[i])
        n += 1; d0.append(e - b); d1.append(e - nb)
        f.write(f"{u} {rec} {nb:.3f} {e:.3f}\n")
for k in ("tag", "text", "wav.scp"):
    open(f"{dst}/{k}", "w").write(open(f"{src}/{k}").read())
d0, d1 = np.array(d0), np.array(d1)
print(f"区間 {n}・元の頭のまま {keep}")
print("頭から区間末までの長さ（秒）の 10・50・90・99% 点  元：", np.percentile(d0, [10, 50, 90, 99]).round(2), " かたまり：", np.percentile(d1, [10, 50, 90, 99]).round(2))
print(f"かたまりのほうが 0.3 秒以上長い {np.mean(d1 - d0 > 0.3):.1%}・0.3 秒以上短い {np.mean(d0 - d1 > 0.3):.1%}・12 秒より長い {np.mean(d1 > 12):.1%}")
