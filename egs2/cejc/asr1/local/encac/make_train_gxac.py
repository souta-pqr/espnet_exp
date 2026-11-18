#!/usr/bin/env python
"""提案手法の学習データ data/train_gxac を作る（2026-11-11 に silver13 でその場で作ったものを，スクリプトに書き起こした）．

train_gxac = train_gx の全区間 ＋「前の区間 間 今の区間」をつないだ区間（dump/raw/train_gxm の ID が m で終わるもの）．
つないだ区間は，前の区間の頭から今の区間の終わりまでを 1 つの区間として segments に書く（録音から切り出すだけで，音声は複製しない）．
ラベルと書き起こしは train_gxm のもの（今の区間のラベル）を使う．

  cd egs2/cejc/asr1 && python local/encac/make_train_gxac.py
"""
import os
import re

src, m, out = "data/train_gx", "dump/raw/train_gxm", "data/train_gxac"
os.makedirs(out, exist_ok=True)
gx = set(l.split()[0] for l in open(f"{src}/segments"))
tag = dict(l.split() for l in open(f"{m}/tag"))
txt = {l.split(" ", 1)[0]: l.rstrip("\n") for l in open(f"{m}/text")}
new = [u for u in tag if u.endswith("m")]
wav = dict(l.split() for l in open(f"{src}/wav.scp"))
seg = open(f"{src}/segments").read().splitlines()
bad = 0
for u in new:
    mm = re.match(r"(.+)_(\d{7})_(\d{7})m$", u)
    rec, b, e = mm.group(1), int(mm.group(2)) / 1000, int(mm.group(3)) / 1000
    if rec not in wav:
        bad += 1
        continue
    seg.append(f"{u} {rec} {b:.3f} {e:.3f}")
open(f"{out}/segments", "w").write("\n".join(seg) + "\n")
gtag = dict(l.split() for l in open(f"{src}/tag"))
gtag.update({u: tag[u] for u in new})
open(f"{out}/tag", "w").write("".join(f"{u} {t}\n" for u, t in gtag.items()))
gtxt = {l.split(" ", 1)[0]: l.rstrip("\n") for l in open(f"{src}/text")}
gtxt.update({u: txt.get(u, u) for u in new})
open(f"{out}/text", "w").write("\n".join(gtxt.values()) + "\n")
open(f"{out}/wav.scp", "w").write(open(f"{src}/wav.scp").read())
print("元", len(gx), "足した", len(new) - bad, "録音なし", bad)
