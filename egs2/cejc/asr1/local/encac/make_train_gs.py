#!/usr/bin/env python
"""実時間と同じ切り方の学習データ data/<出力> を作る（2026-11-19）．

実時間（exp-realtime-epd の epd/models/encac.py，ctx="p0"）では，silero-vad の確率 0.5 未満を無音とし，0.4 秒以上の無音で区切った
音声区間の頭（音声の始まりの 0.2 秒前）から今までを入れる．判定するのは 0.4 秒以上続いた無音の間．
そこで，0.4 秒以上の無音ごとに 1 つ，その時点で実時間のモデルに入る音声（音声区間の頭 〜 無音の頭）を区間として書く．
train_ac.py はこの区間の終わり（＝無音の頭）の後 0〜2 秒を足して読むので，実時間で判定する入力と同じになる．
ラベル：
  - 無音が書き起こしの区間の中にある（区間の頭 +0.1 秒より後に始まり，区間末 −0.1 秒より前に終わる）→ 継続
  - それ以外は，無音の頭 +0.3 秒までに終わった同じ話者の最後の区間のラベル．その区間の後で最初の 0.4 秒の無音だけを使い，
    無音の頭が区間末の 5 秒より後なら使わない（評価で応答をその区間に割り当てない）
書き起こし（音声認識の補助タスク用）：音声区間に入る書き起こしの区間の文字をつなぐ．音声区間の頭が書き起こしの区間の途中にあるとき
（区間の中の無音で区切られたとき）と，区間の中の無音のときは，文字と音声が合わないので空にする（train_ac.py は CTC 損失を取らない）．

  cd egs2/cejc/asr1 && python make_train_gs.py data/train_gx tt_analysis/vad_silero/train_g data/train_gs
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np

src, sd, out = sys.argv[1:4]
H = 512 / 16000
tag = dict(l.split() for l in open(f"{src}/tag"))
txt = {}
for l in open(f"{src}/text"):
    u, *t = l.rstrip("\n").split(" ", 1)
    txt[u] = t[0] if t else ""          # 1 文字ずつ空白で区切った形のまま（train_et_data.norm が「(F 」などを消してから空白を除く）
segs = defaultdict(list)
for l in open(f"{src}/segments"):
    u, r, b, e = l.split()
    segs[r].append((float(b), float(e), u))

os.makedirs(out, exist_ok=True)
fs, ft, fx = open(f"{out}/segments", "w"), open(f"{out}/tag", "w"), open(f"{out}/text", "w")
n = Counter()
for r in sorted(segs):
    ss = sorted(segs[r])
    p = np.load(f"{sd}/{r}.npy").astype(np.float32)
    sp = p >= 0.5
    ed = np.diff(np.r_[0, (~sp).astype(int), 0])
    s0, s1 = np.where(ed == 1)[0], np.where(ed == -1)[0]
    lg = (s1 - s0) * H >= 0.4
    ls0, ls1 = s0[lg] * H, s1[lg] * H
    on = np.where(np.diff(np.r_[0, sp.astype(int)]) == 1)[0]
    if len(on) == 0:
        continue
    heads = np.r_[on[0] * H, ls1] - 0.2
    bs = np.array([b for b, _, _ in ss]); es = np.array([e for _, e, _ in ss])
    used = set()
    for s in ls0:
        hh = heads[heads < s]
        if len(hh) == 0:
            continue
        h = max(0.0, hh[-1])
        inside = np.where((bs + 0.1 < s) & (s + 0.4 < es - 0.1))[0]
        if len(inside):
            lab, t, kind = "0", "", "区間の中の無音（継続）"
        else:
            k = np.where(es <= s + 0.3)[0]
            if len(k) == 0:
                continue
            k = k[-1]
            heard = sp[int(bs[k] / H):int(es[k] / H) + 1].any()   # silero が一度も声を拾わない区間は，実時間でも判定しようがない
            if k in used or s > es[k] + 5.0 or h >= es[k] or not heard:
                n["使わない無音"] += 1
                continue
            used.add(k)
            u = ss[k][2]
            lab = tag[u]
            part = np.where((bs < h - 0.3) & (es > h + 0.3))[0]
            full = [ss[i][2] for i in range(len(ss)) if bs[i] >= h - 0.3 and es[i] <= s + 0.3]
            t = "" if len(part) else " ".join(txt[v] for v in full)
            kind = {"0": "継続", "1": "完了", "2": "相槌"}[lab] + ("（複数の区間）" if len(full) > 1 else "") + ("（文字なし）" if not t else "")
        n[kind] += 1
        uid = f"{r}_{int(round(h * 1000)):07d}_{int(round(s * 1000)):07d}s"
        fs.write(f"{uid} {r} {h:.3f} {s:.3f}\n"); ft.write(f"{uid} {lab}\n"); fx.write(f"{uid} {t}\n")
for f in (fs, ft, fx):
    f.close()
open(f"{out}/wav.scp", "w").write(open(f"{src}/wav.scp").read())
for k, v in sorted(n.items()):
    print(f"{k}\t{v}")
print("計", sum(v for k, v in n.items() if k != "使わない無音"), "（元の区間", sum(len(v) for v in segs.values()), "）")
