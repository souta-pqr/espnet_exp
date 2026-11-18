#!/usr/bin/env python
"""学習データ train_gx に「間を含む長い発話」を足した train_gxm を作る（2026-11-04）．

実時間では，モデルが継続の区間の後で待つと発話が切れず，「前の区間＋間＋今の区間」がひとつながりでモデルに入る．
学習は区間の頭から始まる音声だけだったので，この形で話を取る誤りが多かった（実時間でだけ話を取る区間の 9 割）．
そこで，継続の区間 u0 の後に同じ話者の区間 u1 が 0.4〜3 秒の間で続く組ごとに，次の学習例を足す．
  音声：u0 の flac ＋ 間の音声（元の録音から切り出す）＋ u1 の flac（concat_sound で読むときにつなぐ）
  ラベル・区間末の後の 2 秒：u1 のもの　過去の文脈：u0 のもの（u0 より前の発話）　書き起こし：u0 ＋ u1
ディスクに書くのは間の音声だけ（約 1 GB）．既存の区間はそのまま残す．

  cd ~/2026/espnet_tt/egs2/cejc/asr1 && python <ここ>/make_gxm.py [--limit N]
"""
import argparse
import os
import shutil
from collections import defaultdict

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="dump/raw/train_gx")
ap.add_argument("--dst", default="dump/raw/train_gxm")
ap.add_argument("--gap_dir", default="dump/raw/org/train_gxm/gap")
ap.add_argument("--stats", default="exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail")
ap.add_argument("--stats_dst", default="exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail_gxm")
ap.add_argument("--gmin", type=float, default=0.4)
ap.add_argument("--gmax", type=float, default=3.0)
ap.add_argument("--limit", type=int, default=0, help="試しに作る組の数（0 なら全部）")
a = ap.parse_args()
SR = 16000


def read(p):
    """key → 行の残り（文字列のまま）"""
    d = {}
    for l in open(p, encoding="utf-8"):
        k, _, v = l.rstrip("\n").partition(" ")
        d[k] = v
    return d


src = {f: read(f"{a.src}/{f}") for f in ("wav.scp", "tag", "text", "utt2spk", "utt2num_samples", "past_speech.scp",
                                          "past_speech_same_n5_p30.scp", "past_bounds.scp",
                                          "past_speech_same_n5_p30_bounds.scp", "tail_speech.scp")}
rwav = read("data/train_gx/wav.scp")
seg = {}
for l in open("data/train_gx/segments"):
    u, rec, b, e = l.split()
    seg[u] = (rec, float(b), float(e))
by = defaultdict(list)
for u in src["tag"]:
    rec, b, e = seg[u]
    by[rec].append((b, e, u))

pairs = []
for rec, L in by.items():
    L.sort()
    for (b0, e0, u0), (b1, e1, u1) in zip(L, L[1:]):
        if src["tag"][u0].strip() == "0" and a.gmin <= b1 - e0 < a.gmax:
            pairs.append((rec, b0, e0, u0, b1, e1, u1))
if a.limit:
    pairs = pairs[:a.limit]
print("組", len(pairs), flush=True)

os.makedirs(a.gap_dir, exist_ok=True)
gap_abs = os.path.abspath(a.gap_dir)
new = {f: {} for f in src}
shape = defaultdict(dict)
for i, (rec, b0, e0, u0, b1, e1, u1) in enumerate(pairs):
    k = f"{rec}_{round(b0 * 1000):07d}_{round(e1 * 1000):07d}m"          # 末尾の m で元の区間と区別する
    gp = f"{gap_abs}/{k}.flac"
    if not os.path.exists(gp):
        x, sr = sf.read(rwav[rec], start=int(round(e0 * SR)), stop=int(round(b1 * SR)), dtype="int16")
        assert sr == SR
        sf.write(gp, x, SR, format="FLAC")
    n0, n1 = int(src["utt2num_samples"][u0]), int(src["utt2num_samples"][u1])
    ng = sf.info(gp).frames
    new["wav.scp"][k] = f"{src['wav.scp'][u0]} {gp} {src['wav.scp'][u1]}"
    new["tag"][k] = src["tag"][u1]
    new["text"][k] = f"{src['text'][u0]} {src['text'][u1]}"
    new["utt2spk"][k] = src["utt2spk"][u1]
    new["utt2num_samples"][k] = str(n0 + ng + n1)
    for f in ("past_speech.scp", "past_speech_same_n5_p30.scp", "past_bounds.scp", "past_speech_same_n5_p30_bounds.scp"):
        new[f][k] = src[f][u0]
    new["tail_speech.scp"][k] = src["tail_speech.scp"][u1]
    shape["src"][k] = (u0, u1, n0 + ng + n1)
    if i % 5000 == 0:
        print(i, flush=True)

# データの場所：既存の行＋足した行（キーの順に並べる）
os.makedirs(a.dst, exist_ok=True)
for f, d in src.items():
    with open(f"{a.dst}/{f}", "w", encoding="utf-8") as o:
        for k in sorted(list(d) + list(new[f])):
            o.write(f"{k} {d[k] if k in d else new[f][k]}\n")
shutil.copy(f"{a.src}/feats_type", f"{a.dst}/feats_type")
spk = defaultdict(list)
for l in open(f"{a.dst}/utt2spk"):
    k, s = l.split()
    spk[s].append(k)
with open(f"{a.dst}/spk2utt", "w") as o:
    for s in sorted(spk):
        o.write(s + " " + " ".join(spk[s]) + "\n")

# 統計の場所：valid と平均・分散はそのまま，train の形の一覧に足した行を加える
if os.path.exists(a.stats_dst):
    shutil.rmtree(a.stats_dst)
shutil.copytree(a.stats, a.stats_dst, ignore=shutil.ignore_patterns("logdir"))
tr = f"{a.stats_dst}/train"
for f in os.listdir(tr):
    if f.endswith(".npz"):
        continue
    d = read(f"{tr}/{f}")
    for k, (u0, u1, n) in shape["src"].items():
        if f == "speech_shape":
            d[k] = str(n)
        elif f.startswith("text_shape"):
            l0, l1 = d[u0].split(",")[0], d[u1].split(",")[0]
            d[k] = ",".join([str(int(l0) + int(l1))] + d[u1].split(",")[1:])
        elif f.startswith("past_"):
            d[k] = d[u0]
        else:
            d[k] = d[u1]
    with open(f"{tr}/{f}", "w") as o:
        for k in sorted(d):
            o.write(f"{k} {d[k]}\n")
print("完了", len(pairs), "組を足した →", a.dst, a.stats_dst)
