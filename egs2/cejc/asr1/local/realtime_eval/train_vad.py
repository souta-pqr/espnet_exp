#!/usr/bin/env python
"""録音ごとに silero-vad の確率（32 ms ごと）を書き出す（2026-11-14．学習データの声のかたまりを実時間と同じ切り方で求めるため）．
  python train_vad.py <wav.scp> <出力の場所> <shard> <nshard>      exp-realtime-epd の中で動かす（epd.vad を使う）"""
import os, sys, wave
import numpy as np, torch
from epd.vad import SileroVAD, HOP
torch.set_num_threads(1)
scp, out, i, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
recs = sorted(l.split() for l in open(scp) if l.strip())
v = SileroVAD()
for rec, path in recs[i::k]:
    if os.path.exists(f"{out}/{rec}.npy"): continue
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    v.reset()
    p = np.array([v(x[j:j + HOP])[0] for j in range(0, len(x) - HOP + 1, HOP)], np.float32)
    np.save(f"{out}/{rec}.tmp.npy", p); os.replace(f"{out}/{rec}.tmp.npy", f"{out}/{rec}.npy")
    print(rec, len(p), flush=True)
