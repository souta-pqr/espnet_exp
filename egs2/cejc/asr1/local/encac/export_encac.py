"""音響だけのモデル（~/2026/work/easy_turn/cejc の train_ac.py．エンコーダは提案手法と同じ contextual block conformer）の重みを，
デモ（epd/models/encac.py）で読める形に書き出す．照合用に，開発データの 1 区間で出した確率も残す（tools/check_encac.py で比べる）．

  cd ~/2026/work/easy_turn/cejc
  ENC=espnet:<pth> PYTHONPATH=~/2025/espnet ~/anaconda3/envs/espnet/bin/python ~/2026/work/exp-realtime-epd/tools/export_encac.py \
      ~/2025/espnet/egs2/cejc/asr1/exp/encac_espm_20261104/best.pt ~/2026/work/exp-realtime-epd/weights/encac_espm.pt
"""
import os
import sys
import wave

import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
import enc_plug as P

src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(P.GTCONF))
m = P.ESP()
m.load_state_dict(torch.load(src, map_location="cpu"))
m.eval()
torch.save({"frontend_conf": cfg["frontend_conf"], "encoder_conf": cfg["encoder_conf"],
            "state_dict": m.state_dict(), "tail": P.TAIL, "pool": P.POOL}, dst)

# 照合用：開発データの区間の頭から，区間末 + 0〜3 秒の各時刻まで（0.1 秒ごと）の音声で確率を出す
utt = "K007_001_IC01_0023265_0024669"
rec = utt.rsplit("_", 2)[0]
b, e = int(utt.rsplit("_", 2)[1]) / 1000, int(utt.rsplit("_", 2)[2]) / 1000
wavs = dict(l.split() for l in open(f"{P.ASR1}/data/dev_g/wav.scp"))
w = wave.open(wavs[rec])
w.setpos(int(round(b * 16000)))
x = np.frombuffer(w.readframes(int(round((e - b + 3.2) * 16000))), dtype=np.int16).astype(np.float32) / 32768
ts = np.round(np.arange(1, int((e - b + 3.0) / 0.1) + 1) * 0.1, 4)
with torch.no_grad():
    prob = np.stack([m([torch.from_numpy(x[:int(round(t * 16000))])]).softmax(-1)[0].numpy() for t in ts])
np.savez(dst.replace(".pt", "_ref.npz"), utt=utt, x=x, ts=ts, prob=prob)
print(f"書き出した：{dst}．照合用の区間 {utt}（{len(ts)} 時刻）")
