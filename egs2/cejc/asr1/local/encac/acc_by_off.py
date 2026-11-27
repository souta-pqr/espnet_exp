"""判定する時刻（区間末から何秒後か）ごとの正解率（2026-11-26）．
最後のフレームで判定すると正解率が下がるのは，判定する時刻が無音の中にあるからか，を確かめる．
検証データ（dev_g から学習と同じ乱数で選んだ 4000 区間）の頭の N 区間を，区間末の後 OFFS 秒の各時刻まで入れて判定する．
  cd ~/2026/work/encac_pool
  ENC=... PYTHONPATH=~/2025/espnet python acc_by_off.py <best.pt> <POOL> [N]
"""
import os, sys, wave
import numpy as np, torch
sys.path.insert(0, os.getcwd())
import enc_plug as P
from train_et_data import load, SR

w, pool = sys.argv[1], int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
OFFS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
CLS = {"<incomplete>": 0, "<complete>": 1, "<backchannel>": 2}
P.POOL = pool
m = P.ESP()
m.load_state_dict(torch.load(w, map_location="cpu"))
m = m.cuda().eval()
va = load("dev_g")
va = [va[i] for i in np.random.default_rng(0).permutation(len(va))[:4000]][:N]   # train_ac.py の検証データと同じ並び
y = np.array([CLS[t] for *_, t in va])
pred = np.zeros((len(OFFS), N), int)
with torch.no_grad():
    for i, (path, b, e, _, _) in enumerate(va):
        wv = wave.open(path)
        xs = []
        for off in OFFS:
            s1 = min(int(round((e + off) * SR)), wv.getnframes())
            s0 = max(int(round(b * SR)), s1 - 12 * SR)       # 学習と同じ（MAXSEC=12）
            wv.setpos(s0)
            xs.append(torch.from_numpy(np.frombuffer(wv.readframes(max(s1 - s0, 400)), dtype=np.int16).astype(np.float32) / 32768))
        pred[:, i] = m(xs).argmax(-1).cpu().numpy()
print(f"{os.path.basename(os.path.dirname(w))} POOL={pool}  {N} 区間（継続 {np.mean(y == 0):.0%}・完了 {np.mean(y == 1):.0%}・相槌 {np.mean(y == 2):.0%}）")
print("区間末の後(秒)\t正解率\t継続の再現率\t完了の再現率\t相槌の再現率")
for k, off in enumerate(OFFS):
    ok = pred[k] == y
    print(f"{off:.1f}\t{ok.mean():.3f}\t" + "\t".join(f"{ok[y == c].mean():.3f}" for c in range(3)))
