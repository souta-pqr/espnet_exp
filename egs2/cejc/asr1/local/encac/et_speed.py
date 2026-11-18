"""Easy Turn を 1 本ずつ（バッチ 1）実時間で使ったときの計算時間を測る（11/02，silver13）。
区切り（途中の間の始まり・区間末）ごとに書き起こし 1 回＋タグ 1 回。区間の頭からの音声を毎回与える。
  python et_speed.py <best.pt> dev_g [区切りの数]
"""
import sys, os, time, numpy as np, torch
sys.argv += [] ; ck, d = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
sys.argv = [sys.argv[0], ck, d, "/dev/null", "400"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "et_dump.py")).read().split("t0 = time.time()")[0]
exec(src)
keys = []
for u in utts:
    rec, b, e = seg[u]; t, run = runs[u]
    st = t[(run > 0) & (np.r_[0, run[:-1]] == 0)] - EV.SIL_HOP
    keys += [(u, float(s)) for s in st[(st > 0.1) & (st < e - b - 0.3)]] + [(u, e - b)]
keys = keys[:N]
for u, c in keys[:5]:   # 暖機
    f = E.logmel(audio(seg[u][0], seg[u][1], c)); E.tag_probs(m, [f], transcribe([f]))
tr, tg, dur = [], [], []
for u, c in keys:
    torch.cuda.synchronize(); a = time.time()
    f = E.logmel(audio(seg[u][0], seg[u][1], c)); s = transcribe([f])
    torch.cuda.synchronize(); b2 = time.time()
    E.tag_probs(m, [f], s)
    torch.cuda.synchronize(); tr.append(b2 - a); tg.append(time.time() - b2); dur.append(c)
tr, tg, dur = map(np.array, (tr, tg, dur))
q = lambda x: " ".join(f"{p}%:{np.percentile(x, p):.3f}" for p in (50, 90, 99))
print(f"区切り {len(keys)} 個  音声の長さ 中央 {np.median(dur):.2f} 秒")
print("書き起こし（秒）", q(tr)); print("タグ（秒）", q(tg)); print("合計（秒）", q(tr + tg))
