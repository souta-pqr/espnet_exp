#!/usr/bin/env python
"""前向きグリッドの各時点における「直近 0.25 秒のエネルギー」を計算して保存する。

停止ルールの状態に無音の手がかりを入れるため。実運用では音声区間検出が同じ情報を
持っているので、これを使うのは因果的に正当（発話末の位置は使っていない）。
確率ダンプには含まれない情報なので、元録音から直接計算する。

  python local/turntaking/make_energy.py eval
"""
import sys
import numpy as np
import soundfile as sf

GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
WIN = 0.25          # 直近このぶんのエネルギーを見る


def main(dset):
    full = {}
    for line in open(f"data/{dset}/wav.scp"):
        q = line.split(maxsplit=1)
        if len(q) == 2: full[q[0]] = q[1].strip()
    segs = []
    for line in open(f"data/{dset}/segments"):
        u, r, b, e = line.split(); segs.append((u, r, float(b), float(e)))

    out = {}
    for k, (u, rec, b, e) in enumerate(segs):
        if rec not in full:
            continue
        try:
            w = sf.read(full[rec], dtype="float32",
                        start=int(b * 16000), frames=int(GRID[-1] * 16000),
                        always_2d=False)[0]
        except Exception:
            continue
        if w.ndim > 1: w = w[:, 0]
        v = []
        for g in GRID:
            hi = min(int(g * 16000), len(w))
            lo = max(0, hi - int(WIN * 16000))
            seg = w[lo:hi]
            v.append(float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0)
        out[u] = v
        if (k + 1) % 5000 == 0:
            print(f"  ...{k+1}/{len(segs)}", flush=True)
    np.savez(f"tt_preds/energy_{dset}.npz",
             utt=np.array(list(out.keys())), val=np.array(list(out.values())))
    print(f"保存: tt_preds/energy_{dset}.npz  {len(out)} 区間 × {len(GRID)} 点")


if __name__ == "__main__":
    main(sys.argv[1])
