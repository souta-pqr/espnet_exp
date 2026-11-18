#!/usr/bin/env python
"""録音を実時間のプログラム（epd.engine）に流し，応答はさせずに，各時刻の silero の出力とスコアを書き出す（2026-11-12）．
応答の決め方を，実時間のスコアで後から何通りも試すため（rt_rule.py）．
「今の声のかたまりだけ」を入れるモデル（--encac_ctx p0）では，スコアは無音ごとに決まり，前に応答したかどうかによらないので，
後から決め方を当てはめた結果は実際に流した結果とほぼ一致する（rt_rule.py --check で確かめる）．

  cd ~/2026/work/exp-realtime-epd
  python <この場所>/rt_dump.py --wav_scp <data>/wav.scp --out <出力> --shard 0 --nshard 6 --model encac --device cuda --encac_ctx p0 --encac_cost ...
出力：<録音>.npz（p・run：32 ms ごとの silero の確率と無音の長さ，st・ss・sb：スコアが出た時刻（秒）・スコア・P(相槌)）
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())
from epd.rule import Rule
from epd.vad import HOP, SR
from run_live import add_model_args, build
from rt_eval import cap


def run_one(a, rec, path):
    t0 = time.perf_counter()
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR and x.ndim == 1, (path, sr, x.shape)
    (_, _, e, _), = build(a)
    e.rule = Rule()                       # 応答しない（発話は無音 5 秒で終わる）
    cap(e.scorer)                         # rt_eval.py と同じ（提案手法が見る発話の長さの上限）
    P, R, sc = [], [], []
    vad, push = e.vad, e.scorer.push

    def vad2(c):
        p, run = vad(c)
        P.append(p); R.append(run)
        return p, run

    def push2(c, silent=True):
        out = push(c, silent=silent)
        sc.extend(((e.n + HOP) / SR, s, 0.0 if b is None else b) for _, s, b in out)
        return out
    e.vad, e.scorer.push = vad2, push2
    if a.model != "encac":                # 11/13：ほかの方式．発話の頭ではスコアがまだ無い（−1 を入れておく）
        start = e._start

        def start2():
            sc.append((e.n / SR, -1.0, 0.0))
            return start()
        e._start = start2
    for i in range(0, len(x) - HOP + 1, HOP):
        e.feed(x[i:i + HOP])
    sc = np.array(sc, np.float64).reshape(-1, 3)
    np.savez(f"{a.out}/{rec}.tmp.npz", p=np.array(P, np.float32), run=np.array(R, np.float32), st=sc[:, 0], ss=sc[:, 1].astype(np.float32), sb=sc[:, 2].astype(np.float32))
    os.replace(f"{a.out}/{rec}.tmp.npz", f"{a.out}/{rec}.npz")
    el = time.perf_counter() - t0
    print(f"{rec} {len(x) / SR / 60:.1f} 分を {el / 60:.1f} 分", flush=True)


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [l.split() for l in open(a.wav_scp) if l.strip()]
    recs = [r for i, r in enumerate(sorted(recs)) if i % a.nshard == a.shard]
    ctx = mp.get_context("spawn")
    for rec, path in recs:
        if os.path.exists(f"{a.out}/{rec}.npz"):
            continue
        p = ctx.Process(target=run_one, args=(a, rec, path))
        p.start(); p.join()
        if p.exitcode:
            print(f"{rec}: 失敗（終了コード {p.exitcode}）", flush=True)


if __name__ == "__main__":
    main()
