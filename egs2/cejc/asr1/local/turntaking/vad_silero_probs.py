#!/usr/bin/env python
"""各話者チャネルの録音全体に silero-vad をかけ、32 ms ごとの音声確率を保存する（2026-10-09）。

実際の VAD で「発話区間末」を決めたときの評価（vad_real_eval.py）の前処理。
閾値や無音長はあとから変えられるよう、確率そのものを保存する（1 チャネル 20 分で約 150KB）。
silero-vad 6.2.3（pip install --no-deps silero-vad で espnet 環境に追加）、CPU・1 スレッドで動かす。

  python local/turntaking/vad_silero_probs.py --dset eval --nj 4
出力：tt_analysis/vad_silero/<dset>/<録音ID>.npy（float16, 32 ms 刻み = 512 サンプル）
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

OUT = Path("tt_analysis/vad_silero")


def run(job):
    rec, path, out = job
    if out.exists():
        return rec, "skip"
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav[:, 0]
    assert sr == 16000, (path, sr)
    with torch.no_grad():
        p = model.audio_forward(torch.from_numpy(wav), sr).squeeze(0).numpy()
    np.save(out, p.astype(np.float16))
    return rec, f"{len(p)} frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dset", required=True)
    ap.add_argument("--nj", type=int, default=4)
    a = ap.parse_args()
    d = OUT / a.dset
    d.mkdir(parents=True, exist_ok=True)
    jobs = []
    for l in open(f"data/{a.dset}/wav.scp"):
        rec, path = l.split(maxsplit=1)
        jobs.append((rec, path.strip(), d / f"{rec}.npy"))
    with mp.get_context("spawn").Pool(a.nj) as pool:
        for i, (rec, msg) in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"[{i}/{len(jobs)}] {rec} {msg}", flush=True)


if __name__ == "__main__":
    main()
