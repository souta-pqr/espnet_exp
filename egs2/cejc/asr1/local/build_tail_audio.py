#!/usr/bin/env python
"""発話後の音声（tail）を元録音から切り出して tail_speech.scp を作る。

打ち切り学習は「発話を途中で切る」ことしか教えておらず、「発話の後に無音が続く」
状態を学習時に一度も見せていない（早期確定 11 節）。実運用のストリーミングでは
区間検出が発火するまでの 0.25〜0.5 秒、場合によってはそれ以上の無音（や次の発話）を
モデルが聞くので、その音声を学習データとして用意する。

各区間 u について、元録音の [end, end + tail_sec] を flac に書き出す。
録音末尾で足りなければあるだけ（最低 10 ms の無音）。

出力:
  dump/raw/org/<set>/tail/<utt>.flac
  dump/raw/<set>/tail_speech.scp     … <utt> <絶対パス>

  python local/build_tail_audio.py --dset train_nodup --tail_sec 1.5 --nj 16
"""
import argparse
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1].strip()
    return m


def work(job):
    reco_path, segs, out_dir, tail_n = job
    out_dir = Path(out_dir)
    rows = []
    with sf.SoundFile(reco_path) as f:
        assert f.samplerate == SR, (reco_path, f.samplerate)
        total = f.frames
        for u, end in segs:
            start = int(round(end * SR))
            n = max(0, min(tail_n, total - start))
            if n > 0:
                f.seek(start)
                x = f.read(n, dtype="int16", always_2d=False)
                if x.ndim > 1:
                    x = x[:, 0]
            else:
                x = np.zeros(0, dtype="int16")
            if len(x) < SR // 100:                       # 10 ms 未満なら無音で埋める
                x = np.concatenate([x, np.zeros(SR // 100 - len(x), dtype="int16")])
            p = out_dir / f"{u}.flac"
            sf.write(str(p), x, SR, format="FLAC", subtype="PCM_16")
            rows.append((u, str(p.resolve()), len(x)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dset", required=True)
    ap.add_argument("--tail_sec", type=float, default=1.5)
    ap.add_argument("--nj", type=int, default=16)
    args = ap.parse_args()

    data = Path("data") / args.dset
    dump = Path("dump/raw") / args.dset
    out_dir = Path("dump/raw/org") / args.dset / "tail"
    out_dir.mkdir(parents=True, exist_ok=True)

    reco = read_map(data / "wav.scp")
    utts = set(read_map(dump / "wav.scp"))
    by_reco = defaultdict(list)
    with open(data / "segments", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and p[0] in utts:
                by_reco[p[1]].append((p[0], float(p[3])))
    tail_n = int(args.tail_sec * SR)
    jobs = [(reco[r], segs, str(out_dir), tail_n) for r, segs in by_reco.items()]
    print(f"[{args.dset}] 録音 {len(jobs)} 本 / 区間 {sum(len(s) for _, s in by_reco.items())}",
          flush=True)
    rows = []
    with Pool(args.nj) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  ...{i}/{len(jobs)} 録音", flush=True)
    rows.sort()
    with open(dump / "tail_speech.scp", "w", encoding="utf-8") as f:
        for u, p, _ in rows:
            f.write(f"{u} {p}\n")
    ns = np.array([n for _, _, n in rows])
    print(f"[{args.dset}] 書き出し {len(rows)} 件 / 平均 {ns.mean()/SR:.2f} 秒 / "
          f"満尺 {(ns >= tail_n).mean()*100:.1f}%  → {dump/'tail_speech.scp'}", flush=True)


if __name__ == "__main__":
    main()
