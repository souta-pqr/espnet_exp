#!/usr/bin/env python
"""凍結 ASR エンコーダ特徴 + 未来VAラベル を抽出してシャードに保存.

各区間を [start, end+cap] で元録音から読み、frozen encoder で encoder_out (T,256)。
同時に segments から「未来VA」自己教師ラベルを frame 単位で生成:
  va[i, j] = 対象話者ch が (t_i, t_i + horizon_j] に発話しているか (0/1)
turn-end 教師ラベルは区間単位 tag (0=継続/no, 1=区間末/yes, 2=相槌/other)。

保存: out_dir/shard_XXXXX.npz  (feats f16, va u8, lengths i32, labels i8, uttids <U40)
      out_dir/manifest.txt     (シャード一覧)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from espnet2.tasks.asr import CIFASRTask

FS = 16000


def load_scp(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(None, 1)
                d[k] = v
    return d


def load_segments(path):
    seg = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                seg[p[0]] = (p[1], float(p[2]), float(p[3]))
    return seg


def build_channel_timeline(seg):
    """rec(=channel) -> (starts[], prefmax_end[]) for O(log n) future-activity query."""
    by_rec = {}
    for u, (r, s, e) in seg.items():
        by_rec.setdefault(r, []).append((s, e))
    tl = {}
    for r, lst in by_rec.items():
        lst.sort()
        starts = np.array([x[0] for x in lst])
        ends = np.array([x[1] for x in lst])
        tl[r] = (starts, np.maximum.accumulate(ends))
    return tl


def active_in(tl_rec, a, b):
    """rec が (a, b] に発話しているか。starts<b の中で max(end)>a なら True。"""
    starts, prefmax = tl_rec
    i = np.searchsorted(starts, b, side="right")
    if i == 0:
        return False
    return bool(prefmax[i - 1] > a)


@torch.no_grad()
def encode(model, speech, device):
    sp = torch.from_numpy(speech).unsqueeze(0).to(device)
    sl = torch.tensor([sp.shape[1]], device=device)
    enc = model.encode(sp, sl)
    eo, eol = enc[0], enc[1]
    if isinstance(eo, tuple):
        eo = eo[0]
    L = min(int(eol[0].item()), eo.shape[1])
    if L == 0:
        return None
    return eo[0, :L, :].cpu().numpy()      # (T, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cap", type=float, default=1.0, help="区間末より先に読む秒数")
    ap.add_argument("--horizons", default="0.2,0.4,0.6,1.0", help="未来VAビン(秒)")
    ap.add_argument("--shard_size", type=int, default=4000)
    ap.add_argument("--max_utts", type=int, default=0, help="0=全件(スモークテスト用に制限)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    horizons = [float(x) for x in args.horizons.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"モデルロード: {args.model}")
    model, _ = CIFASRTask.build_model_from_file(args.config, args.model, args.device)
    model.to(args.device).eval()

    wav = load_scp(args.wav_scp)
    seg = load_segments(args.segments)
    tag = load_scp(args.tag)
    tl = build_channel_timeline(seg)

    utts = [u for u in seg if u in tag and seg[u][0] in wav]
    utts.sort()
    if args.max_utts > 0:
        utts = utts[:args.max_utts]
    print(f"対象区間: {len(utts)}  (cap={args.cap}s, horizons={horizons})")

    info = {}

    def nframes(p):
        if p not in info:
            info[p] = sf.info(p).frames
        return info[p]

    feats, vas, lens, labs, uids = [], [], [], [], []
    shard_id, fail, written = 0, 0, 0
    manifest = open(out / "manifest.txt", "w")

    def flush():
        nonlocal shard_id, feats, vas, lens, labs, uids
        if not lens:
            return
        path = out / f"shard_{shard_id:05d}.npz"
        np.savez(path,
                 feats=np.concatenate(feats).astype(np.float16),
                 va=np.concatenate(vas).astype(np.uint8),
                 lengths=np.array(lens, np.int32),
                 labels=np.array(labs, np.int8),
                 uttids=np.array(uids, dtype="<U40"))
        manifest.write(path.name + "\n")
        manifest.flush()
        shard_id += 1
        feats, vas, lens, labs, uids = [], [], [], [], []

    for n, u in enumerate(utts):
        rec, s, e = seg[u]
        path = wav[rec]
        try:
            total = nframes(path)
            st = max(0, int(round(s * FS)))
            en = min(total, int(round((e + args.cap) * FS)))
            if en - st < FS // 10:
                fail += 1
                continue
            sp, sr = sf.read(path, start=st, stop=en, dtype="float32")
            if sr != FS:
                fail += 1
                continue
            if sp.ndim > 1:
                sp = sp[:, 0]
        except Exception:
            fail += 1
            continue
        eo = encode(model, sp, args.device)
        if eo is None:
            fail += 1
            continue
        T = eo.shape[0]
        used = (en - st) / FS
        fd = used / T                                  # frame 長(秒)
        t_abs = s + (np.arange(T) + 0.5) * fd           # 各 frame の絶対時刻
        va = np.zeros((T, len(horizons)), np.uint8)
        for i in range(T):
            a = t_abs[i]
            for j, h in enumerate(horizons):
                va[i, j] = active_in(tl[rec], a, a + h)
        feats.append(eo)
        vas.append(va)
        lens.append(T)
        labs.append(int(tag[u].strip()))
        uids.append(u)
        written += 1
        if len(lens) >= args.shard_size:
            flush()
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(utts)}  written={written} fail={fail}")

    flush()
    manifest.close()
    print(f"完了: written={written} fail={fail} shards={shard_id} -> {out}")


if __name__ == "__main__":
    main()
