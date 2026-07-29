#!/usr/bin/env python
"""Stage2(turntaking_xfmr) 用: 過去N発話の **max-pool ベクトル列** past_vec を事前計算する。

凍結する ASR エンコーダで各発話を符号化し、**発話ごとに max pooling** → uvec_max (D次元)。
現発話ごとに、直前N発話（同一話者 or セッション）の uvec_max を **時系列順に積んだ行列 (n, D)** を
past_vec として ark+scp に保存する（n は 0〜N の可変。過去が無い場合はゼロ1行）。

Transformer 入力は [past_vec の n 本 ; 現発話 encoder 出力 T フレーム] になる。
出力: <dump_dir>/<out_name>.{ark,scp}
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import kaldiio
import numpy as np
import soundfile as sf
import torch


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1].strip()
    return m


def session_of(reco):
    return re.sub(r"_IC\d+$", "", reco)


@torch.no_grad()
def utt_maxvec(model, wav, device):
    """1発話 → 凍結エンコーダ → 有効フレームで max pooling → (D,)"""
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    xl = torch.tensor([x.shape[1]], device=device)
    enc = model.encode(x, xl)
    eo = enc[0]
    if isinstance(eo, tuple):
        eo = eo[0]
    n = int(enc[1][0])
    return eo[0, :n].max(dim=0).values.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Stage1(純粋ASR)の config.yaml")
    ap.add_argument("--model", required=True, help="Stage1(純粋ASR)の model.pth（凍結エンコーダ）")
    ap.add_argument("--dump_dir", required=True, help="dump/raw/<set>")
    ap.add_argument("--seg_dir", required=True, help="data/<set>（segments）")
    ap.add_argument("--scope", choices=["same", "session"], default="same")
    ap.add_argument("--n_past", type=int, default=2)
    ap.add_argument("--max_past_sec", type=float, default=30.0)
    ap.add_argument("--out_name", default="past_vec")
    ap.add_argument("--uvec_cache", default="",
                    help="発話ごとの max-pool ベクトルのキャッシュ(.npz)。N を変えても中身は同じなので再利用する")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    wav_scp = read_map(Path(args.dump_dir) / "wav.scp")
    seg = {}
    with open(Path(args.seg_dir) / "segments", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                seg[p[0]] = (p[1], float(p[2]))          # reco, start

    utts = [u for u in wav_scp if u in seg]

    # 1) 各発話の max-pool ベクトル（N に依存しないのでキャッシュを再利用）
    cache = Path(args.uvec_cache) if args.uvec_cache else None
    if cache is not None and cache.is_file():
        z = np.load(cache)
        uvec = {k: z[k] for k in z.files}
        print(f"uvec キャッシュを再利用: {cache} ({len(uvec)} utt)")
    else:
        from espnet2.tasks.asr import TurnTakingASRTask
        model, _ = TurnTakingASRTask.build_model_from_file(
            args.config, args.model, args.device
        )
        model.eval()                                     # BN は running 統計を使う（更新しない）
        uvec = {}
        for k, u in enumerate(utts):
            path = wav_scp[u]
            if "|" in path:
                continue
            try:
                wav, _ = sf.read(path, dtype="float32", always_2d=False)
            except Exception:
                continue
            if wav.ndim > 1:
                wav = wav[:, 0]
            if len(wav) < 320:
                continue
            uvec[u] = utt_maxvec(model, wav, args.device)
            if (k + 1) % 2000 == 0:
                print(f"  ...{k+1} 符号化", flush=True)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **uvec)
            print(f"uvec キャッシュを保存: {cache}")
    dim = len(next(iter(uvec.values())))
    print(f"max-pool 済み {len(uvec)} utt (dim={dim}, scope={args.scope})")

    # 2) 現発話ごとに直前N発話の uvec を時系列順に積む → (n, dim)
    groups = defaultdict(list)
    for u in utts:
        reco = seg[u][0]
        key = reco if args.scope == "same" else session_of(reco)
        groups[key].append(u)
    for key in groups:
        groups[key].sort(key=lambda x: seg[x][1])         # start 昇順
    idx = {u: i for key in groups for i, u in enumerate(groups[key])}

    out = Path(args.dump_dir) / args.out_name
    n_nopast = 0
    with kaldiio.WriteHelper(f"ark,scp:{out}.ark,{out}.scp") as w:
        for u in utts:
            reco, s = seg[u]
            key = reco if args.scope == "same" else session_of(reco)
            sib = groups[key]
            i = idx[u]
            vs = []
            for j in range(i - 1, max(-1, i - 1 - args.n_past), -1):
                pu = sib[j]
                if seg[pu][1] >= s - args.max_past_sec and pu in uvec:
                    vs.append(uvec[pu])
                else:
                    break
            vs.reverse()                                   # 古→新（時系列順）
            if vs:
                mat = np.stack(vs).astype(np.float32)      # (n, dim)
            else:
                mat = np.zeros((1, dim), dtype=np.float32)  # 過去なし＝ゼロ1本
                n_nopast += 1
            w(u, mat)
    print(f"出力: {out}.scp  (過去なし={n_nopast}/{len(utts)}, N={args.n_past}, scope={args.scope})")


if __name__ == "__main__":
    main()
