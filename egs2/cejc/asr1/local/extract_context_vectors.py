#!/usr/bin/env python
"""過去文脈ベクトル ctx_vec を抽出して ark+scp 保存する（3 方式を切替）。

共通: 学習済み encoder で各発話を符号化→有効フレーム平均 → uvec(256)。
ctx_vec(区間 i) の作り方を --context_scope で切替（モデル・配線は不変）:
  same    : 同一 reco（=同一話者）の直前 N 発話の uvec 平均
  session : 同一セッション（reco から _IC* を除く＝両話者）の直前 N 発話の uvec 平均（相手の発話も含む）
  cluster : 全 uvec を KMeans(K) でクラスタリングし、各 utt を割当クラスタの重心ベクトルに置換
            （重心は train で fit し、--centroids_file で dev/eval と共有する）
直前が無い/該当無しはゼロベクトル。出力 <dump_dir>/<out_name>.{ark,scp}。
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import kaldiio


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1].strip()
    return m


@torch.no_grad()
def utt_vector(model, wav, device, feat_source="encoder", feat_norm="global"):
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    xl = torch.tensor([x.shape[1]], device=device)
    if feat_source == "raw":
        # エンコーダを通さない生の音響特徴量（フロントエンドの log-mel fbank）を時間平均。
        feats, fl = model._extract_feats(x, xl)         # (1, T, n_mels)
        if feat_norm == "global" and model.normalize is not None:
            feats, fl = model.normalize(feats, fl)      # encoder が実際に受け取る入力（global_mvn）
        n = int(fl[0])
        return feats[0, :n].mean(0).cpu().numpy().astype(np.float32)   # (n_mels,)
    enc = model.encode(x, xl)
    eo = enc[0]
    if isinstance(eo, tuple):
        eo = eo[0]
    n = int(enc[1][0])
    return eo[0, :n].mean(0).cpu().numpy().astype(np.float32)   # (256,)


def session_of(reco):
    return re.sub(r"_IC\d+$", "", reco)   # C001_006_IC04 -> C001_006


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump_dir", required=True, help="dump/raw/<set>（wav.scp・出力先）")
    ap.add_argument("--seg_dir", required=True, help="data/<set>（segments で順序/話者）")
    ap.add_argument("--context_scope", choices=["same", "session", "cluster"], default="same")
    ap.add_argument("--n_past", type=int, default=1, help="same/session: 直前発話数")
    ap.add_argument("--max_past_sec", type=float, default=10.0)
    ap.add_argument("--top_k", type=int, default=5,
                    help="cluster: 過去の類似発話を上位何件集めて平均するか")
    ap.add_argument("--feat_source", choices=["encoder", "raw"], default="encoder",
                    help="encoder=encoder出力の平均(256次元) / raw=エンコーダ非経由の生fbank平均(n_mels次元)")
    ap.add_argument("--feat_norm", choices=["global", "none"], default="global",
                    help="raw時のみ: global=global_mvn適用(encoder入力そのもの) / none=生log-mel")
    ap.add_argument("--out_name", default="ctx_vec", help="出力ファイル名（<dump>/<out_name>.ark/scp）")
    ap.add_argument("--max_utts", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from espnet2.tasks.asr import TurnTakingASRTask
    model, _ = TurnTakingASRTask.build_model_from_file(args.config, args.model, args.device)
    model.eval()

    wav_scp = read_map(Path(args.dump_dir) / "wav.scp")
    seg = {}
    with open(Path(args.seg_dir) / "segments", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                seg[p[0]] = (p[1], float(p[2]))   # reco, start

    utts = [u for u in wav_scp if u in seg]
    if args.max_utts > 0:
        utts = utts[: args.max_utts]

    # 1) 各 utt の encoder 平均ベクトル uvec
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
        uvec[u] = utt_vector(model, wav, args.device, args.feat_source, args.feat_norm)
        if (k + 1) % 2000 == 0:
            print(f"  ...{k+1} 符号化", flush=True)
    dim = len(next(iter(uvec.values())))
    print(f"符号化 {len(uvec)} utt (dim={dim}, scope={args.context_scope})")

    # 2) ctx_vec を作る
    ctx = {}
    if args.context_scope in ("same", "session"):
        # グルーピングキー：same=reco / session=セッション(両話者)
        groups = defaultdict(list)
        for u in utts:
            reco, _ = seg[u]
            key = reco if args.context_scope == "same" else session_of(reco)
            groups[key].append(u)
        for key in groups:
            groups[key].sort(key=lambda x: seg[x][1])   # start でソート
        idx = {u: i for key in groups for i, u in enumerate(groups[key])}
        for u in utts:
            reco, s = seg[u]
            key = reco if args.context_scope == "same" else session_of(reco)
            sib = groups[key]
            i = idx[u]
            vs = []
            for j in range(i - 1, max(-1, i - 1 - args.n_past), -1):
                pu = sib[j]
                if seg[pu][1] >= s - args.max_past_sec and pu in uvec:
                    vs.append(uvec[pu])
                else:
                    break
            ctx[u] = (np.mean(vs, axis=0).astype(np.float32) if vs
                      else np.zeros(dim, dtype=np.float32))
    else:  # cluster = 過去の類似発話 top_k を集めて平均（kNN・過去発話のみ・同一セッション）
        # 同一セッション（両話者）の中で、現発話より前の発話から
        # uvec のコサイン類似度上位 top_k を取り、その平均を ctx_vec とする。
        groups = defaultdict(list)
        for u in utts:
            groups[session_of(seg[u][0])].append(u)
        for key in groups:
            groups[key].sort(key=lambda x: seg[x][1])   # start 時刻順（両話者が交互に並ぶ）
        for key, sib in groups.items():
            rows = [u for u in sib if u in uvec]         # 時間順・uvecありのみ
            if not rows:
                for u in sib:
                    ctx[u] = np.zeros(dim, dtype=np.float32)
                continue
            U = np.stack([uvec[u] for u in rows])                       # (M, dim)
            Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)  # 正規化（cos用）
            for r, u in enumerate(rows):
                if r == 0:
                    ctx[u] = np.zeros(dim, dtype=np.float32)            # 過去なし
                    continue
                sims = Un[:r] @ Un[r]                                   # 過去 r 件との類似度
                k = min(args.top_k, r)
                top = np.argpartition(-sims, k - 1)[:k]                 # 類似上位 k（過去のみ）
                ctx[u] = U[top].mean(0).astype(np.float32)
            for u in sib:
                ctx.setdefault(u, np.zeros(dim, dtype=np.float32))
        for u in utts:
            ctx.setdefault(u, np.zeros(dim, dtype=np.float32))

    # 3) 出力
    out = Path(args.dump_dir) / args.out_name
    with kaldiio.WriteHelper(f"ark,scp:{out}.ark,{out}.scp") as w:
        for u in utts:
            w(u, ctx[u])
    nz = sum(1 for u in utts if np.any(ctx[u]))
    print(f"出力: {out}.scp  (非ゼロ {nz}/{len(utts)}, scope={args.context_scope})")


if __name__ == "__main__":
    main()
