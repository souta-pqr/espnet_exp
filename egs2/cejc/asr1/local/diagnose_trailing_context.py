#!/usr/bin/env python
"""決定的診断: 「0.62 上限」は streaming の限界か、それともセグメント化のアーティファクトか。

仮説: 区間末判定の決定打（発話直後の無音・次話者ターン）は、タイトに切られた
セグメントでは存在しない。だが encoder には look_ahead=9 フレーム(≈300ms)の
合法的な未来文脈があり、元録音にはセグメント間ギャップとして無音が残っている。
→ 各区間を [start, end+Δ] で読み直し、区間末フレームの encoder 出力で no/yes を
   線形プローブ。look_ahead が本物の trailing 音声で埋まると 0.62 を超えるかを測る。

Δ は look_ahead 予算(≈300ms)を中心に振る。超えれば:
  ・情報不足の原因は「未来禁止」ではなく「セグメント化」
  ・修正は再セグメント化(=データ)で、アーキ変更も追加遅延も不要(既存 look_ahead 内)
  ・SPRT/ECONOMY 共通の slave(tag_classifier)事後確率が底上げされる
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from espnet2.tasks.asr import CIFASRTask
from espnet2.asr.espnet_model_cif_tag import _cif_function_impl

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


@torch.no_grad()
def extract(model, speech, device, orig_frac, k=10):
    """speech=[start, end+Δ] の生波形。区間末フレーム(orig_frac)とその周辺の表現を返す。"""
    speech = torch.from_numpy(speech).unsqueeze(0).to(device)
    slen = torch.tensor([speech.shape[1]], device=device)
    enc = model.encode(speech, slen)
    eo, eol = enc[0], enc[1]
    if isinstance(eo, tuple):
        eo = eo[0]
    L = min(int(eol[0].item()), eo.shape[1])
    if L == 0:
        return None
    eo = eo[0, :L, :]                                   # (L, D)
    # 元区間末に対応するフレーム（look_ahead で trailing 側を取り込む決定点）
    end_idx = min(L - 1, max(0, int(round(orig_frac * (L - 1)))))
    kk = min(k, L)

    B, T, D = enc[0].shape if not isinstance(enc[0], tuple) else (1, L, eo.shape[-1])
    alpha = model.alpha_predictor(eo.unsqueeze(0)).squeeze(-1)
    cif = _cif_function_impl(inputs=eo.unsqueeze(0), alpha=alpha,
                             target_lengths=None, beta=model.cif_threshold)
    co = cif["cif_out"][0]
    nf = int(cif["alpha_sum"][0].round().long().clamp(min=1)[0].item())
    cif_last = (co[0, min(nf, co.shape[1]) - 1].cpu().numpy()
                if co.shape[1] > 0 else np.zeros(eo.shape[-1], np.float32))
    return {
        "enc_at_end": eo[end_idx].cpu().numpy(),        # 決定点フレーム
        "enc_lastk":  eo[-kk:].mean(0).cpu().numpy(),    # 末尾 k 平均（trailing 領域）
        "cif_last":   cif_last,
    }


def probe(X, y, name, seed=0):
    rng = np.random.RandomState(seed)
    g = np.arange(len(y)); rng.shuffle(g)
    te = np.zeros(len(y), bool); te[g[:int(len(y) * 0.3)]] = True
    tr = ~te
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te])); yt = y[te]
    bacc = np.mean([(pred[yt == c] == c).mean() for c in (0, 1)])
    return bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_per_class", type=int, default=2380)
    ap.add_argument("--deltas", default="0.0,0.15,0.3,0.5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    print(f"モデルロード: {args.model}")
    model, _ = CIFASRTask.build_model_from_file(args.config, args.model, args.device)
    model.to(args.device).eval()

    wav = load_scp(args.wav_scp)
    seg = load_segments(args.segments)
    tag = load_scp(args.tag)

    rng = np.random.RandomState(args.seed)
    by = {0: [], 1: []}
    for u in sorted(set(seg) & set(tag) & {k for k in tag}):
        if u not in seg:
            continue
        c = int(tag[u].strip())
        if c in by and seg[u][0] in wav:
            by[c].append(u)
    sel = []
    for c in (0, 1):
        lst = by[c][:]; rng.shuffle(lst)
        sel += [(u, c) for u in lst[:args.max_per_class]]
    rng.shuffle(sel)
    print(f"対象: no={min(len(by[0]),args.max_per_class)}, "
          f"yes={min(len(by[1]),args.max_per_class)} (計 {len(sel)})")

    info_cache = {}

    def nframes(path):
        if path not in info_cache:
            info_cache[path] = sf.info(path).frames
        return info_cache[path]

    keys = ["enc_at_end", "enc_lastk", "cif_last"]
    print("=" * 78)
    print(f"線形プローブ <no> vs <yes>（balanced, chance=0.5, n={len(sel)}）")
    print("look_ahead=9フレーム≈300ms。Δ=trailing 追加秒数。")
    print("=" * 78)
    print(f"{'Δ(ms)':>7} | " + " | ".join(f"{k:>11}" for k in keys))
    print("-" * 78)

    curve = {k: [] for k in keys}
    for d in deltas:
        feats = {k: [] for k in keys}
        ys = []
        fail = 0
        for u, c in sel:
            rec, s, e = seg[u]
            path = wav[rec]
            total = nframes(path)
            st = max(0, int(round(s * FS)))
            en = min(total, int(round((e + d) * FS)))
            if en - st < FS // 10:
                fail += 1; continue
            try:
                sp, _ = sf.read(path, start=st, stop=en, dtype="float32")
            except Exception:
                fail += 1; continue
            if sp.ndim > 1:
                sp = sp[:, 0]
            used_dur = (en - st) / FS
            orig_dur = e - s
            orig_frac = orig_dur / used_dur if used_dur > 0 else 1.0
            r = extract(model, sp, args.device, orig_frac)
            if r is None:
                fail += 1; continue
            for k in keys:
                feats[k].append(r[k])
            ys.append(c)
        y = np.array(ys)
        row = []
        for k in keys:
            b = probe(np.stack(feats[k]), y, k, args.seed)
            curve[k].append(b); row.append(f"{b:11.3f}")
        print(f"{d*1000:7.0f} | " + " | ".join(row) + f"   (n={len(y)}, fail={fail})")

    print("=" * 78)
    base = curve["enc_at_end"][0]
    best = max(curve["enc_at_end"])
    print(f"判定(enc_at_end): Δ=0 → {base:.3f},  最良 → {best:.3f}  (Δ={best-base:+.3f})")
    if best - base >= 0.03:
        print("  → trailing 文脈で 0.62 を突破。情報不足の原因は『未来禁止』ではなく")
        print("     『セグメント化で trailing 無音を捨てている』こと。")
        print("     修正: 再セグメント化(end+~300ms) = データ修正。既存 look_ahead 内で追加遅延なし。")
    else:
        print("  → trailing 文脈でも改善せず。0.62 はより本質的な限界。")
    print("=" * 78)


if __name__ == "__main__":
    main()
