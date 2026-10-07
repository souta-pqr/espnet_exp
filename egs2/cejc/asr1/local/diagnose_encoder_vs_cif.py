#!/usr/bin/env python
"""C-2 診断: no/yes(エンドポイント) 情報は cif_out と encoder_out のどちらにあるか。

ラベル: <no>=区間末でない, <yes>=区間末である, <other>=相槌。
→ エンドポイント検出は韻律（語末の伸び・ピッチ下降・直後の無音）に依存。
   CIF はトークン境界で発火するため語末の減衰・無音を取りこぼす疑い。
   encoder_out（フレーム単位）にはその情報が残っている可能性を検証する。

各発話表現で <no> vs <yes> を線形プローブ（balanced, 発話単位 split）:
  cif_last        : 最終 fire の cif_out（現状の実質入力）
  cif_mean        : 全 fire 平均
  enc_mean        : encoder_out 全フレーム平均
  enc_last        : encoder_out 最終フレーム
  enc_lastk       : encoder_out 末尾 k フレーム平均（エンドポイント領域）
  enc_max         : encoder_out 時間方向 max-pool
  cif_last+enc_lastk : 両者を結合
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from espnet2.tasks.asr import CIFASRTask
from espnet2.asr.espnet_model_cif_tag import _cif_function_impl


def load_scp(path, base_dir=None):
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(None, 1)
            if base_dir and not Path(val).is_absolute():
                val = str(Path(base_dir) / val)
            data[key] = val
    return data


@torch.no_grad()
def extract(model, speech, device, k=10):
    """1 発話の各種表現を dict で返す。失敗時 None。"""
    speech = torch.from_numpy(speech).unsqueeze(0).to(device)
    speech_lengths = torch.tensor([speech.shape[1]], device=device)
    enc = model.encode(speech, speech_lengths)
    encoder_out, encoder_out_lens = enc[0], enc[1]
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    L = int(encoder_out_lens[0].item())
    L = min(L, encoder_out.shape[1])
    if L == 0:
        return None
    eo = encoder_out[0, :L, :]                       # (L, D)

    B, T, D = encoder_out.shape
    alpha = model.alpha_predictor(encoder_out).squeeze(-1)
    pad_mask = (torch.arange(T, device=device).unsqueeze(0)
                < encoder_out_lens.unsqueeze(1))
    alpha = alpha * pad_mask.float()
    cif_result = _cif_function_impl(inputs=encoder_out, alpha=alpha,
                                    target_lengths=None, beta=model.cif_threshold)
    cif_out = cif_result["cif_out"][0]
    n_fires = int(cif_result["alpha_sum"][0].round().long().clamp(min=1)[0].item())
    if cif_out.shape[1] == 0:
        return None
    n_fires = min(n_fires, cif_out.shape[1])
    co = cif_out[0, :n_fires, :]                      # (n_fires, D)

    kk = min(k, L)
    return {
        "cif_last":  co[-1].cpu().numpy(),
        "cif_mean":  co.mean(0).cpu().numpy(),
        "enc_mean":  eo.mean(0).cpu().numpy(),
        "enc_last":  eo[-1].cpu().numpy(),
        "enc_lastk": eo[-kk:].mean(0).cpu().numpy(),
        "enc_max":   eo.max(0).values.cpu().numpy(),
    }


def probe(X, y, groups, name, seed=0, mlp=False):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    test_g = set(uniq[:int(len(uniq) * 0.3)].tolist())
    te = np.array([g in test_g for g in groups])
    tr = ~te
    scaler = StandardScaler().fit(X[tr])
    if mlp:
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                            early_stopping=True, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X[tr]), y[tr])
    pred = clf.predict(scaler.transform(X[te]))
    yt = y[te]
    bacc = np.mean([(pred[yt == c] == c).mean() for c in (0, 1)])
    f1s = []
    for c in (0, 1):
        tp = ((pred == c) & (yt == c)).sum(); fp = ((pred == c) & (yt != c)).sum()
        fn = ((pred != c) & (yt == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    tag = " [MLP]" if mlp else ""
    print(f"  {name:22s}{tag}: bal_acc={bacc:.3f}  macroF1={np.mean(f1s):.3f}  "
          f"(no_f1={f1s[0]:.3f}, yes_f1={f1s[1]:.3f})")
    return bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_per_class", type=int, default=2380)
    ap.add_argument("--lastk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"モデルロード: {args.model}")
    model, _ = CIFASRTask.build_model_from_file(args.config, args.model, args.device)
    model.to(args.device).eval()

    wav_scp = load_scp(args.wav_scp, base_dir=args.base_dir)
    tag_scp = load_scp(args.tag)
    uttids = sorted(set(wav_scp) & set(tag_scp))
    rng = np.random.RandomState(args.seed)
    by_cls = {0: [], 1: []}
    for u in uttids:
        c = int(tag_scp[u].strip())
        if c in by_cls:
            by_cls[c].append(u)
    sel = []
    for c in (0, 1):
        lst = by_cls[c]; rng.shuffle(lst)
        sel += [(u, c) for u in lst[:args.max_per_class]]
    rng.shuffle(sel)
    print(f"対象発話: no={min(len(by_cls[0]),args.max_per_class)}, "
          f"yes={min(len(by_cls[1]),args.max_per_class)} (計 {len(sel)})")

    keys = ["cif_last", "cif_mean", "enc_mean", "enc_last", "enc_lastk", "enc_max"]
    feats = {k: [] for k in keys}
    ys = []
    fail = 0
    for n, (u, c) in enumerate(sel):
        speech, _ = sf.read(wav_scp[u], dtype="float32")
        d = extract(model, speech, args.device, k=args.lastk)
        if d is None:
            fail += 1
            continue
        for k in keys:
            feats[k].append(d[k])
        ys.append(c)
        if (n + 1) % 1000 == 0:
            print(f"  抽出 {n + 1}/{len(sel)} 件")
    print(f"抽出失敗: {fail} 件\n")

    feats = {k: np.stack(v) for k, v in feats.items()}
    y = np.array(ys)
    g = np.arange(len(y))

    print("=" * 72)
    print(f"線形プローブ: <no> vs <yes>（balanced, 発話単位 split, chance=0.5, n={len(y)}）")
    print("=" * 72)
    res = {}
    for k in keys:
        res[k] = probe(feats[k], y, g, k, args.seed)

    combo = np.concatenate([feats["cif_last"], feats["enc_lastk"]], axis=1)
    res["cif_last+enc_lastk"] = probe(combo, y, g, "cif_last+enc_lastk", args.seed)

    # 最良の encoder 表現で nonlinear 天井も確認
    best_enc = max(["enc_mean", "enc_last", "enc_lastk", "enc_max"], key=lambda k: res[k])
    print("-" * 72)
    probe(feats[best_enc], y, g, best_enc, args.seed, mlp=True)
    probe(combo, y, g, "cif_last+enc_lastk", args.seed, mlp=True)

    print("=" * 72)
    print("判定:")
    best_enc_acc = max(res["enc_mean"], res["enc_last"], res["enc_lastk"], res["enc_max"])
    print(f"  cif 系 最良 = {max(res['cif_last'], res['cif_mean']):.3f}")
    print(f"  enc 系 最良 = {best_enc_acc:.3f}  (best={best_enc})")
    print(f"  結合       = {res['cif_last+enc_lastk']:.3f}")
    if best_enc_acc - max(res["cif_last"], res["cif_mean"]) >= 0.03:
        print("  → encoder_out が cif_out より no/yes を分離できる。")
        print("     tag_classifier に encoder 特徴（特に発話末領域）を合流させる設計が有効。")
    else:
        print("  → encoder_out でも分離が伸びない。発話内音響だけでは不足。")
        print("     発話をまたぐ文脈や明示的な韻律特徴（pitch/pause）の導入を検討。")
    print("=" * 72)


if __name__ == "__main__":
    main()
