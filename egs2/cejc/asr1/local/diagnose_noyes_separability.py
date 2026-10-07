#!/usr/bin/env python
"""C 診断: <no> vs <yes> が cif_out から分離可能かを線形プローブで確認する。

目的:
  causal GRU（方針 A）に進む前に、「文脈を足せば <no>/<yes> が分離できるか」を
  軽量に検証する。学習済み Stage1 モデルから cif_out を抽出し、以下の表現で
  <no> vs <yes> の線形分類（logistic regression）精度を比較する。

  (1) per-fire single   : 各 fire の cif_out[i]（現状の tag_classifier 入力）
  (2) per-fire causal-mean: mean(cif_out[0..i])（因果的に過去を集約した最小構成）
  (3) utt last-fire     : 最終 fire の cif_out（発話代表＝現状の tag_acc 計測点）
  (4) utt mean          : 全 fire 平均（非因果の発話代表）

  (2) >> (1) なら「過去文脈を足せば分離可能」＝ causal GRU を入れる A に価値あり。
  (1) ≈ (2) で両方低いなら cif_out 自体に no/yes 情報が無い → 方針転換。

モデル・データには一切書き込まない（読み取りのみ）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # espnet root
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
def extract_cif_fires(model, speech, device):
    """1 発話の cif_out fire 列 (n_fires, D) を返す。失敗時 None。"""
    speech = torch.from_numpy(speech).unsqueeze(0).to(device)
    speech_lengths = torch.tensor([speech.shape[1]], device=device)
    enc = model.encode(speech, speech_lengths)
    encoder_out, encoder_out_lens = enc[0], enc[1]
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    B, T, D = encoder_out.shape
    alpha = model.alpha_predictor(encoder_out).squeeze(-1)
    pad_mask = (
        torch.arange(T, device=device).unsqueeze(0) < encoder_out_lens.unsqueeze(1)
    )
    alpha = alpha * pad_mask.float()
    cif_result = _cif_function_impl(
        inputs=encoder_out, alpha=alpha, target_lengths=None, beta=model.cif_threshold
    )
    cif_out = cif_result["cif_out"][0]          # (1, T_cif, D)
    alpha_sum = cif_result["alpha_sum"][0]
    n_fires = int(alpha_sum.round().long().clamp(min=1)[0].item())
    if cif_out.shape[1] == 0 or n_fires == 0:
        return None
    n_fires = min(n_fires, cif_out.shape[1])
    return cif_out[0, :n_fires, :].float().cpu().numpy()   # (n_fires, D)


def probe(X, y, groups, name, seed=0):
    """発話単位 train/test split で logistic regression を学習し精度を表示。"""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = int(len(uniq) * 0.3)
    test_g = set(uniq[:n_test].tolist())
    te = np.array([g in test_g for g in groups])
    tr = ~te

    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X[tr]), y[tr])
    pred = clf.predict(scaler.transform(X[te]))

    yt = y[te]
    acc = (pred == yt).mean()
    # macro F1 (2 class)
    f1s = []
    for c in (0, 1):
        tp = ((pred == c) & (yt == c)).sum()
        fp = ((pred == c) & (yt != c)).sum()
        fn = ((pred != c) & (yt == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    # balanced accuracy (各クラス recall の平均)
    bacc = np.mean([(pred[yt == c] == c).mean() for c in (0, 1)])
    print(f"  {name:24s}: acc={acc:.3f}  bal_acc={bacc:.3f}  "
          f"macroF1={np.mean(f1s):.3f}  (no_f1={f1s[0]:.3f}, yes_f1={f1s[1]:.3f}, "
          f"n_test={te.sum()})")
    return bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_per_class", type=int, default=2380,
                    help="クラスあたり最大発話数（balance 用）")
    ap.add_argument("--max_fires", type=int, default=40,
                    help="1 発話あたり probe に使う最大 fire 数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"モデルロード: {args.model}")
    model, _ = CIFASRTask.build_model_from_file(args.config, args.model, args.device)
    model.to(args.device).eval()

    wav_scp = load_scp(args.wav_scp, base_dir=args.base_dir)
    tag_scp = load_scp(args.tag)
    uttids = sorted(set(wav_scp) & set(tag_scp))

    # <no>(0), <yes>(1) のみ、クラス均衡にサンプリング
    rng = np.random.RandomState(args.seed)
    by_cls = {0: [], 1: []}
    for u in uttids:
        c = int(tag_scp[u].strip())
        if c in by_cls:
            by_cls[c].append(u)
    sel = []
    for c in (0, 1):
        lst = by_cls[c]
        rng.shuffle(lst)
        sel += [(u, c) for u in lst[:args.max_per_class]]
    rng.shuffle(sel)
    print(f"対象発話: no={min(len(by_cls[0]),args.max_per_class)}, "
          f"yes={min(len(by_cls[1]),args.max_per_class)} (計 {len(sel)})")

    # 特徴抽出
    Xs_single, Xs_cmean, ys_fire, grp_fire = [], [], [], []
    Xs_last, Xs_umean, ys_utt = [], [], []
    fail = 0
    for n, (u, c) in enumerate(sel):
        speech, _ = sf.read(wav_scp[u], dtype="float32")
        fires = extract_cif_fires(model, speech, args.device)
        if fires is None:
            fail += 1
            continue
        cmean = np.cumsum(fires, axis=0) / np.arange(1, len(fires) + 1)[:, None]
        k = min(len(fires), args.max_fires)
        idx = np.linspace(0, len(fires) - 1, k).round().astype(int)
        Xs_single.append(fires[idx])
        Xs_cmean.append(cmean[idx])
        ys_fire.append(np.full(k, c))
        grp_fire.append(np.full(k, n))
        Xs_last.append(fires[-1])
        Xs_umean.append(fires.mean(axis=0))
        ys_utt.append(c)
        if (n + 1) % 1000 == 0:
            print(f"  抽出 {n + 1}/{len(sel)} 件")

    print(f"抽出失敗: {fail} 件\n")

    X_single = np.concatenate(Xs_single)
    X_cmean = np.concatenate(Xs_cmean)
    y_fire = np.concatenate(ys_fire)
    g_fire = np.concatenate(grp_fire)
    X_last = np.stack(Xs_last)
    X_umean = np.stack(Xs_umean)
    y_utt = np.array(ys_utt)
    g_utt = np.arange(len(y_utt))

    print("=" * 70)
    print("線形プローブ: <no> vs <yes> 分離精度（発話単位 split, test 30%）")
    print("  bal_acc=0.5 はチャンスレベル。高いほど分離可能。")
    print("=" * 70)
    print(f"[per-fire レベル] サンプル数={len(y_fire)}")
    b1 = probe(X_single, y_fire, g_fire, "(1) per-fire single", args.seed)
    b2 = probe(X_cmean, y_fire, g_fire, "(2) per-fire causal-mean", args.seed)
    print(f"[発話レベル] サンプル数={len(y_utt)}")
    b3 = probe(X_last, y_utt, g_utt, "(3) utt last-fire", args.seed)
    b4 = probe(X_umean, y_utt, g_utt, "(4) utt mean(all)", args.seed)

    print("=" * 70)
    print("判定:")
    print(f"  per-fire:  single={b1:.3f} → causal-mean={b2:.3f}  (Δ={b2-b1:+.3f})")
    print(f"  utt    :  last={b3:.3f} → mean={b4:.3f}  (Δ={b4-b3:+.3f})")
    if b2 - b1 >= 0.03 or b4 - b3 >= 0.03:
        print("  → 文脈集約で改善。causal GRU（方針A）に進む価値あり。")
    else:
        print("  → 文脈集約しても改善せず。cif_out 自体に no/yes 情報が乏しい可能性。")
        print("     上流（encoder出力 / テキスト文脈）の活用やラベル見直しを検討。")
    print("=" * 70)


if __name__ == "__main__":
    main()
