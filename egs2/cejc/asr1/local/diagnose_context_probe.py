#!/usr/bin/env python
"""C-3 診断: 過去文脈で no/yes(エンドポイント) 分離が 0.62 を超えるか。

streaming 制約: 未来禁止・過去のみ可。同一会話セッション内の時系列を再構成し、
対象区間の直前にあった発話（自話者 / 相手 / 誰でも）の表現を結合して線形プローブ。

ID 構造: T005_001_IC01_0096549_0097609
  conv=T005_001（会話）, channel=IC01（話者）, start=0096549, end=0097609（共通クロック）

文脈表現: 直前区間の cif_mean（内容サマリ）。存在しなければ 0 ベクトル。
gap 特徴: 直前区間の終了から現在開始までの間隔（因果的に観測可能）。

比較:
  cur                : 現在区間のみ（baseline ≈0.62）
  cur + own_prev     : + 自話者の直前
  cur + intloc_prev  : + 相手の直前
  cur + any_prev     : + 直前（誰でも）
  cur + own + intloc + gaps : 全部
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


def parse_id(uttid):
    parts = uttid.split("_")
    conv = "_".join(parts[:2])
    channel = parts[2]
    start, end = int(parts[-2]), int(parts[-1])
    return conv, channel, start, end


@torch.no_grad()
def extract(model, speech, device):
    speech = torch.from_numpy(speech).unsqueeze(0).to(device)
    speech_lengths = torch.tensor([speech.shape[1]], device=device)
    enc = model.encode(speech, speech_lengths)
    encoder_out, encoder_out_lens = enc[0], enc[1]
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]
    B, T, D = encoder_out.shape
    alpha = model.alpha_predictor(encoder_out).squeeze(-1)
    pad_mask = (torch.arange(T, device=device).unsqueeze(0)
                < encoder_out_lens.unsqueeze(1))
    alpha = alpha * pad_mask.float()
    cif = _cif_function_impl(inputs=encoder_out, alpha=alpha,
                             target_lengths=None, beta=model.cif_threshold)
    co = cif["cif_out"][0]
    nf = int(cif["alpha_sum"][0].round().long().clamp(min=1)[0].item())
    if co.shape[1] == 0:
        return None
    nf = min(nf, co.shape[1])
    co = co[0, :nf, :]
    return {"cif_last": co[-1].cpu().numpy(), "cif_mean": co.mean(0).cpu().numpy()}


def probe(X, y, g, name, seed=0):
    rng = np.random.RandomState(seed)
    uniq = np.unique(g); rng.shuffle(uniq)
    test_g = set(uniq[:int(len(uniq) * 0.3)].tolist())
    te = np.array([gg in test_g for gg in g]); tr = ~te
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te])); yt = y[te]
    bacc = np.mean([(pred[yt == c] == c).mean() for c in (0, 1)])
    f1s = []
    for c in (0, 1):
        tp = ((pred == c) & (yt == c)).sum(); fp = ((pred == c) & (yt != c)).sum()
        fn = ((pred != c) & (yt == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    print(f"  {name:28s}: bal_acc={bacc:.3f}  macroF1={np.mean(f1s):.3f}  "
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"モデルロード: {args.model}")
    model, _ = CIFASRTask.build_model_from_file(args.config, args.model, args.device)
    model.to(args.device).eval()

    wav = load_scp(args.wav_scp, base_dir=args.base_dir)
    tag = load_scp(args.tag)
    uttids = sorted(set(wav) & set(tag))

    # 会話タイムライン構築
    conv = {}
    for u in uttids:
        c, ch, s, e = parse_id(u)
        conv.setdefault(c, []).append((s, e, ch, u))
    for c in conv:
        conv[c].sort()

    def find_prevs(u):
        c, ch, s, e = parse_id(u)
        own = intloc = anyp = None
        for (ss, ee, cch, uu) in conv[c]:
            if ee > s:
                break
            anyp = (uu, s - ee)
            if cch == ch:
                own = (uu, s - ee)
            else:
                intloc = (uu, s - ee)
        return own, intloc, anyp

    # 対象 no/yes をバランスサンプリング
    rng = np.random.RandomState(args.seed)
    by = {0: [], 1: []}
    for u in uttids:
        c = int(tag[u].strip())
        if c in by:
            by[c].append(u)
    sel = []
    for c in (0, 1):
        lst = by[c][:]; rng.shuffle(lst)
        sel += [(u, c) for u in lst[:args.max_per_class]]
    rng.shuffle(sel)
    print(f"対象: no={min(len(by[0]),args.max_per_class)}, "
          f"yes={min(len(by[1]),args.max_per_class)} (計 {len(sel)})")

    # 必要な全区間（対象＋文脈）を収集
    need = set()
    prev_map = {}
    for u, c in sel:
        own, intloc, anyp = find_prevs(u)
        prev_map[u] = (own, intloc, anyp)
        need.add(u)
        for p in (own, intloc, anyp):
            if p:
                need.add(p[0])
    print(f"抽出対象区間（対象＋文脈, ユニーク）: {len(need)}")

    # 特徴抽出
    feat = {}
    fail = 0
    for n, u in enumerate(sorted(need)):
        try:
            speech, _ = sf.read(wav[u], dtype="float32")
        except Exception:
            fail += 1; continue
        if speech.ndim > 1:
            speech = speech[:, 0]
        d = extract(model, speech, args.device)
        if d is None:
            fail += 1; continue
        feat[u] = d
        if (n + 1) % 2000 == 0:
            print(f"  抽出 {n + 1}/{len(need)} 件")
    print(f"抽出失敗: {fail} 件\n")

    D = next(iter(feat.values()))["cif_mean"].shape[0]
    zero = np.zeros(D, np.float32)

    def ctx(p):
        return feat[p[0]]["cif_mean"] if (p and p[0] in feat) else zero
    def gap(p):
        return np.float32(np.log1p(p[1])) if p else np.float32(-1.0)

    rows = {"cur": [], "own": [], "intloc": [], "anyp": [], "all": []}
    ys = []
    for u, c in sel:
        if u not in feat:
            continue
        own, intloc, anyp = prev_map[u]
        cur = feat[u]["cif_last"]
        rows["cur"].append(cur)
        rows["own"].append(np.concatenate([cur, ctx(own)]))
        rows["intloc"].append(np.concatenate([cur, ctx(intloc)]))
        rows["anyp"].append(np.concatenate([cur, ctx(anyp)]))
        rows["all"].append(np.concatenate(
            [cur, ctx(own), ctx(intloc), [gap(own)], [gap(intloc)]]))
        ys.append(c)
    y = np.array(ys); g = np.arange(len(y))

    print("=" * 76)
    print(f"線形プローブ: <no> vs <yes>（balanced, 発話単位 split, chance=0.5, n={len(y)}）")
    print("=" * 76)
    b0 = probe(np.stack(rows["cur"]), y, g, "cur のみ (baseline)", args.seed)
    b1 = probe(np.stack(rows["own"]), y, g, "cur + own_prev", args.seed)
    b2 = probe(np.stack(rows["intloc"]), y, g, "cur + intloc_prev", args.seed)
    b3 = probe(np.stack(rows["anyp"]), y, g, "cur + any_prev", args.seed)
    b4 = probe(np.stack(rows["all"]), y, g, "cur + own + intloc + gaps", args.seed)
    print("=" * 76)
    best = max(b1, b2, b3, b4)
    print(f"判定: baseline={b0:.3f} → 文脈つき最良={best:.3f}  (Δ={best-b0:+.3f})")
    if best - b0 >= 0.03:
        print("  → 過去文脈で改善。tag_classifier への過去文脈合流（v3）に進む価値あり。")
    else:
        print("  → 過去文脈でも 0.62 を破れない。streaming 制約下では構造的に困難。")
        print("     タスク枠組み（セグメント化・ラベル・早期確定）の再検討を推奨。")
    print("=" * 76)


if __name__ == "__main__":
    main()
