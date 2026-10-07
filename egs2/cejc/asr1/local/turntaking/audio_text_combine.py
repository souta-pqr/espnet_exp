#!/usr/bin/env python
"""音声モデルと「ASR 認識結果 → BERT」を組み合わせて上回るか調べる。

両者は偶然にも同じ精度（継続 vs 完了 AUC 0.725 対 0.7246）だが、
**同じ誤りをしているとは限らない**。誤りが部分的に独立なら、
確率を混ぜるだけで両方を上回れる。逆に完全に相関していれば、
音声モデルが認識結果以上の情報を使えていないことの裏づけになる。

判定時点はどちらも「音声区間検出が発話末 + 0.25 秒で発火した瞬間」にそろえる。

  python local/turntaking/audio_text_combine.py
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from text_ceiling import auc_cont_end, build                        # noqa: E402


def audio_probs_at(stem, ckpt, dset, offset):
    """フレーム単位ダンプから「発話末 + offset 秒」の確率を取り出す。"""
    el = collections.defaultdict(list); pr = collections.defaultdict(list); lab = {}
    with open(f"tt_preds/{stem}_{dset}_{ckpt}.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            u = p[0]; lab[u] = int(p[1])
            el[u].append(float(p[3]))
            pr[u].append((float(p[4]), float(p[5]), float(p[6])))
    seg = {}
    for line in open(f"data/{dset}/segments"):
        u, r, b, e = line.split(); seg[u] = float(e) - float(b)
    out = {}
    for u in el:
        e = np.asarray(el[u]); q = np.asarray(pr[u])
        j = int(np.clip(np.searchsorted(e, seg[u] + offset, side="right") - 1,
                        0, len(e) - 1))
        out[u] = q[j]
    return out, lab


def text_probs(model_dir, dset, text_scp, bs=256, max_len=128):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = build(dset, text_scp=text_scp)
    tk = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()

    class DS(Dataset):
        def __init__(self, r): self.r = r
        def __len__(self): return len(self.r)
        def __getitem__(self, i): return self.r[i]

    def collate(b):
        return (tk([x["text"] for x in b], truncation=True, max_length=max_len,
                   padding=True, return_tensors="pt"), [x["utt"] for x in b])

    out = {}
    with torch.no_grad():
        for enc, utts in DataLoader(DS(rows), batch_size=bs, collate_fn=collate,
                                    num_workers=4):
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p = model(**enc).logits.float().softmax(-1).cpu().numpy()
            for u, q in zip(utts, p):
                out[u] = q
    return out


def report(name, P, y):
    pred = P.argmax(1)
    f1, mac, uar, acc = prf(y, pred)
    a = auc_cont_end(P, y)
    print(f"{name:>34}  macro-F1 {mac:.4f}   AUC {a:.4f}   "
          f"継続 {f1[0]:.3f} 完了 {f1[1]:.3f} 相槌 {f1[2]:.3f}")
    return mac, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--dset", default="eval")
    ap.add_argument("--bert", default="exp/text_ceiling_bert")
    ap.add_argument("--asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/eval/text")
    ap.add_argument("--offset", type=float, default=0.25,
                    help="区間検出が発火する時刻（発話末からの秒数）")
    ap.add_argument("--dev_asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/"
                    "train_dev_dec/text",
                    help="dev の ASR 認識結果。混合重みはここで選ぶ")
    args = ap.parse_args()

    A, lab = audio_probs_at(args.stem, args.ckpt, args.dset, args.offset)
    T_asr = text_probs(args.bert, args.dset, args.asr_text)
    T_ref = text_probs(args.bert, args.dset, None)

    utts = sorted(set(A) & set(T_asr) & set(T_ref))
    y = np.array([lab[u] for u in utts])
    Pa = np.array([A[u] for u in utts])
    Pt = np.array([T_asr[u] for u in utts])
    Pr = np.array([T_ref[u] for u in utts])
    print(f"{args.dset} {len(utts)} 区間 / 判定時点 = 発話末 + {args.offset} 秒\n")

    report("音声モデル", Pa, y)
    report("ASR 認識結果 → BERT", Pt, y)
    report("正解書き起こし → BERT（到達不能）", Pr, y)

    # 誤りが独立か
    ea = Pa.argmax(1) != y; et = Pt.argmax(1) != y
    both = (ea & et).mean(); either = (ea | et).mean()
    print(f"\n両方とも誤り {both:.3f} / どちらかが誤り {either:.3f} / "
          f"音声のみ誤り {(ea & ~et).mean():.3f} / テキストのみ誤り {(~ea & et).mean():.3f}")
    print(f"誤りの一致度（φ係数）: "
          f"{np.corrcoef(ea.astype(float), et.astype(float))[0,1]:.3f}")

    # 混合重みは dev で選ぶ（eval 上で掃引すると楽観的になる）
    Ad, labd = audio_probs_at(args.stem, args.ckpt, "train_dev", args.offset)
    Td = text_probs(args.bert, "train_dev", args.dev_asr_text)
    du = sorted(set(Ad) & set(Td))
    yd = np.array([labd[u] for u in du])
    Pad = np.array([Ad[u] for u in du]); Ptd = np.array([Td[u] for u in du])
    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    devs = [(w, prf(yd, (w * Pad + (1 - w) * Ptd).argmax(1))[1]) for w in grid]
    w_best = max(devs, key=lambda x: x[1])[0]
    print(f"\ndev {len(du)} 区間で選んだ重み: 音声 {w_best:.2f} "
          f"（dev macro-F1 {max(d[1] for d in devs):.4f}）")
    print()
    report(f"混合（dev で選択・音声{w_best:.2f}）", w_best * Pa + (1 - w_best) * Pt, y)
    for w in (0.3, 0.5, 0.7):
        report(f"　参考 音声{w:.1f}", w * Pa + (1 - w) * Pt, y)


if __name__ == "__main__":
    main()
