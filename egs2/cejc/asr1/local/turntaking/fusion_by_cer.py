#!/usr/bin/env python
"""融合の損失が「認識が崩れた発話」に集中しているかを測る。

天井（正解書き起こし）との差の 8 割近くが認識誤りで失われている。
その損失が高 CER の発話に集中しているなら N-best や信頼度重み付けが効く。
一様に散っているなら、1-best を増やしても取り返せない。

発話ごとに CER を出し、階層に分けて「音声のみ／ASR 融合／正解融合」を比べる。

  python local/turntaking/fusion_by_cer.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from frame_rules import utt_lengths                                 # noqa: E402
from frame_fusion import fuse, prepare                              # noqa: E402


def read_map(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            d[p[0]] = p[1] if len(p) > 1 else ""
    return d


def cer(ref, hyp):
    """文字単位の編集距離 ÷ 参照長。"""
    r = ref.replace(" ", ""); h = hyp.replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--bert_ref", default="exp/text_ceiling_bert")
    ap.add_argument("--asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/eval/text")
    ap.add_argument("--w", type=float, default=0.45, help="ASR 融合の重み（dev 選択）")
    ap.add_argument("--w_ref", type=float, default=0.35, help="正解融合の重み")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_sec", type=float, default=3.0)
    args = ap.parse_args()

    grid = np.arange(args.step, args.max_sec + 1e-9, args.step)
    P, Pt, lab, utts = prepare(args.stem, args.ckpt, "eval",
                               args.bert, args.asr_text, grid)
    _, Pt_ref, _, _ = prepare(args.stem, args.ckpt, "eval",
                              args.bert_ref, None, grid)
    lens = utt_lengths(utts, "eval")
    cap = np.clip(np.searchsorted(grid, lens + args.max_wait, side="right") - 1,
                  0, len(grid) - 1)
    idx = (cap, np.arange(len(utts)))

    ref = read_map("data/eval/text")
    hyp = read_map(args.asr_text)
    c = np.array([cer(ref.get(u, ""), hyp.get(u, "")) for u in utts])

    Pa = P[idx]
    Pf = fuse(P, Pt, grid, lens, args.w)[idx]
    Pr = fuse(P, Pt_ref, grid, lens, args.w_ref)[idx]

    print(f"eval {len(utts)} 区間 / 判定時点＝区間検出（発話末 + {args.max_wait} 秒）")
    print(f"CER 平均 {c.mean():.3f} / 中央値 {np.median(c):.3f} / "
          f"CER=0 の割合 {(c == 0).mean():.3f}\n")

    edges = [(0.0, 0.0), (0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 9.9)]
    print(f"{'CER 帯':>12}{'件数':>8}{'割合':>7}{'音声のみ':>10}{'ASR 融合':>10}"
          f"{'正解融合':>10}{'取り逃し':>10}")
    tot_gap = 0.0
    for lo, hi in edges:
        m = (c == 0) if hi == lo else ((c > lo) & (c <= hi))
        if m.sum() < 30:
            continue
        ma = prf(lab[m], Pa[m].argmax(1))[1]
        mf = prf(lab[m], Pf[m].argmax(1))[1]
        mr = prf(lab[m], Pr[m].argmax(1))[1]
        gap = mr - mf
        tot_gap += gap * m.mean()
        name = "0（完全一致）" if hi == lo else f"{lo:.1f}〜{hi:.1f}"
        print(f"{name:>12}{m.sum():8d}{m.mean():7.3f}{ma:10.4f}{mf:10.4f}"
              f"{mr:10.4f}{gap:+10.4f}")

    print(f"\n全体{'':>8}{len(utts):8d}{1.0:7.3f}"
          f"{prf(lab, Pa.argmax(1))[1]:10.4f}{prf(lab, Pf.argmax(1))[1]:10.4f}"
          f"{prf(lab, Pr.argmax(1))[1]:10.4f}")
    print(f"\n帯ごとの取り逃しを件数で重み付けした和: {tot_gap:+.4f}")
    print("（集中しているなら N-best や信頼度重み付けに余地がある）")


if __name__ == "__main__":
    main()
