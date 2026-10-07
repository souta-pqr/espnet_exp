#!/usr/bin/env python
"""認識が信用できない発話ではテキストを使わない（門番つき融合）。

CER 帯別の分析で、認識が崩れた発話ではテキストを混ぜると**悪化する**と分かった。
  CER 0.5〜1.0（18.7%）音声のみ 0.6400 → 融合 0.6396
  CER 1.0 超　 （ 4.6%）音声のみ 0.5822 → 融合 0.5638
7 割強で稼いだ分を 2 割強で吐き出している。

正解 CER で門番をした場合（到達不能な上限）と、実際に使える代理量
  cps  … 1 秒あたり文字数。暴走した挿入で異常に高くなる
  rep  … 文字 2-gram の反復率。Transducer のループ
での結果を並べる。閾値は dev で選ぶ。

  python local/turntaking/fusion_gate.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from frame_rules import utt_lengths                                 # noqa: E402
from frame_fusion import prepare                                    # noqa: E402
from fusion_by_cer import cer, read_map                             # noqa: E402


def feats(utts, hyp, lens):
    """発話ごとの代理量。"""
    cps = np.zeros(len(utts)); rep = np.zeros(len(utts))
    for i, u in enumerate(utts):
        h = hyp.get(u, "").replace(" ", "")
        cps[i] = len(h) / max(lens[i], 0.1)
        g = [h[j:j + 2] for j in range(len(h) - 1)]
        rep[i] = 1.0 - len(set(g)) / max(len(g), 1)
    return cps, rep


def mix(Pa, Pt, w, avail):
    """発話ごとの重み w（音声側）で混ぜる。w=1 なら音声のみ。

    avail は「判定時点で発話末を過ぎている」か。フレームダンプは 3 秒までなので、
    長い発話では判定時点が発話の途中に来る。そこで認識結果を使うと未来が漏れる。
    """
    w = np.where(avail, w, 1.0)[:, None]
    return w * Pa + (1.0 - w) * Pt


def load(stem, ckpt, dset, bert, asr_text, grid, max_wait):
    P, Pt, lab, utts = prepare(stem, ckpt, dset, bert, asr_text, grid)
    lens = utt_lengths(utts, dset)
    cap = np.clip(np.searchsorted(grid, lens + max_wait, side="right") - 1,
                  0, len(grid) - 1)
    Pa = P[cap, np.arange(len(utts))]
    avail = grid[cap] >= lens            # 判定時点で発話末を過ぎているか
    hyp = read_map(asr_text) if asr_text else read_map(f"data/{dset}/text")
    cps, rep = feats(utts, hyp, lens)
    return Pa, Pt, lab, utts, lens, cps, rep, hyp, avail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/eval/text")
    ap.add_argument("--dev_asr_text", default="exp/asr_20260713-pureasr/"
                    "decode_cbs_transducer_bounded_asr_model_valid.loss.ave/"
                    "train_dev_dec/text")
    ap.add_argument("--w", type=float, default=0.45)
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_sec", type=float, default=3.0)
    args = ap.parse_args()

    grid = np.arange(args.step, args.max_sec + 1e-9, args.step)
    print("dev を読み込み中…")
    dPa, dPt, dlab, dutts, dlens, dcps, drep, _, davail = load(
        args.stem, args.ckpt, "train_dev", args.bert, args.dev_asr_text,
        grid, args.max_wait)
    print("eval を読み込み中…")
    Pa, Pt, lab, utts, lens, cps, rep, hyp, avail = load(
        args.stem, args.ckpt, "eval", args.bert, args.asr_text,
        grid, args.max_wait)

    def macro(w, P_a, P_t, y, av):
        return prf(y, mix(P_a, P_t, w, av).argmax(1))[1]

    base_a = macro(np.ones(len(lab)), Pa, Pt, lab, avail)
    base_f = macro(np.full(len(lab), args.w), Pa, Pt, lab, avail)
    print(f"\neval {len(utts)} 区間 / 判定時点＝区間検出（発話末 + {args.max_wait} 秒）")
    print(f"判定時点で発話末を過ぎている割合 {avail.mean():.3f}"
          f"（残りは 3 秒の上限に張り付いた長い発話。テキストは使えない）")
    print(f"音声のみ {base_a:.4f} ／ 門番なし融合（音声{args.w:.2f}）{base_f:.4f}\n")

    print(f"{'門番':>22}{'閾値':>8}{'通過率':>8}{'eval macroF1':>14}{'差':>9}")

    # 到達不能な上限：正解 CER で門番をする
    ref = read_map("data/eval/text")
    c = np.array([cer(ref.get(u, ""), hyp.get(u, "")) for u in utts])
    for th in (0.2, 0.5, 0.8, 1.0):
        w = np.where(c <= th, args.w, 1.0)
        m = macro(w, Pa, Pt, lab, avail)
        print(f"{'正解 CER（到達不能）':>22}{th:8.2f}{(c <= th).mean():8.3f}"
              f"{m:14.4f}{m - base_f:+9.4f}")

    # 実際に使える代理量：閾値は dev で選ぶ
    for name, dv, ev in (("1 秒あたり文字数", dcps, cps), ("2-gram 反復率", drep, rep)):
        cand = np.quantile(dv, np.clip(np.arange(0.50, 1.001, 0.02), 0.0, 1.0))
        devs = [(t, macro(np.where(dv <= t, args.w, 1.0), dPa, dPt, dlab, davail))
                for t in cand]
        t_best = max(devs, key=lambda x: x[1])[0]
        w = np.where(ev <= t_best, args.w, 1.0)
        m = macro(w, Pa, Pt, lab, avail)
        print(f"{name:>22}{t_best:8.2f}{(ev <= t_best).mean():8.3f}"
              f"{m:14.4f}{m - base_f:+9.4f}")

    # 代理量と正解 CER がどれだけ対応しているか
    print(f"\n代理量と CER の順位相関（Spearman 近似）")
    for name, v in (("1 秒あたり文字数", cps), ("2-gram 反復率", rep)):
        r = np.corrcoef(np.argsort(np.argsort(v)).astype(float),
                        np.argsort(np.argsort(c)).astype(float))[0, 1]
        print(f"{name:>22}{r:8.3f}")


if __name__ == "__main__":
    main()
