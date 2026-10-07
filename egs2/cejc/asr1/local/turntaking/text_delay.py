#!/usr/bin/env python
"""テキスト（BERT）が届くのが遅れたら、融合の結果はどう変わるか（2026-10-07）。

これまでの評価は、認識結果と BERT の確率が「発話末ちょうど」から使えるとしていた。
実運用では認識の確定（先読み 0.1 秒）と BERT の計算の分だけ遅れるので、
テキストを混ぜ始める時刻を 発話末 + Δ 秒 に遅らせて測り直す。

- 区間検出が発火した時点（発話末 + max_wait）の最終判定では、Δ ≤ max_wait ならテキストが届いているとする
  （0.05 秒刻みの格子だと発火時点が発話末 + 0.20〜0.25 秒になり、Δ=0.25 で一切使えなくなるのを避けるため）。
- Δ = ∞ はテキストを使わない（音声入力の判定器のみ）。
- 融合の重みは Δ ごとに開発データで選ぶ（区間検出まで待つときの macro-F1 が最大）。
- 早期確定は「P(完了) − P(継続) ≥ δ が 3 フレーム続いたら確定」の δ を開発データで選ぶ（commit_rules.py と同じ）。

  python local/turntaking/text_delay.py --stem frame_attn-frame2-blk42xh2_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commit_rules import Data, persist, TOL                      # noqa: E402
from frame_fusion import prepare                                 # noqa: E402
from frame_rules import utt_lengths                              # noqa: E402

ASR42 = ("exp/asr_20260921-pureasr-blk42-sp/"
         "decode_cbs_transducer_bounded_asr_model_valid.loss.ave")


def fuse_delay(P, Pt, grid, lens, w, delta, cap):
    """発話末 + delta 秒以降のフレームだけ、音声 w・テキスト (1-w) で混ぜる。"""
    if not np.isfinite(delta):
        return P
    avail = grid[:, None] >= lens[None, :] + delta - 1e-9
    S, N = avail.shape
    # 区間検出が発火した時点では届いている。ただし判定は発話開始から 3.0 秒までなので、
    # それより長い発話は最後の判定時点が発話末より前になる → そこでテキストを混ぜると未来が漏れる。
    ok = grid[cap] >= lens                          # 最後の判定時点が発話末以降の区間だけ
    avail[cap[ok], np.arange(N)[ok]] = True
    mixed = w * P + (1.0 - w) * Pt[None, :, :]
    return np.where(avail[:, :, None], mixed, P)


def persist_rules(d):
    out = {}
    marg = d.P[..., 1] - d.P[..., 0]
    for dl in np.round(np.arange(0.0, 1.0, 0.04), 2):
        out[dl] = d.point(persist(marg >= dl, 3))[:4]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--asr_dir", default=ASR42)
    ap.add_argument("--dev_set", default="train_dev")
    ap.add_argument("--dev_asr_name", default="train_dev_dec")
    ap.add_argument("--eval_set", default="eval")
    ap.add_argument("--delays", default="0 0.1 0.15 0.2 0.25 inf")
    ap.add_argument("--max_wait", type=float, default=0.25)
    args = ap.parse_args()
    grid = np.arange(0.05, 3.0 + 1e-9, 0.05)

    Pd, Ptd, labd, uttsd = prepare(args.stem, args.ckpt, args.dev_set, args.bert,
                                   f"{args.asr_dir}/{args.dev_asr_name}/text", grid)
    Pe, Pte, labe, uttse = prepare(args.stem, args.ckpt, args.eval_set, args.bert,
                                   f"{args.asr_dir}/{args.eval_set}/text", grid)
    lensd = utt_lengths(uttsd, args.dev_set); lense = utt_lengths(uttse, args.eval_set)
    capd = Data(Pd, labd, lensd, grid, args.max_wait).cap
    cape = Data(Pe, labe, lense, grid, args.max_wait).cap
    nod = np.zeros((len(grid), len(labd)), bool); noe = np.zeros((len(grid), len(labe)), bool)
    print(f"評価 {len(labe)} 区間 / 判定は 0.05 秒刻み・発話開始から 3.0 秒まで / 区間検出 {args.max_wait} 秒")
    print(f"テキストが届く前に区間検出が発火する区間は無い前提（Δ ≤ {args.max_wait}）\n")
    print(f"{'Δ（秒）':>8}{'重み':>6}  {'区間検出まで待つ':^28}  {'早期確定（δ＋持続3，開発で選択）':^40}  {'τ=0.70（固定）':^28}")
    print(f"{'':14}{'応答ミス':>8}{'誤割込':>7}{'macroF1':>9}   {'δ':>4}{'応答ミス':>8}{'誤割込':>7}{'macroF1':>9}{'短縮':>7} 条件  "
          f"{'応答ミス':>8}{'誤割込':>7}{'macroF1':>9}{'短縮':>7}")
    for ds in args.delays.split():
        delta = float(ds)
        ws = [1.0] if not np.isfinite(delta) else np.round(np.arange(0.0, 1.01, 0.05), 2)
        w = max(ws, key=lambda w: Data(fuse_delay(Pd, Ptd, grid, lensd, w, delta, capd),
                                       labd, lensd, grid, args.max_wait).point(nod)[3])
        dd = Data(fuse_delay(Pd, Ptd, grid, lensd, w, delta, capd), labd, lensd, grid, args.max_wait)
        de = Data(fuse_delay(Pe, Pte, grid, lense, w, delta, cape), labe, lense, grid, args.max_wait)
        bd = dd.point(nod); be = de.point(noe)
        rd = persist_rules(dd)
        cand = [(k, v) for k, v in rd.items() if v[0] <= bd[0] + TOL and v[1] <= bd[1] + TOL]
        kd = min(cand, key=lambda x: x[1][2])[0] if cand else None
        line = f"{ds:>8}{w:6.2f}  {be[0]:8.3f}{be[1]:7.3f}{be[3]:9.4f}   "
        if kd is not None:
            v = de.point(persist((de.P[..., 1] - de.P[..., 0]) >= kd, 3))
            ok = "○" if v[0] <= be[0] + TOL and v[1] <= be[1] + TOL else "×"
            line += f"{kd:4.2f}{v[0]:8.3f}{v[1]:7.3f}{v[3]:9.4f}{be[2] - v[2]:7.3f}  {ok}   "
        else:
            line += f"{'（基準を超えられない）':>36}   "
        v = de.point(de.P[..., 1] >= 0.70)
        line += f"{v[0]:8.3f}{v[1]:7.3f}{v[3]:9.4f}{be[2] - v[2]:7.3f}"
        print(line, flush=True)
    print("\nΔ=inf はテキストを使わない（音声入力の判定器のみ）。短縮は区間検出まで待つ場合からの前倒し（秒）。")


if __name__ == "__main__":
    main()
