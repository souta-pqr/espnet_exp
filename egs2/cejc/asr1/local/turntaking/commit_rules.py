#!/usr/bin/env python
"""早期確定の規則を dev で選び、eval で報告する（融合確率に対応）。

これまでの frame_fusion.py は融合の重みこそ dev で選んでいたが、
「基準を崩さず最も早い完了 τ」は **eval 上で選んでいた**（楽観的）。
動作点は dev から eval へ転移しにくいことが分かっている（HANDOFF §2.5）ので、
規則も dev で選んで eval で報告し、条件（どちらの失敗も基準 +0.005 以内）が
eval でも守れているかを併記する。

規則の族（どれも「完了」だけを早期確定し、それ以外は区間検出まで待つ）:
  完了τ          … p(完了) ≥ τ
  ＋持続 k       … k フレーム（0.05 秒刻み）連続で条件を満たしたら確定
  ＋ゲート g     … 発話開始から g 秒は確定しない
  余裕δ          … p(完了) − p(継続) ≥ δ（τ の代わり）

  python local/turntaking/commit_rules.py --stems frame_attn-frame2-blk42xh2_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import prf                                        # noqa: E402
from frame_rules import utt_lengths                                 # noqa: E402
from frame_fusion import fuse, prepare                              # noqa: E402

TOL = 0.005
ASR42 = ("exp/asr_20260921-pureasr-blk42-sp/"
         "decode_cbs_transducer_bounded_asr_model_valid.loss.ave")


def persist(ok, k):
    """ok (S,N) → k フレーム連続で真になった時点で真。"""
    if k == 1:
        return ok
    run = np.zeros(ok.shape, dtype=np.int32)
    run[0] = ok[0]
    for s in range(1, ok.shape[0]):
        run[s] = np.where(ok[s], run[s - 1] + 1, 0)
    return run >= k


class Data:
    def __init__(self, P, lab, lens, grid, max_wait):
        self.P, self.lab, self.lens, self.grid = P, lab, lens, grid
        S = len(grid)
        self.cap = np.clip(np.searchsorted(grid, lens + max_wait, side="right") - 1,
                           0, S - 1)
        self.am = P.argmax(-1)                                       # (S,N)
        self.ix = np.arange(len(lab))

    def point(self, ok):
        """ok (S,N) が初めて真になった時点で「完了」と確定。
        → (応答ミス, 誤割り込み, 完了の遅延, macro-F1, 予測)"""
        S = len(self.grid)
        first = np.where(ok.any(0), ok.argmax(0), S)
        fired = first <= self.cap
        idx = np.where(fired, first, self.cap)
        pr = np.where(fired, 1, self.am[idx, self.ix])
        lat = self.grid[idx] - self.lens
        _, mac, _, _ = prf(self.lab, pr)
        m1 = self.lab == 1
        return ((pr[m1] != 1).mean(), ((pr == 1) & ~m1).sum() / max((~m1).sum(), 1),
                lat[m1].mean(), mac, pr)


def rule_masks(d):
    """規則名 → ok (S,N)。"""
    P, g = d.P, d.grid[:, None]
    out = {}
    for t in np.round(np.arange(0.40, 1.00, 0.02), 2):
        base = P[..., 1] >= t
        for k in (1, 2, 3):
            pk = persist(base, k)
            for gate in (0.0, 0.25, 0.5, 0.75, 1.0):
                out[("完了τ", t, k, gate)] = pk & (g >= gate) if gate else pk
    marg = P[..., 1] - P[..., 0]
    for dl in np.round(np.arange(0.0, 1.0, 0.04), 2):
        base = marg >= dl
        for k in (1, 2, 3):
            pk = persist(base, k)
            for gate in (0.0, 0.5):
                out[("余裕δ", dl, k, gate)] = pk & (g >= gate) if gate else pk
    return out


def name(r):
    fam, v, k, gate = r
    s = f"{fam}={v:.2f}"
    if k > 1:
        s += f" 持続{k}"
    if gate:
        s += f" ゲート{gate:g}s"
    return s


def evaluate_all(d):
    return {r: d.point(ok)[:4] for r, ok in rule_masks(d).items()}


def fastest_ok(pts, base, tol=TOL):
    """どちらの失敗も基準 +tol 以内で、完了の遅延が最小の規則。"""
    cand = [(r, v) for r, v in pts.items()
            if v[0] <= base[0] + tol and v[1] <= base[1] + tol]
    return min(cand, key=lambda x: x[1][2]) if cand else None


def load(stem, ckpt, dset, asr_text, grid, max_wait, w, bert="exp/text_bert_asr"):
    P, Pt, lab, utts = prepare(stem, ckpt, dset, bert, asr_text, grid)
    lens = utt_lengths(utts, dset)
    return P, Pt, lab, utts, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--asr_dir", default=ASR42)
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--dev_set", default="train_dev")
    ap.add_argument("--dev_asr_name", default="train_dev_dec",
                    help="--asr_dir の下の dev 側の認識結果のディレクトリ名（新しい分割では dev_g）")
    ap.add_argument("--eval_set", default="eval")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max_sec", type=float, default=3.0)
    ap.add_argument("--gold", default="tt_analysis/gold1000_uttid.tsv")
    ap.add_argument("--persist", type=int, default=0,
                    help=">0 なら dev での選択を「持続 k フレーム」の規則に限る（転移の頑健さを見る）")
    args = ap.parse_args()
    grid = np.arange(args.step, args.max_sec + 1e-9, args.step)
    gold = {}
    if Path(args.gold).exists():
        gold = {l.split("\t")[0]: int(l.split("\t")[1])
                for l in list(open(args.gold))[1:]}

    for stem in args.stems:
        print(f"\n{'=' * 90}\n■ {stem}\n{'=' * 90}")
        Pd, Ptd, labd, uttsd, lensd = load(stem, args.ckpt, args.dev_set,
                                           f"{args.asr_dir}/{args.dev_asr_name}/text",
                                           grid, args.max_wait, None, args.bert)
        Pe, Pte, labe, uttse, lense = load(stem, args.ckpt, args.eval_set,
                                           f"{args.asr_dir}/{args.eval_set}/text",
                                           grid, args.max_wait, None, args.bert)
        # 融合の重みは dev の「区間検出まで待つ」macro-F1 で選ぶ（frame_fusion.py と同じ手続き）
        ws = np.round(np.arange(0.0, 1.01, 0.05), 2)
        never = np.zeros((len(grid), len(labd)), dtype=bool)
        w_best = max(ws, key=lambda w: Data(fuse(Pd, Ptd, grid, lensd, w), labd, lensd,
                                            grid, args.max_wait).point(never)[3])
        print(f"融合の重み（dev 選択）: 音声 {w_best:.2f}")

        for cond, w in (("音声のみ", 1.0), (f"融合（音声{w_best:.2f}）", w_best)):
            dd = Data(fuse(Pd, Ptd, grid, lensd, w), labd, lensd, grid, args.max_wait)
            de = Data(fuse(Pe, Pte, grid, lense, w), labe, lense, grid, args.max_wait)
            nev = np.zeros((len(grid), len(labe)), dtype=bool)
            bd = dd.point(np.zeros((len(grid), len(labd)), dtype=bool))
            be = de.point(nev)
            ptsd = evaluate_all(dd)
            ptse = evaluate_all(de)
            print(f"\n--- {cond} ---")
            print(f"{'':>36}{'応答ミス':>9}{'誤割り込み':>10}{'完了の遅延':>11}"
                  f"{'macroF1':>9}{'短縮':>8}  条件")
            def row(lbl, v, base):
                okc = "○" if (v[0] <= base[0] + TOL and v[1] <= base[1] + TOL) else "×"
                print(f"{lbl:>36}{v[0]:9.3f}{v[1]:10.3f}{v[2]:+11.3f}{v[3]:9.4f}"
                      f"{base[2] - v[2]:8.3f}  {okc}")
            row("基準（区間検出まで待つ）", be[:4], be)
            # 従来の比較対象：完了τ だけ、eval で選ぶ（frame_fusion.py の「最良 τ」）
            simple = {r: v for r, v in ptse.items() if r[0] == "完了τ" and r[2] == 1 and not r[3]}
            r = fastest_ok(simple, be)
            if r:
                row(f"従来: {name(r[0])}（eval で選択）", r[1], be)
            # 完了τ だけ・dev で選択
            simple_d = {r: v for r, v in ptsd.items() if r[0] == "完了τ" and r[2] == 1 and not r[3]}
            r = fastest_ok(simple_d, bd)
            if r:
                row(f"{name(r[0])}（dev で選択）", ptse[r[0]], be)
            # 全規則・dev で選択
            if args.persist:
                ptsd = {k: v for k, v in ptsd.items() if k[2] == args.persist}
            r = fastest_ok(ptsd, bd)
            if r:
                row(f"{name(r[0])}（dev で選択）", ptse[r[0]], be)
                best_rule = r[0]
            else:
                best_rule = None
            # 全規則・eval で選択（到達上限の参考）
            r = fastest_ok(ptse, be)
            if r:
                row(f"参考 {name(r[0])}（eval で選択）", r[1], be)
            # 参考：τ=0.70（早さ重視の固定点）
            row("参考 完了τ=0.70（固定）", ptse[("完了τ", 0.70, 1, 0.0)], be)

            # 人手正解との突き合わせ（eval と dev の学習外の区間のみ）
            if gold and best_rule is not None:
                rows = []
                for D, utts in ((de, uttse), (dd, uttsd)):
                    pr_w = D.point(np.zeros((len(grid), len(utts)), dtype=bool))[4]
                    pr_r = D.point(rule_masks_one(D, best_rule))[4]
                    for i, u in enumerate(utts):
                        if u in gold and D.lab[i] in (0, 1):
                            rows.append((gold[u], int(D.lab[i]), int(pr_w[i]), int(pr_r[i])))
                if rows:
                    a = np.array(rows)
                    print(f"  人手正解のある学習外区間 n={len(a)}: "
                          f"人手との一致 待つ {np.mean(a[:, 0] == a[:, 2]):.3f} / "
                          f"選んだ規則 {np.mean(a[:, 0] == a[:, 3]):.3f} / "
                          f"LLM ラベル {np.mean(a[:, 0] == a[:, 1]):.3f}")
                    fa = (a[:, 3] == 1) & (a[:, 1] != 1)
                    if fa.any():
                        print(f"  選んだ規則の「誤割り込み」（LLM ラベル≠完了で完了と確定）{fa.sum()} 件のうち "
                              f"人手では完了 {np.mean(a[fa, 0] == 1):.2f}")


def rule_masks_one(d, r):
    fam, v, k, gate = r
    P, g = d.P, d.grid[:, None]
    base = (P[..., 1] >= v) if fam == "完了τ" else (P[..., 1] - P[..., 0] >= v)
    pk = persist(base, k)
    return pk & (g >= gate) if gate else pk


if __name__ == "__main__":
    main()
