#!/usr/bin/env python
"""「継続」を重視したときの判定と早期確定（2026-10-09）。

システムにとっては、継続を見逃して相手を遮る方が害が大きい。そこで
  1. VAD 発火時の最終判定で P(継続) に重み b を掛けて「継続」寄りにずらし、継続と完了の再現率の入れ替わりを見る。
  2. 早期確定の規則を「継続の再現率と完了の適合率を基準より落とさない（0.005 以内）」条件で開発データから選ぶ。
  3. 新しいラベル（3 モデルの多数決）で測り直す。
音声のみ（実際の推論）と、BERT も混ぜる構成（参考）の両方。

  python local/turntaking/keizoku_focus.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commit_rules import Data, persist, ASR42, TOL                  # noqa: E402
from frame_fusion import prepare, fuse                              # noqa: E402
from frame_rules import utt_lengths                                 # noqa: E402
from vote_tags import read_votes                                    # noqa: E402

V = "tt_analysis/relabel_votes_final/"


def biased(P, b):
    """P(継続) に b を掛けて正規化。"""
    Q = P.copy()
    Q[..., 0] *= b
    return Q / Q.sum(-1, keepdims=True)


def scores(lab, pr):
    out = {}
    for c, nm in ((0, "継続"), (1, "完了"), (2, "相槌")):
        tp = ((pr == c) & (lab == c)).sum()
        p = tp / max((pr == c).sum(), 1)
        r = tp / max((lab == c).sum(), 1)
        out[nm] = (r, p, 2 * p * r / max(p + r, 1e-9))
    out["macro"] = np.mean([out[k][2] for k in ("継続", "完了", "相槌")])
    return out


def row(lbl, s, speed=None):
    sp = f"{speed:7.2f}" if speed is not None else f"{'—':>7s}"
    print(f"{lbl:30s}{s['継続'][0]:8.3f}{s['継続'][1]:8.3f}{s['完了'][0]:8.3f}{s['完了'][1]:8.3f}"
          f"{s['相槌'][2]:8.3f}{s['macro']:8.3f}{sp}")


def head():
    print(f"{'':30s}{'継続再現':>8s}{'継続適合':>8s}{'完了再現':>8s}{'完了適合':>8s}{'相槌F1':>8s}{'macro':>8s}{'早くなる':>7s}")


def new_labels(utts, lab):
    """新しいラベル：LLM が判定する区間だけ多数決で置き換える（相槌・ルールの区間はそのまま）。"""
    a = {}
    for f in ("Qwen3-32B-AWQ-nothink.tsv", "Qwen3-32B-AWQ-nothink.silver13.tsv"):
        a.update(read_votes(V + f))
    b = read_votes(V + "Qwen3-8B-think.tsv")
    u2l = {}; q = {}
    for l in list(open("tt_analysis/qwen3_fc_os_uttid.tsv"))[1:]:
        ln, u, r = l.rstrip("\n").split("\t")
        u2l[u] = int(ln); q[u] = int(r)
    out = lab.copy(); n = 0
    for i, u in enumerate(utts):
        ln = u2l.get(u)
        if ln is None or q[u] == 2 or ln not in a or ln not in b:
            continue
        v = int(q[u] + a[ln] + b[ln] >= 2)
        if lab[i] in (0, 1):
            out[i] = v; n += 1
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2-blk42xh2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    args = ap.parse_args()
    grid = np.arange(0.05, 3.0 + 1e-9, 0.05)
    Pd, Ptd, labd, uttsd = prepare(args.stem, args.ckpt, "train_dev", "exp/text_bert_asr",
                                   f"{ASR42}/train_dev_dec/text", grid)
    Pe, Pte, labe, uttse = prepare(args.stem, args.ckpt, "eval", "exp/text_bert_asr",
                                   f"{ASR42}/eval/text", grid)
    Ld = utt_lengths(uttsd, "train_dev"); Le = utt_lengths(uttse, "eval")
    labe_new, nchg = new_labels(uttse, labe)
    labd_new, _ = new_labels(uttsd, labd)
    print(f"評価 {len(labe)} 区間（継続 {np.sum(labe == 0)}・完了 {np.sum(labe == 1)}・相槌 {np.sum(labe == 2)}）")
    print(f"新しいラベル：継続 {np.sum(labe_new == 0)}・完了 {np.sum(labe_new == 1)}（{np.sum(labe_new != labe)} 区間が変わる）")

    for cond, w in (("音声のみ（実際の推論）", 1.0), ("参考：BERT も混ぜる", 0.6)):
        Fd = fuse(Pd, Ptd, grid, Ld, w); Fe = fuse(Pe, Pte, grid, Le, w)
        print(f"\n{'=' * 90}\n■ {cond}")
        for lname, ld, le in (("元のラベル", labd, labe), ("新しいラベル", labd_new, labe_new)):
            print(f"\n--- 1. 最終判定を継続寄りにする（{lname}で評価．早期確定なし）---")
            head()
            for b in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
                de = Data(biased(Fe, b), le, Le, grid, 0.25)
                pr = de.point(np.zeros((len(grid), len(le)), bool))[4]
                row(f"P(継続)×{b:g}", scores(le, pr))

        print(f"\n--- 2. 早期確定の規則の選び方（元のラベルで選び，評価）---")
        head()
        dd = Data(Fd, labd, Ld, grid, 0.25); de = Data(Fe, labe, Le, grid, 0.25)
        nod = np.zeros((len(grid), len(labd)), bool); noe = np.zeros((len(grid), len(labe)), bool)
        bd = scores(labd, dd.point(nod)[4]); be_pt = de.point(noe)
        row("基準：VAD まで待つ", scores(labe, be_pt[4]))
        margd = dd.P[..., 1] - dd.P[..., 0]; marge = de.P[..., 1] - de.P[..., 0]
        cands = []
        for dl in np.round(np.arange(0.0, 1.0, 0.02), 2):
            v = dd.point(persist(margd >= dl, 3))
            s = scores(labd, v[4])
            cands.append((dl, v, s))
        bdpt = dd.point(nod)
        # 従来：応答ミス・誤割り込みが基準 +0.005 以内
        old = [c for c in cands if c[1][0] <= bdpt[0] + TOL and c[1][1] <= bdpt[1] + TOL]
        # 継続重視：継続の再現率と完了の適合率が基準 −0.005 以内
        new = [c for c in cands if c[2]["継続"][0] >= bd["継続"][0] - TOL and c[2]["完了"][1] >= bd["完了"][1] - TOL]
        for nm, cs in (("従来の選び方", old), ("継続重視の選び方", new)):
            if not cs:
                print(f"{nm}: 条件を満たす δ なし"); continue
            dl = min(cs, key=lambda c: c[1][2])[0]
            v = de.point(persist(marge >= dl, 3))
            row(f"{nm}（δ={dl:.2f}）", scores(labe, v[4]), be_pt[2] - v[2])
        v = de.point(de.P[..., 1] >= 0.70)
        row("参考 P(完了)≥0.70", scores(labe, v[4]), be_pt[2] - v[2])

        print(f"\n--- 3. 継続寄りの最終判定＋継続重視の早期確定（b と δ を開発データ・元のラベルで選ぶ）---")
        head()
        # b：完了の再現率の低下が 0.03 以内に収まる中で最も大きいもの（開発データ）
        # δ：その b の下で，継続の再現率と完了の適合率を落とさない（0.005 以内）中で最も早いもの
        best = None
        for b in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
            ddb = Data(biased(Fd, b), labd, Ld, grid, 0.25)
            sb = scores(labd, ddb.point(nod)[4])
            if sb["完了"][0] < bd["完了"][0] - 0.03:
                continue
            m = ddb.P[..., 1] - ddb.P[..., 0]
            ok = []
            for dl in np.round(np.arange(0.0, 1.0, 0.02), 2):
                v = ddb.point(persist(m >= dl, 3)); s = scores(labd, v[4])
                if s["継続"][0] >= sb["継続"][0] - TOL and s["完了"][1] >= sb["完了"][1] - TOL:
                    ok.append((v[2], dl))
            if ok:
                best = (b, min(ok)[1])
        if best:
            b, dl = best
            deb = Data(biased(Fe, b), labe, Le, grid, 0.25)
            m = deb.P[..., 1] - deb.P[..., 0]
            v = deb.point(persist(m >= dl, 3))
            row(f"P(継続)×{b:g}，δ={dl:.2f}", scores(labe, v[4]), be_pt[2] - v[2])
            v2 = deb.point(persist(m >= dl, 3))
            row("  （同じ規則を新しいラベルで評価）", scores(labe_new, v2[4]), be_pt[2] - v2[2])


if __name__ == "__main__":
    main()
