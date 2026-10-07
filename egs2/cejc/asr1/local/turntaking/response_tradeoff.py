#!/usr/bin/env python
"""応答ミス ／ 誤割り込み ／ 応答遅延 の交換比で停止規則とモデルを比べる。

macro-F1 は 3 クラスを対称に扱うが、このタスクの動機は対称ではない。
中間審査資料が挙げる課題は

  応答ミス   … 言い終えたのに <完了> と判定せず、システムが応答しない
  応答遅延   … 言い終えてから確定するまでの時間

で、<継続> の判定が問題になるのは「誤って <完了> と言って相手を遮る」ときだけ。
そこで本スクリプトは macro-F1 ではなく次の 3 つで評価する。

  応答ミス率   = P(pred != 完了 | 真 = 完了)
  誤割り込み率 = P(pred == 完了 | 真 != 完了)
  完了の遅延   = 真に完了だった区間での（確定時刻 − 発話末）

応答ミス率と誤割り込み率は一方を下げれば他方が上がるので、
**誤割り込み率を揃えた上で応答ミス率と遅延を比べる**のが公平な比較になる。
どの規則も「区間検出が発話末 + max_wait 秒で発火する」制約の下で測る。

  python local/turntaking/response_tradeoff.py --stems pool_attn-trunc03_same_n5 ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anticipate_commit import cap_index, load_forward, utt_lengths   # noqa: E402

GRID9 = "0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0"
GRID17 = ("0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 "
          "1.75 2.0 2.25 2.5 2.75 3.0")


def operating_point(first, pred_override, P, lab, ev, cap):
    """規則の動作点 → (応答ミス率, 誤割り込み率, 完了の遅延, 全体の遅延)。"""
    S = len(ev)
    fired = first < S
    idx = np.where(fired, np.minimum(first, cap), cap)
    pr = P.argmax(-1)[idx, np.arange(len(lab))]
    if pred_override is not None:
        pr = np.where(fired & (first <= cap), pred_override, pr)
    lat = ev[idx]
    m1 = lab == 1
    return ((pr[m1] != 1).mean(),                       # 応答ミス
            ((pr == 1) & ~m1).sum() / max((~m1).sum(), 1),   # 誤割り込み
            lat[m1], lat)


def all_rules(P, H, lab, ev, cap, hn):
    """規則名 → 動作点。閾値は細かめに掃く。"""
    S = len(ev)
    out = {}
    mx = P.max(-1)
    taus = np.round(np.arange(0.30, 1.00, 0.02), 2)

    def add(name, ok, override):
        first = np.where(ok.any(0), ok.argmax(0), S)
        out[name] = operating_point(first, override, P, lab, ev, cap)

    out["区間検出まで待つ"] = operating_point(
        np.full(len(lab), S), None, P, lab, ev, cap)
    ones = np.ones(len(lab), dtype=int)
    for t in taus:
        add(f"単一τ={t:.2f}", mx >= t, None)
        add(f"完了τ={t:.2f}", P[..., 1] >= t, ones)
    for g in [0.5, 0.75, 1.0, 1.25]:
        for t in taus:
            add(f"単一τ={t:.2f}+ゲート{g}", (mx >= t) & (ev[:, None] >= g), None)
            add(f"完了τ={t:.2f}+ゲート{g}", (P[..., 1] >= t) & (ev[:, None] >= g), ones)
    for k, h in enumerate(hn):
        for t in taus:
            add(f"先読みh={h:g} θ={t:.2f}", H[..., k] >= t, ones)
    return out


def best_at(rules, target_fa, tol=0.005, key="miss"):
    """誤割り込み率が target_fa 以下の規則のうち、応答ミス（または遅延）が最小のもの。"""
    cand = [(nm, v) for nm, v in rules.items() if v[1] <= target_fa + tol]
    if not cand:
        return None
    if key == "miss":
        return min(cand, key=lambda x: x[1][0])
    return min(cand, key=lambda x: x[1][2].mean())


def fastest_no_worse(rules, base, tol=0.005):
    """応答ミスも誤割り込みも基準以下に保ったまま、完了の遅延が最小の規則。

    「どちらの失敗も悪化させずに、どれだけ早く応答できるか」という
    このタスクでいちばん知りたい問いに直接答える。
    """
    cand = [(nm, v) for nm, v in rules.items()
            if v[0] <= base[0] + tol and v[1] <= base[1] + tol]
    if not cand:
        return None
    return min(cand, key=lambda x: x[1][2].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--targets", default="0.20 0.25 0.30 0.35",
                    help="揃える誤割り込み率")
    args = ap.parse_args()

    targets = [float(t) for t in args.targets.split()]
    per_stem = {}
    for stem in args.stems:
        # 実在するダンプから格子を決める（モデル名で決め打ちすると取り違える）
        grid = [e for e in GRID17.split()
                if Path(f"tt_preds/{stem}_e{e}s_{args.ckpt}.tsv").exists()]
        if not grid:
            print(f"！ {stem}: 無音込みダンプが見つからない"); continue
        P, H, lab, utts, hn = load_forward(stem, args.ckpt, grid, sil=True)
        L = utt_lengths(utts, "eval")
        ev = np.array([float(e) for e in grid])
        cap = cap_index(ev, L, args.max_wait)
        # 遅延は「確定時刻 − 発話末」に直す
        rules = all_rules(P, H, lab, ev, cap, hn)
        rules = {k: (v[0], v[1], v[2] - L[lab == 1], v[3] - L) for k, v in rules.items()}
        per_stem[stem] = rules
        base = rules["区間検出まで待つ"]
        print(f"\n■ {stem}  （eval {len(lab)} 区間 / 区間検出 {args.max_wait} 秒）")
        print(f"  基準（区間検出まで待つ）: 応答ミス {base[0]:.3f} / "
              f"誤割り込み {base[1]:.3f} / 完了の遅延 {base[2].mean():+.3f} 秒")

    print(f"\n{'='*78}\n"
          "どちらの失敗も基準（区間検出まで待つ）より悪化させずに、最も早く応答できる規則\n"
          f"{'='*78}")
    print(f"{'モデル':>26} {'最良の規則':>22}{'応答ミス':>9}{'誤割り込み':>10}"
          f"{'完了の遅延':>11}{'短縮':>9}")
    for stem, rules in per_stem.items():
        base = rules["区間検出まで待つ"]
        r = fastest_no_worse(rules, base)
        if r is None:
            print(f"{stem:>26} {'（基準を超えられない）':>22}")
            continue
        nm, v = r
        print(f"{stem:>26} {nm:>22}{v[0]:9.3f}{v[1]:10.3f}{v[2].mean():+11.3f}"
              f"{base[2].mean() - v[2].mean():9.3f}")

    for key, lbl in [("miss", "応答ミス率を最小化"), ("lat", "完了の遅延を最小化（参考・下の注記）")]:
        print(f"\n{'='*78}\n誤割り込み率を揃えて {lbl}\n{'='*78}")
        print(f"{'誤割り込み':>8} {'モデル':>26} {'最良の規則':>22}"
              f"{'応答ミス':>9}{'完了の遅延':>11}")
        for fa in targets:
            for stem, rules in per_stem.items():
                r = best_at(rules, fa, key=key)
                if r is None:
                    print(f"{fa:8.2f} {stem:>26} {'（到達できない）':>22}")
                    continue
                nm, v = r
                print(f"{fa:8.2f} {stem:>26} {nm:>22}{v[0]:9.3f}{v[2].mean():+11.3f}")
            print()


if __name__ == "__main__":
    main()
