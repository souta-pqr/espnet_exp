#!/usr/bin/env python
"""「完了クラスだけに閾値」の τ を dev（train_dev）で選び直し、eval で報告する。

response_tradeoff.py は eval だけを読み、eval 上で
「応答ミス・誤割り込みとも基準 +0.005 以内で完了の遅延が最小」の規則を選んでいた
（資料 report_20260928 6 節の τ=0.76/0.78/0.74）。評価の作法（閾値は dev だけで選ぶ）に
合わせ、同じ基準で dev 上で τ を選び、その τ を eval に当てる。
あわせて、選んだ τ の行の macro-F1 と、従来の推奨（τ=0.95＋ゲート1.0）を同じ表に並べる。

  python local/turntaking/stoprule_devselect.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anticipate_commit import cap_index, load_forward, utt_lengths   # noqa: E402
from early_commit import prf                                        # noqa: E402

GRID = "0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0".split()
STEMS = [("30", "pool_attn-trunc-new_same_n5"),
         ("1", "pool_attn-trunc-new-seed1_same_n5"),
         ("2", "pool_attn-trunc-new-seed2_same_n5")]
TAUS = np.round(np.arange(0.30, 1.00, 0.02), 2)
TOL = 0.005


def load(stem, dev):
    P, _, lab, utts, _ = load_forward(stem, "valid.loss.ave", GRID, dev=dev, sil=True)
    L = utt_lengths(utts, "train_dev" if dev else "eval")
    ev = np.array([float(e) for e in GRID])
    return P, lab, L, ev, cap_index(ev, L, 0.25)


def run(P, lab, L, ev, cap, ok, override):
    """規則 → (応答ミス, 誤割り込み, 完了の遅延, macro-F1)。response_tradeoff と同じ定義。"""
    S = len(ev)
    first = np.where(ok.any(0), ok.argmax(0), S)
    fired = first < S
    idx = np.where(fired, np.minimum(first, cap), cap)
    j = np.arange(len(lab))
    pr = P.argmax(-1)[idx, j]
    if override:
        pr = np.where(fired & (first <= cap), 1, pr)
    lat = ev[idx] - L
    m1 = lab == 1
    return ((pr[m1] != 1).mean(), ((pr == 1) & ~m1).sum() / (~m1).sum(),
            lat[m1].mean(), prf(lab, pr)[1])


def rules(P, ev):
    S, N = P.shape[0], P.shape[1]
    out = {"基準": (np.zeros((S, N), bool), False)}
    for t in TAUS:
        out[f"完了τ={t:.2f}"] = (P[..., 1] >= t, True)
    out["従来 τ=0.95+ゲート1.0"] = ((P.max(-1) >= 0.95) & (ev[:, None] >= 1.0), False)
    return out


def pick(res):
    """基準 +TOL 以内で完了の遅延が最小の 完了τ。"""
    b = res["基準"]
    cand = [(k, v) for k, v in res.items() if k.startswith("完了τ")
            and v[0] <= b[0] + TOL and v[1] <= b[1] + TOL]
    return min(cand, key=lambda x: x[1][2])[0]


def main():
    print("無音込み・区間検出 0.25 秒・前向きグリッド 9 点。τ の選択基準：応答ミス・誤割り込みとも"
          f"基準 +{TOL} 以内で完了の遅延が最小\n")
    hdr = f"{'seed':>4} {'規則':<22}{'応答ミス':>8}{'誤割り込み':>9}{'完了の遅延':>10}{'早くなる量':>9}{'macroF1':>9}"
    summ = []
    for seed, stem in STEMS:
        Pd, labd, Ld, ev, capd = load(stem, True)
        dres = {k: run(Pd, labd, Ld, ev, capd, *v) for k, v in rules(Pd, ev).items()}
        Pe, labe, Le, ev, cape = load(stem, False)
        eres = {k: run(Pe, labe, Le, ev, cape, *v) for k, v in rules(Pe, ev).items()}
        tdev, teval = pick(dres), pick(eres)
        print(hdr)
        b = eres["基準"]
        for name, lab in [("基準", "基準（区間検出まで待つ）"),
                          (tdev, f"{tdev}（dev 選択）"),
                          (teval, f"{teval}（eval 選択・従来の表）"),
                          ("従来 τ=0.95+ゲート1.0", "従来の推奨 τ0.95+ゲート1.0")]:
            v = eres[name]
            sh = "—" if name == "基準" else f"{b[2] - v[2]:.3f}"
            print(f"{seed:>4} {lab:<22}{v[0]:8.3f}{v[1]:9.3f}{v[2]:+10.3f}{sh:>9}{v[3]:9.4f}")
        d = dres[tdev]; db = dres["基準"]
        print(f"     （dev 上: 基準 応答ミス {db[0]:.3f}/誤割り込み {db[1]:.3f} → {tdev} "
              f"{d[0]:.3f}/{d[1]:.3f}、早くなる量 {db[2] - d[2]:.3f} 秒）\n")
        summ.append((seed, tdev, b, eres[tdev]))
    sh = [b[2] - v[2] for _, _, b, v in summ]
    print(f"dev 選択の早くなる量: {' / '.join(f'{x:.3f}' for x in sh)} → 平均 {np.mean(sh):.3f}・SD {np.std(sh, ddof=1):.3f}")
    print("基準からの差（dev 選択）: 応答ミス "
          + " / ".join(f"{v[0] - b[0]:+.3f}" for _, _, b, v in summ)
          + "、誤割り込み " + " / ".join(f"{v[1] - b[1]:+.3f}" for _, _, b, v in summ))


if __name__ == "__main__":
    main()
