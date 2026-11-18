#!/usr/bin/env python
"""無音が長くなるほど θ を下げる 2 段の規則（2026-10-24）。eot_eval.py の関数をそのまま使い，eot_eval.py は変えない。

  応答：(無音 ≥ g1 かつ p(完了) ≥ θ1) または (無音 ≥ g2 かつ p(完了) ≥ θ2) または (無音 ≥ S かつ p(相槌) < φ)
  θ2 < θ1，g2 > g1．θ2 を θ1 にすると今の規則（gm+silero+fb）と同じ。

  開発：python two_stage.py dev  <data> <dump> <out_dir>          → curve_2stage.tsv
  評価：python two_stage.py eval <data> <dump> <out_dir> "<動作点>" ...  → 会話単位ブートストラップの 95% 区間
"""
import sys
import numpy as np
sys.path.insert(0, "local/turntaking")
import eot_eval as E

import os
LA, W, FREE = float(os.environ.get("LA", 0.2)), 5.0, 2.0      # 先読み（提案法 0.2，VAP・Smart Turn は 0）
G1 = [0.6, 0.8, 1.0, 1.2, 1.5]
TH1 = list(E.THETAS)
G2 = [1.2, 1.5, 2.0, 2.5]
TH2 = [0.2, 0.3, 0.4, 0.5]
S_ = [2.0, 2.5, 3.0]
PHI = [0.1, 0.3, 0.5, 1.01]


def setup(data, dump_p):
    seg, gap, tag = E.load_segments(data)
    dump = E.load_dump(dump_p)
    utts = sorted(u for u in set(tag) & set(seg) if u in dump)
    runs = E.silence_runs(f"tt_analysis/vad_silero/{data.rstrip('/').split('/')[-1]}", utts, seg, W)
    return seg, gap, tag, dump, utts, runs


def name(g1, t1, g2, t2, S, phi):
    return f"無音{g1:g}s θ1={t1:.2f} 無音{g2:g}s θ2={t2:.2f} 無音{S:g}s・相槌<{phi:g}"


def main():
    mode, data, dump_p, out = sys.argv[1:5]
    seg, gap, tag, dump, utts, runs = setup(data, dump_p)
    G1x = sorted(set(G1 + G2))
    T1, n1 = E.gated_points(dump, runs, utts, LA, G1x)          # 列：gate × THETAS
    col = {(float(n.split("s")[0][2:]), round(float(n.split("θ=")[1]), 2)): j for j, n in enumerate(n1)}
    TF = {(S, phi): E.fallback_points(dump, runs, utts, LA, S, phi) for S in S_ for phi in PHI}
    if mode == "dev":
        with open(f"{out}/curve_2stage{os.environ.get('TAG', '')}.tsv", "w") as f:
            f.write("point\tcutin\tlat_p50_free\tlat_p90_free\tnoresp_free\tbc_resp\n")
            for g1 in G1:
                A = T1[:, [col[(g1, round(t, 2))] for t in TH1]]
                for g2 in G2:
                    if g2 <= g1: continue
                    for t2 in TH2:
                        B = T1[:, col[(g2, t2)]][:, None]
                        for (S, phi), Tf in TF.items():
                            T = np.minimum(np.minimum(A, B), Tf[:, None])
                            r = E.summarize(T, utts, seg, gap, tag, W, FREE)
                            for j, t1 in enumerate(TH1):
                                f.write(f"{name(g1, t1, g2, t2, S, phi)}\t{r['cutin'][j]:.4f}\t{r['p50_free'][j]:.4f}"
                                        f"\t{r['p90_free'][j]:.4f}\t{r['noresp_free'][j]:.4f}\t{r['bc_resp'][j]:.4f}\n")
        return
    # 評価：指定した動作点（name() の形，または「今の規則」の形）を会話単位ブートストラップで
    import re
    dur = np.array([seg[u][2] - seg[u][1] for u in utts]); g = np.array([gap[u] for u in utts])
    hor = dur + np.minimum(g, W); lab = np.array([tag[u] for u in utts]); free = g >= FREE
    conv = np.array([re.sub(r"[a-z]$", "", seg[u][0].rsplit("_", 1)[0]) for u in utts])
    members = [np.where(conv == c)[0] for c in sorted(set(conv))]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) for _ in range(1000)]
    allidx = np.arange(len(utts))
    keys = ("cutin", "p50_free", "p90_free", "noresp_free", "bc_resp")
    ts = {}
    for p in sys.argv[5:]:
        m = re.match(r"無音([\d.]+)s θ1=([\d.]+) 無音([\d.]+)s θ2=([\d.]+) 無音([\d.]+)s・相槌<([\d.]+)", p)
        g1, t1, g2, t2, S, phi = map(float, m.groups())
        ts[p] = np.minimum(np.minimum(T1[:, col[(g1, round(t1, 2))]], T1[:, col[(g2, round(t2, 2))]]), TF[(S, phi)])
    lines = ["動作点\t遮り率\t遅れ中央値\t遅れ90%点\t応答なし\t相槌に応答"]
    st = lambda t, i: E.point_stats(t, i, dur, hor, lab, free, W)
    for p, t in ts.items():
        est = st(t, allidx); bs = [st(t, b) for b in boots]
        cell = [f"{E.fmt(est[k])} [{E.fmt(np.quantile([x[k] for x in bs], .025))}, {E.fmt(np.quantile([x[k] for x in bs], .975))}]" for k in keys]
        lines.append(p + "\t" + "\t".join(cell))
    ps = list(ts)
    for p in ps[1:]:                      # 1 つ目との差
        d = {k: [st(ts[p], b)[k] - st(ts[ps[0]], b)[k] for b in boots] for k in keys}
        e = {k: st(ts[p], allidx)[k] - st(ts[ps[0]], allidx)[k] for k in keys}
        lines.append(f"差（{p} − 1 つ目）\t" + "\t".join(f"{e[k]:+.3f} [{np.quantile(d[k], .025):+.3f}, {np.quantile(d[k], .975):+.3f}]" for k in keys))
    txt = "\n".join(lines); print(txt)
    open(f"{out}/pick_eval{os.environ.get('TAG', '')}.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
