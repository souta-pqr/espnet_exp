#!/usr/bin/env python
"""予算を決めて比べる（2026-10-28）。LiveKit eot-bench の見せ方に合わせる。
  - 遮りの予算（cut-in rate ≤ 8%・10%）：その中で latency の中央値が最小の点
  - 遅れの予算（latency の中央値 ≤ 0.5 秒・1.0 秒）：その中で cut-in rate が最小の点
  - 相槌：cut-in rate ≤ 8% の点での「相槌の後に応答」と，「相槌の後に応答 ≤ 30%」も足したときの latency の中央値
  - どの点も miss rate ≤ 10%（応答しないことで遮りを減らすのを防ぐ．無いと選ばれる点の miss rate が 18〜24% になった）
点は開発データで選び（候補は eval_all.py と同じ：1 段・1 段＋長い無音・2 段の規則．silero-vad のみは無音の長さだけ），評価データで測る。
latency の中央値は次の発話まで 2 秒以上ある完了区間で数え，応答なしはとても遅い（∞）として含める（eval_all.py と同じ）。
曲線：開発データでの cut-in rate と latency の中央値のパレート最適な点を，評価データで測ったもの → budget_curve.tsv
  python tt_analysis/longlat_20261024/budget.py → budget.txt・budget_curve.tsv
  python tt_analysis/longlat_20261024/budget.py gm abnp abnl → budget_gm_abnp_abnl.txt（方式を指定．曲線は作らない．差は gm − 各方式）
"""
import csv, math, os, re, sys
import numpy as np
sys.path.insert(0, "local/turntaking")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eot_eval as E
from eval_all import SYS, D, L, W, FREE, parse

METHODS = ["silero", "smart", "vap", "smartft", "vapft", "gm", "gt"]   # smartft・vapft：CEJC で学習し直したもの（10/29）
CUT, LAT, BC, MISS = (0.08, 0.10), (0.5, 1.0), 0.30, 0.10
ARGS = sys.argv[1:]
if ARGS: METHODS = ARGS
inf_nan = lambda x: math.inf if x == "nan" else float(x)


def rows(n):
    ps = [f"{D}/curve_silero.tsv"] if n == "silero" else \
        [f"{D}/curve_{n}+silero.tsv", f"{D}/curve_{n}+silero+fb.tsv", f"{L}/curve_2stage_{n}.tsv"]
    out = []
    for p in ps:
        for x in csv.DictReader(open(p), delimiter="\t"):
            if float(x["noresp_free"]) > MISS + 1e-12: continue
            out.append(dict(point=x["point"], cut=float(x["cutin"]), lat=inf_nan(x["lat_p50_free"]),
                            miss=float(x["noresp_free"]), bc=float(x["bc_resp"])))
    return out


def best(rs, key, cond):
    ok = [r for r in rs if cond(r)]
    return min(ok, key=key) if ok else None


def front(rs):
    """cut-in rate が小さい順に，latency の中央値がそれまでより短くなる点だけ残す。"""
    out, m = [], math.inf
    for r in sorted(rs, key=lambda r: (r["cut"], r["lat"])):
        if r["lat"] < m - 1e-9:
            out.append(r); m = r["lat"]
    return out


def main():
    seg, gap, tag = E.load_segments("data/eval_g")
    dumps = {n: E.load_dump(SYS[n][0]) for n in METHODS if n != "silero"}
    utts = sorted(u for u in set(tag) & set(seg) if all(u in d for d in dumps.values()))
    runs = E.silence_runs("tt_analysis/vad_silero/eval_g", utts, seg, W)
    dur = np.array([seg[u][2] - seg[u][1] for u in utts]); g = np.array([gap[u] for u in utts])
    hor = dur + np.minimum(g, W); lab = np.array([tag[u] for u in utts]); free = g >= FREE
    conv = np.array([re.sub(r"[a-z]$", "", seg[u][0].rsplit("_", 1)[0]) for u in utts])
    members = [np.where(conv == c)[0] for c in sorted(set(conv))]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) for _ in range(1000)]
    allidx = np.arange(len(utts))
    st = lambda t, i: E.point_stats(t, i, dur, hor, lab, free, W)
    T0, sil_names = E.silero_points(runs, utts, 3.0)
    gcache, fcache = {}, {}

    def times(n, p):
        if n == "silero":
            s = float(re.match(r"無音([\d.]+)s", p)[1])
            return T0[:, [abs(float(x[2:-1]) - s) < 1e-6 for x in sil_names].index(True)]
        g1, t1, g2, t2, S, phi = parse(p)
        d, la = dumps[n], SYS[n][1]

        def gate(gg, th):
            if (n, gg) not in gcache:
                gcache[(n, gg)] = E.gated_points(d, runs, utts, la, [gg])
            T, names = gcache[(n, gg)]
            return T[:, [j for j, x in enumerate(names) if abs(float(x.split("θ=")[1]) - th) < 1e-6][0]]
        t = np.minimum(gate(g1, t1), gate(g2, t2))
        if S is not None:
            if (n, S, phi) not in fcache:
                fcache[(n, S, phi)] = E.fallback_points(d, runs, utts, la, S, phi)
            t = np.minimum(t, fcache[(n, S, phi)])
        return t

    def ci(t, k):
        bs = [st(t, b)[k] for b in boots]
        return st(t, allidx)[k], np.quantile(bs, .025), np.quantile(bs, .975), bs

    picks = {}                       # (列, 方式) → (点, 時刻)
    for n in METHODS:
        rs = rows(n)
        for c in CUT:
            r = best(rs, lambda r: (r["lat"], r["miss"]), lambda r: r["cut"] <= c + 1e-12)
            picks[(f"cut{c}", n)] = r
        for l in LAT:
            r = best(rs, lambda r: (r["cut"], r["lat"]), lambda r: r["lat"] <= l + 1e-12)
            picks[(f"lat{l}", n)] = r
        r = best(rs, lambda r: (r["lat"], r["miss"]), lambda r: r["cut"] <= CUT[0] + 1e-12 and r["bc"] <= BC + 1e-12)
        picks[("bc", n)] = r

    cols = [(f"lat{l}", "cutin", f"latency の中央値 ≤ {l} 秒での cut-in rate") for l in LAT] + \
           [(f"cut{c}", "p50_free", f"cut-in rate ≤ {c:.0%} での latency の中央値") for c in CUT] + \
           [(f"cut{CUT[0]}", "bc_resp", f"cut-in rate ≤ {CUT[0]:.0%} での相槌の後に応答"),
            ("bc", "p50_free", f"cut-in rate ≤ {CUT[0]:.0%}・相槌の後に応答 ≤ {BC:.0%} での latency の中央値")]
    lines = [f"区間 {len(utts)}．点は開発データで選び，評価データで測る．[ ] は会話単位のブートストラップ（{len(members)} 会話，1000 回）の 95% 区間．"]
    est, bss = {}, {}
    for col, k, title in cols:
        lines.append(f"\n## {title}")
        for n in METHODS:
            r = picks[(col, n)]
            if r is None:
                lines.append(f"{n}\t届かない（開発データ）"); continue
            t = times(n, r["point"])
            e, lo, hi, bs = ci(t, k)
            est[(col, k, n)], bss[(col, k, n)] = e, bs
            lines.append(f"{n}\t{E.fmt(e)} [{E.fmt(lo)}, {E.fmt(hi)}]\t評価 miss {st(t, allidx)['noresp_free']:.3f}\t開発 cut-in {r['cut']:.3f}・中央値 {E.fmt(r['lat'])}・相槌 {r['bc']:.3f}\t{r['point']}")
        if ARGS:
            for n in METHODS[1:]:
                if (col, k, n) in est and (col, k, METHODS[0]) in est:
                    d = np.array(bss[(col, k, n)]) - np.array(bss[(col, k, METHODS[0])])
                    lines.append(f"差 {n} − {METHODS[0]}\t{est[(col, k, n)] - est[(col, k, METHODS[0])]:+.3f} [{np.nanquantile(d, .025):+.3f}, {np.nanquantile(d, .975):+.3f}]")
            continue
        others = [n for n in METHODS if n not in ("gm", "gt") and (col, k, n) in est]
        if (col, k, "gt") in est and others:
            nb = min(others, key=lambda n: est[(col, k, n)])          # 従来手法で一番良いもの
            d = np.array(bss[(col, k, "gt")]) - np.array(bss[(col, k, nb)])
            lines.append(f"差 gt − {nb}\t{est[(col, k, 'gt')] - est[(col, k, nb)]:+.3f} [{np.nanquantile(d, .025):+.3f}, {np.nanquantile(d, .975):+.3f}]")
    txt = "\n".join(lines); print(txt, flush=True)
    if ARGS:
        open(f"{L}/budget_{'_'.join(METHODS)}.txt", "w").write(txt + "\n"); return
    open(f"{L}/budget.txt", "w").write(txt + "\n")

    with open(f"{L}/budget_curve.tsv", "w") as f:
        f.write("method\tdev_cutin\tdev_p50\teval_cutin\teval_p50\tpoint\n")
        for n in METHODS:
            fr = [r for r in front(rows(n)) if 0.02 <= r["cut"] <= 0.30]
            fr = [r for i, r in enumerate(fr) if i % max(1, len(fr) // 40) == 0]     # 1 方式 40 点ほど
            for r in fr:
                s = st(times(n, r["point"]), allidx)
                f.write(f"{n}\t{r['cut']:.4f}\t{r['lat']:.4f}\t{s['cutin']:.4f}\t{s['p50_free']:.4f}\t{r['point']}\n")
            print(f"曲線 {n}：{len(fr)} 点", flush=True)


if __name__ == "__main__":
    main()
