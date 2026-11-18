#!/usr/bin/env python
"""全方式に同じ選び方をして，評価データで比べる（2026-10-24）。
  候補：1 段の規則（eot_eval.py の +silero・+silero+fb）と 2 段の規則（two_stage.py）．silero-vad のみは無音の長さだけ．
  選び方（開発データ，全方式で同じ）：
    A：cut-in rate ≤ 8%・miss rate ≤ 10% の点のうち latency の中央値が最小
    B：A の条件に「相槌の後に応答 ≤ BC」を足す（届かない方式は「届かない」）
  評価データ：会話単位のブートストラップ（48 会話，1000 回，全方式で同じ標本）で 95% 区間と，基準の方式との差。
  python eval_all.py <基準の方式> [足す方式 ...] → pick_all.txt（基準が gm 以外なら pick_all_<基準>.txt）
    足す方式：gt・アブレーション（abnp・abnt・abna）など．基準も含めて SYS にあるもの（2026-10-26）
"""
import csv, math, re, sys
import numpy as np
sys.path.insert(0, "local/turntaking")
import eot_eval as E

W, FREE, BC = 5.0, 2.0, 0.30
D, L = "tt_analysis/newlabel_20261021_c8/dev", "tt_analysis/longlat_20261024"
SYS = {  # 名前：(ダンプ, 先読み)
    "gm": ("tt_preds/frame_attn-frame2-blk42xh2-gm_same_n5_eval_g_valid.loss.ave_post5.0.tsv", 0.2),
    "g": ("tt_preds/frame_attn-frame2-blk42xh2-g_same_n5_eval_g_valid.loss.ave_post5.0.tsv", 0.2),
    "vap": ("tt_preds/vap_jp_eval_g_post5.0.tsv", 0.0),
    "smart": ("tt_preds/smart_turn_eval_g_post5.0.tsv", 0.0),
    "smartft": ("tt_preds/smart_turn_ft_eval_g_post5.0.tsv", 0.0),   # CEJC で学習し直したもの（10/29）
    "vapft": ("tt_preds/vap_ft_eval_g_post5.0.tsv", 0.0),
    **{n: (f"tt_preds/frame_attn-frame2-blk42xh2-{n}_same_n5_eval_g_valid.loss.ave_post5.0.tsv", 0.2)
       for n in ("gt", "abnp", "abnt", "abna", "abnl")},
}
inf_nan = lambda x: math.inf if x == "nan" else float(x)


def curves(n):
    ps = [f"{D}/curve_silero.tsv"] if n == "silero" else [f"{D}/curve_{n}+silero.tsv", f"{D}/curve_{n}+silero+fb.tsv"]
    import os
    if os.path.exists(f"{L}/curve_2stage_{n}.tsv"): ps.append(f"{L}/curve_2stage_{n}.tsv")
    rows = []
    for p in ps:
        rows += [x for x in csv.DictReader(open(p), delimiter="\t")
                 if float(x["cutin"]) <= 0.08 + 1e-12 and float(x["noresp_free"]) <= 0.10 + 1e-12]
    return rows


def pick(n, bc):
    r = [x for x in curves(n) if float(x["bc_resp"]) <= bc + 1e-12]
    return min(r, key=lambda x: (inf_nan(x["lat_p50_free"]), float(x["noresp_free"]))) if r else None


def parse(p):
    """動作点の文字列 → (g1, θ1, g2, θ2, S, φ)．S = None は長い無音での応答なし。"""
    m = re.match(r"無音([\d.]+)s θ1=([\d.]+) 無音([\d.]+)s θ2=([\d.]+) 無音([\d.]+)s・相槌<([\d.]+)$", p)
    if m: return tuple(map(float, m.groups()))
    m = re.match(r"無音([\d.]+)s θ=([\d.]+)(?: 無音([\d.]+)s・相槌<([\d.]+)で応答)?$", p)
    g1, t1 = float(m[1]), float(m[2])
    return (g1, t1, g1, t1, float(m[3]) if m[3] else None, float(m[4]) if m[4] else None)


def main():
    base = sys.argv[1]
    methods = ["silero", "smart", "vap", "g", "gm"] + [n for n in [base] + sys.argv[2:] if n not in ("silero", "smart", "vap", "g", "gm")]
    seg, gap, tag = E.load_segments("data/eval_g")
    dumps = {n: E.load_dump(SYS[n][0]) for n in methods if n != "silero"}
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
    keys = ("cutin", "p50_free", "p90_free", "noresp_free", "bc_resp")

    def times(n, p):
        if n == "silero":
            s = float(re.match(r"無音([\d.]+)s", p)[1])
            T, names = E.silero_points(runs, utts, 3.0)
            return T[:, [abs(float(x[2:-1]) - s) < 1e-6 for x in names].index(True)]
        g1, t1, g2, t2, S, phi = parse(p)
        d, la = dumps[n], SYS[n][1]
        def gate(gg, th):
            T, names = E.gated_points(d, runs, utts, la, [gg])
            return T[:, [j for j, x in enumerate(names) if abs(float(x.split("θ=")[1]) - th) < 1e-6][0]]
        t = np.minimum(gate(g1, t1), gate(g2, t2))
        if S is not None:
            t = np.minimum(t, E.fallback_points(d, runs, utts, la, S, phi))
        return t

    lines = [f"区間 {len(utts)}．開発データで選んだ点を評価データで測る．95% 区間は会話単位のブートストラップ（{len(members)} 会話，1000 回）．",
             "選び方 A：cut-in rate ≤ 8%・miss rate ≤ 10% で latency の中央値が最小．B：A ＋ 相槌の後に応答 ≤ 30%．",
             "選び方\t方式\t動作点\tcut-in rate\tlatency 中央値\tlatency 90% 点\tmiss rate\t相槌の後に応答"]
    cols = {}
    for sel, bc in (("A", 1.0), ("B", BC)):
        for n in methods:
            r = pick(n, bc)
            if r is None:
                lines.append(f"{sel}\t{n}\t（届かない）"); continue
            t = times(n, r["point"]); cols[(sel, n)] = t
            est = st(t, allidx); bs = [st(t, b) for b in boots]
            cell = [f"{E.fmt(est[k])} [{E.fmt(np.quantile([x[k] for x in bs], .025))}, {E.fmt(np.quantile([x[k] for x in bs], .975))}]" for k in keys]
            lines.append(f"{sel}\t{n}\t{r['point']}\t" + "\t".join(cell))
    lines.append(f"差（{base} − 各方式）")
    for sel in "AB":
        if (sel, base) not in cols: continue
        for n in [m for m in methods if m != base] + ([base] if sel == "B" else []):
            s2 = sel if (sel, n) in cols and n != base else "A"     # B に届かない方式は A の点と比べる
            if (s2, n) not in cols or (s2, n) == (sel, base): continue
            t, x = cols[(s2, n)], cols[(sel, base)]
            e = {k: st(x, allidx)[k] - st(t, allidx)[k] for k in keys}
            d = {k: [st(x, b)[k] - st(t, b)[k] for b in boots] for k in keys}
            lines.append(f"{sel}{base} − {s2}{n}\t\t\t" + "\t".join(f"{e[k]:+.3f} [{np.nanquantile(d[k], .025):+.3f}, {np.nanquantile(d[k], .975):+.3f}]" for k in keys))
    txt = "\n".join(lines); print(txt)
    open(f"{L}/pick_all.txt" if base == "gm" else f"{L}/pick_all_{base}.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
