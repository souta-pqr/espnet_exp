"""動作点の選び方（全方式に同じものを使う）：cut-in rate ≤ 8%・miss rate ≤ 10% の点のうち，
latency の中央値が最小の点から 1 フレーム（silero の 32 ms）以内の点に絞り，その中で「相槌の後に応答」が最小，
同じなら latency の 90% 点が最小の点を選ぶ．中央値の 32 ms 未満の差は判定の刻みより細かく，意味がないため．"""
import csv, math, sys
TOL = 0.032
f = lambda x: math.inf if x == "nan" else float(x)
def pick(paths, cut=0.08, miss=0.10):
    r = []
    for p in paths:
        r += [dict(x, src=p.split("/")[-1]) for x in csv.DictReader(open(p), delimiter="\t")
              if float(x["cutin"]) <= cut + 1e-12 and float(x["noresp_free"]) <= miss + 1e-12]
    if not r: return None
    m = min(f(x["lat_p50_free"]) for x in r)
    q = [x for x in r if f(x["lat_p50_free"]) <= m + TOL]
    return min(q, key=lambda x: (float(x["bc_resp"]), f(x["lat_p90_free"]), f(x["lat_p50_free"])))
if __name__ == "__main__":
    D = "tt_analysis/newlabel_20261021_c8/dev"; L = "tt_analysis/longlat_20261024"
    src = {"silero": [f"{D}/curve_silero.tsv"]}
    for n in ["vap", "smart", "g", "gm"]:
        src[n + " 1段"] = [f"{D}/curve_{n}+silero+fb.tsv", f"{D}/curve_{n}+silero.tsv"]
    for n in ["vap", "smart", "gm"]:
        src[n + " 2段"] = src[n + " 1段"] + [f"{L}/curve_2stage_{n}.tsv"]
    for nm, ps in src.items():
        x = pick(ps)
        print(nm, "\t", " ".join(f"{k} {x[k]}" for k in ["cutin", "lat_p50_free", "lat_p90_free", "noresp_free", "bc_resp"]), "|", x["point"], f"[{x['src']}]")
