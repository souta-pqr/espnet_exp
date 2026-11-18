# long_lat.py の出力（dev_*.tsv）から，次の発話まで 2 秒以上の完了区間で遅れ 2 秒超の原因を数える（2026-10-26）
#   python cause.py dev_gm.tsv 0.8 0.75
import sys, csv
p, G, TH = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = [r for r in csv.DictReader(open(p), delimiter="\t") if float(r["次の発話まで"]) >= 2]
lat = lambda r: float(r["遅れ"])
n = len(rows); L = [r for r in rows if lat(r) > 2]
c = {"θ に届かない": 0, "θ に届くのが遅い": 0, f"無音 {G}s に達するのが遅い": 0, f"無音が {G}s 続かない": 0}
for r in L:
    ts = float(r["無音0.8sの時刻"])
    if ts == float("inf"): c[f"無音が {G}s 続かない"] += 1
    elif ts > 1.2: c[f"無音 {G}s に達するのが遅い"] += 1
    elif float(r["p完了の最大"]) < TH: c["θ に届かない"] += 1
    else: c["θ に届くのが遅い"] += 1
print(f"{p}：完了区間（次の発話まで 2 秒以上）{n}，遅れ 2 秒超 {len(L)}（{len(L)/n:.1%}）")
for k, v in c.items(): print(f"  {k} {v}（{v/n:.1%}）")
