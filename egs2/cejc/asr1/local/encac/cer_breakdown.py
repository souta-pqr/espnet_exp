"""CER の内訳（2026-11-27）．置換・脱落・挿入，区間の長さ別，「＊」（聞き取れない部分）の影響．
  python cer_breakdown.py <cer_aux.py の出力 tsv> [<元の transducer の認識結果（ESPnet の text 形式）>]
"""
import sys, os
sys.path.insert(0, os.getcwd())
from train_et_data import norm
from collections import defaultdict


def sdi(r, h):
    n, m = len(r), len(h)
    D = [[(j, 0, 0, j) for j in range(m + 1)]]          # (コスト, 置換, 脱落, 挿入)
    for i in range(1, n + 1):
        row = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            a = D[i - 1][j - 1]; a = (a[0] + (r[i - 1] != h[j - 1]), a[1] + (r[i - 1] != h[j - 1]), a[2], a[3])
            b = D[i - 1][j]; b = (b[0] + 1, b[1], b[2] + 1, b[3])
            c = row[j - 1]; c = (c[0] + 1, c[1], c[2], c[3] + 1)
            row.append(min(a, b, c))
        D.append(row)
    return D[n][m][1:]


rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
rows = [(u, r, x[0] if x else "") for u, r, *x in rows]
systems = {"補助タスクの CTC": {u: h for u, r, h in rows}}
if len(sys.argv) > 2:
    systems["元の transducer"] = {(l.split(" ", 1) + [""])[0]: norm((l.rstrip("\n").split(" ", 1) + [""])[1]) for l in open(sys.argv[2])}
ref = {u: r for u, r, h in rows}
BINS = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9))
for name, hyp in systems.items():
    tot = [0, 0, 0]; N = 0; byb = defaultdict(lambda: [0, 0, 0]); star = [0, 0]; nostar = [0, 0]; empty = 0
    for u, r in ref.items():
        h = hyp.get(u, "")
        s, d, i = sdi(r, h)
        tot[0] += s; tot[1] += d; tot[2] += i; N += len(r)
        empty += (h == "")
        for lo, hi in BINS:
            if lo <= len(r) <= hi:
                byb[(lo, hi)][0] += s + d + i; byb[(lo, hi)][1] += len(r); byb[(lo, hi)][2] += 1
        r2 = r.replace("＊", "")
        e2 = sum(sdi(r2, h)); nostar[0] += e2; nostar[1] += len(r2)
        if "＊" in r: star[0] += 1
    E = sum(tot)
    print(f"== {name}：CER {100 * E / N:.1f}%（置換 {100 * tot[0] / N:.1f}・脱落 {100 * tot[1] / N:.1f}・挿入 {100 * tot[2] / N:.1f}）  出力が空の区間 {empty}")
    print(f"   正解から「＊」を消すと CER {100 * nostar[0] / nostar[1]:.1f}%（「＊」を含む区間 {star[0]}）")
    for (lo, hi), (e, n, c) in byb.items():
        print(f"   正解の長さ {lo}〜{hi if hi < 10**9 else ''} 文字：{c} 区間・文字の {100 * n / N:.1f}%・CER {100 * e / n:.1f}%・誤りの {100 * e / E:.1f}%")
