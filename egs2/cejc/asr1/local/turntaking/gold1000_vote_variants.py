#!/usr/bin/env python
"""多数決の票を増やす・安定させると人手正解 922 件の一致率が上がるかを見る（2026-10-12）。

今の多数決（(e) 版）：票 1 元の 8B 思考なし ＋ 票 2 32B (e) ＋ 票 3 8B 思考あり (e)（全件付け直しの票）。
silver13 の ~/relabel_llm/run_prompt_check5.sh で、8B 思考あり (e) を gold だけあと 2 回、
4・5 票目の候補を (e) 版・思考なしで判定した。run_prompt_check6.sh で別系統の LLM（思考あり含む）も足した
（どちらも relabel/prompt_check/out_v/）。
(e) は同じ 922 件を見て作ったので、ここでの差は良く見えやすい。

  python local/turntaking/gold1000_vote_variants.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote_tags import read_votes                                     # noqa: E402

R = "/autofs/diamond5/share/users/kobori/relabel/"
V = R + "prompt_check/out_v/"


def main():
    rows = [json.loads(l) for l in open("tt_analysis/gold1000_prompts.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["route"] == "llm"]
    ln = [r["line_no"] for r in rows]
    y = np.array([r["gold"] for r in rows])

    def g(p):
        v = read_votes(p)
        return np.array([v[l] for l in ln]) if all(l in v for l in ln) else None

    q = np.array([r["qwen"] for r in rows])
    e32 = g(R + "full_e/Qwen3-32B-AWQ-nothink.add_e.tsv")
    think = {"r0（gold 確認）": g(R + "prompt_check/out/Qwen3-8B-think.add_e.tsv"),
             "r1（全件版）": g(R + "full_e/Qwen3-8B-think.add_e.tsv"),
             "r2": g(V + "Qwen3-8B-think.add_e.r2.tsv"),
             "r3": g(V + "Qwen3-8B-think.add_e.r3.tsv")}
    e8 = think["r1（全件版）"]
    cand = {}
    # out_v にある (e) 版の結果を全部（8B 思考ありの再実行 r2・r3 は除く。922 件そろったものだけ）
    for f in sorted(Path(V).glob("*.add_e.tsv")):
        nm = f.name[:-len(".add_e.tsv")]
        if nm.startswith("Qwen3-8B-think"):
            continue
        p = g(str(f))
        if p is not None:
            cand[nm] = p

    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(2000)]

    def f1(p, c):
        tp = np.sum((y == c) & (p == c))
        return 2 * tp / max(np.sum(y == c) + np.sum(p == c), 1)

    base = ((q + e32 + e8) >= 2).astype(int)

    def show(nm, p, ref=True):
        s = (f"{nm:44s}一致率 {np.mean(p == y):.3f}  継続F1 {f1(p, 0):.3f}  完了F1 {f1(p, 1):.3f}  "
             f"継続再現率 {np.mean(p[y == 0] == 0):.3f}  完了率 {p.mean():.2f}")
        if ref:
            d = [np.mean(p[i] == y[i]) - np.mean(base[i] == y[i]) for i in idx]
            s += f"  今の多数決との差 {np.mean(p == y) - np.mean(base == y):+.3f} " \
                 f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
        print(s)

    print(f"人手正解 {len(y)} 件（人手の完了率 {y.mean():.2f}）")
    print("\n単体")
    for nm, p in [("8B 思考なし 元（票 1）", q), ("32B (e)（票 2）", e32)] + \
            [(f"8B 思考あり (e) {k}", v) for k, v in think.items()] + list(cand.items()):
        show(nm, p, ref=False)

    print("\n8B 思考あり (e) の回どうしの一致")
    ks = list(think)
    for a, b in itertools.combinations(ks, 2):
        print(f"  {a} × {b}: {np.mean(think[a] == think[b]):.3f}")

    print("\n多数決")
    show("今の多数決（票 1・32B (e)・8B 思考あり r1）", base)
    t4 = sum(think.values())
    for th in (2, 3):
        t = (t4 >= th).astype(int)
        show(f"票 3 = 8B 思考あり 4 回で {th} 回以上「はい」", ((q + e32 + t) >= 2).astype(int))
    # 票 3 を回ごとに変えたときのばらつき
    accs = [np.mean(((q + e32 + v) >= 2) == y) for v in think.values()]
    print(f"  （票 3 をどの回にするかで一致率は {min(accs):.3f}〜{max(accs):.3f}）")
    for a, b in itertools.combinations(cand, 2):
        show(f"5 票（＋{a.replace('-nothink', '')}・{b.replace('-nothink', '')}）",
             ((q + e32 + e8 + cand[a] + cand[b]) >= 3).astype(int))
    for a in cand:
        for nm, vs in ((f"票 1 を {a} に", (cand[a], e32, e8)), (f"票 2 を {a} に", (q, cand[a], e8)),
                       (f"票 3 を {a} に", (q, e32, cand[a]))):
            show(nm, (sum(vs) >= 2).astype(int))


if __name__ == "__main__":
    main()
