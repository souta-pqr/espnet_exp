#!/usr/bin/env python
"""silver13 の run_check_f.sh の結果をまとめる（2026-10-16）。

  1. 人手正解 1000 件：LLM が判定していた 922 件と，規則で「完了」にしていた 78 件に分けて一致率を出す。
     規則の 78 件は，今の規則（全部「完了」）と比べる。
  2. 意味で確かめた 270 件：Claude の判定（tt_analysis/endcheck_20261016/judged_270.json）との一致率。
  (f) 版 − (e) 版の差は，同じ区間の対でブートストラップ（2000 回）した 95% 区間を付ける。

  python local/turntaking/compare_route_check.py
"""
import json
import random

O = "/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_route/"
N = "gemma-4-31B-it-think.{}.{}.tsv"


def read(p):
    v = {}
    try:
        for l in open(p, encoding="utf-8"):
            q = l.rstrip("\n").split("\t")
            if len(q) >= 2 and q[0].isdigit() and q[1] in ("はい", "いいえ"):
                v[int(q[0])] = int(q[1] == "はい")
    except FileNotFoundError:
        print(f"（まだ無い: {p}）")
    return v


def diff_ci(keys, ref, a, b, n=2000):
    rng = random.Random(0)
    d = []
    for _ in range(n):
        s = [rng.choice(keys) for _ in keys]
        d.append(sum((a[k] == ref[k]) - (b[k] == ref[k]) for k in s) / len(s))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)]


def report(title, ref, groups, e, f, rule=None):
    print(f"## {title}")
    print("区分\t件数\t(e) 版\t(f) 版\t差 (f)−(e) [95% 区間]" + ("\t今の規則（全部完了）" if rule else ""))
    for nm, keys in groups:
        keys = [k for k in keys if k in e and k in f]
        if not keys:
            print(f"{nm}\t0"); continue
        ae = sum(e[k] == ref[k] for k in keys) / len(keys)
        af = sum(f[k] == ref[k] for k in keys) / len(keys)
        lo, hi = diff_ci(keys, ref, f, e)
        line = f"{nm}\t{len(keys)}\t{ae:.3f}\t{af:.3f}\t{af - ae:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if rule:
            line += f"\t{sum(ref[k] == 1 for k in keys) / len(keys):.3f}"
        print(line)
    print()


def main():
    rows = [json.loads(l) for l in open("tt_analysis/gold1000_prompts.jsonl", encoding="utf-8")]
    gold = {r["line_no"]: int(r["gold"]) for r in rows}
    llm = [r["line_no"] for r in rows if r["route"] == "llm"]
    nd = [r["line_no"] for r in rows if r["route"] == "rule_nextdiff"]
    e, f = read(O + N.format("add_e", "gold1000")), read(O + N.format("add_f", "gold1000"))
    report("人手正解 1000 件", gold, [("LLM が判定していた区間", llm), ("規則で完了にしていた区間", nd),
                                    ("合計", llm + nd)], e, f, rule=True)
    J = json.load(open("tt_analysis/endcheck_20261016/judged_270.json", encoding="utf-8"))
    me = {x["line_no"]: int(x["me"]) for x in J}
    e, f = read(O + N.format("add_e", "check270")), read(O + N.format("add_f", "check270"))
    cats = sorted({x["cat"] for x in J})
    report("意味で確かめた 270 件（Claude の判定と比べる）", me,
           [(c, [x["line_no"] for x in J if x["cat"] == c]) for c in cats] + [("合計", list(me))], e, f)


if __name__ == "__main__":
    main()
