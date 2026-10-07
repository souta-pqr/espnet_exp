#!/usr/bin/env python
"""人手正解 1000 件で、Claude と Qwen3 のラベル付与精度と一致度を比べる（完了 / 継続の 2 値）。

どちらも同じ手順で付けている（local/turntaking/gold1000_prompts.py）:
  直後に別話者が話す区間はルールで「完了」（両者同じ）、それ以外を LLM が full_context/oneshot で判定。
Claude の判定は tt_analysis/gold1000_claude.tsv（line_no と はい/いいえ）。

標本は「完了・継続が半々」になるよう偏って抜かれているので、母集団（相槌を除く全区間）の
比率で重み付けした値（事後層別。層＝経路 × Qwen の判定）も出す。

  python local/turntaking/gold1000_compare.py
"""
import collections
import importlib.util
import json
import re

import numpy as np

PROMPTS = "tt_analysis/gold1000_prompts.jsonl"
CLAUDE = "tt_analysis/gold1000_claude.tsv"
RES = "/autofs/diamond5/share/users/kobori/result_oneshot.txt"
spec = importlib.util.spec_from_file_location(
    "det27", "/home/kobori/work/cejc-label/detect_section.text.py")
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)


def population_strata():
    """母集団（ルールで相槌にならない全区間）の層ごとの件数。層＝(経路, Qwen の判定)。"""
    LR = re.compile(r"^\[LINE (\d+)\]")
    UR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")
    cnt = collections.Counter(); sec = None; t = None; post = []
    for raw in open(RES, encoding="utf-8"):
        if raw.startswith("[LINE"):
            sec = None; t = None; post = []; continue
        if raw.startswith("[TARGET]"): sec = "t"; continue
        if raw.startswith("[POST_HISTORY]"): sec = "p"; continue
        if raw.startswith("[HISTORY]"): sec = "h"; continue
        if raw.startswith("[RESULT]"):
            q = int(raw.split()[1])
            if q == 2:
                continue
            route = "rule_nextdiff" if det.is_turn_end_next_different_speaker(t, post) else "llm"
            cnt[(route, q)] += 1
            continue
        if sec in ("t", "p"):
            m = UR.match(raw.rstrip("\n"))
            if m:
                u = det.EvalUtterance(*m.groups())
                if sec == "t": t = u
                else: post.append(u)
    return cnt


def metrics(y, p, w=None):
    y = np.asarray(y); p = np.asarray(p)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    acc = (w * (y == p)).sum() / w.sum()
    f = []
    for c in (0, 1):
        tp = (w * ((y == c) & (p == c))).sum()
        f.append(2 * tp / ((w * (y == c)).sum() + (w * (p == c)).sum()))
    return acc, f[0], f[1], (f[0] + f[1]) / 2


def kappa(a, b, w=None):
    a = np.asarray(a); b = np.asarray(b)
    w = np.ones(len(a)) if w is None else np.asarray(w)
    po = (w * (a == b)).sum() / w.sum()
    pe = sum(((w * (a == c)).sum() / w.sum()) * ((w * (b == c)).sum() / w.sum()) for c in (0, 1))
    return po, (po - pe) / (1 - pe)


def main():
    R = [json.loads(l) for l in open(PROMPTS)]
    cl = {}
    for l in open(CLAUDE, encoding="utf-8"):
        p = l.strip().split("\t")
        if len(p) == 2:
            cl[int(p[0])] = 1 if p[1] == "はい" else 0
    llm = [r for r in R if r["route"] == "llm"]
    miss = [r["line_no"] for r in llm if r["line_no"] not in cl]
    print(f"Claude の判定 {len(cl)} 件 / LLM 判定の対象 {len(llm)} 件（欠け {len(miss)}）")
    for r in R:
        r["claude"] = 1 if r["route"] == "rule_nextdiff" else cl.get(r["line_no"])
    R = [r for r in R if r["claude"] is not None]

    pop = population_strata()
    smp = collections.Counter((r["route"], r["qwen"]) for r in R)
    print("\n層（経路, Qwen）  母集団  標本")
    for k in sorted(pop):
        print(f"  {str(k):22s} {pop[k]:8d} {smp[k]:5d}")
    w = np.array([pop[(r["route"], r["qwen"])] / smp[(r["route"], r["qwen"])] for r in R])

    y = [r["gold"] for r in R]
    print(f"\n人手正解との一致（完了/継続の 2 値・{len(R)} 件。左＝標本そのまま／右＝母集団の比率で重み付け）")
    print(f"{'':12s}{'一致率':>14s}{'継続F1':>14s}{'完了F1':>14s}{'macro-F1':>14s}")
    for nm in ("qwen", "claude"):
        a = metrics(y, [r[nm] for r in R]); b = metrics(y, [r[nm] for r in R], w)
        print(f"{'Qwen3-8B' if nm == 'qwen' else 'Claude':12s}"
              + "".join(f"{a[i]:7.3f}/{b[i]:.3f}" for i in range(4)))
    L = [i for i, r in enumerate(R) if r["route"] == "llm"]
    yl = [y[i] for i in L]
    print(f"\nLLM が判定した区間だけ（{len(L)} 件・標本そのまま）")
    for nm in ("qwen", "claude"):
        a = metrics(yl, [R[i][nm] for i in L])
        print(f"  {'Qwen3-8B' if nm == 'qwen' else 'Claude':10s} 一致率 {a[0]:.3f}  継続F1 {a[1]:.3f}  完了F1 {a[2]:.3f}  macro {a[3]:.3f}")
    po, k = kappa([R[i]["qwen"] for i in L], [R[i]["claude"] for i in L])
    pw, kw = kappa([R[i]["qwen"] for i in L], [R[i]["claude"] for i in L], w[L])
    print(f"\nClaude と Qwen の一致（LLM 判定分）: 一致率 {po:.3f}（κ {k:.3f}）／重み付け {pw:.3f}（κ {kw:.3f}）")
    c = collections.Counter((R[i]["qwen"], R[i]["claude"], R[i]["gold"]) for i in L)
    print("\n内訳（LLM 判定分）  Qwen  Claude  人手 → 件数")
    for (q, cc, g), n in sorted(c.items()):
        print(f"  {'完了' if q else '継続'}  {'完了' if cc else '継続'}  {'完了' if g else '継続'}  {n:4d}")
    dis = [i for i in L if R[i]["qwen"] != R[i]["claude"]]
    print(f"\nQwen と Claude が割れた {len(dis)} 件で人手に合っていたのは: "
          f"Qwen {sum(R[i]['qwen'] == y[i] for i in dis)} / Claude {sum(R[i]['claude'] == y[i] for i in dis)}")
    print(f"Claude の判定の内訳: 完了 {sum(R[i]['claude'] for i in L)} / 継続 {sum(1 - R[i]['claude'] for i in L)}"
          f"（Qwen は 完了 {sum(R[i]['qwen'] for i in L)} / 継続 {sum(1 - R[i]['qwen'] for i in L)}、"
          f"人手は 完了 {sum(yl)} / 継続 {len(yl) - sum(yl)}）")


if __name__ == "__main__":
    main()
