#!/usr/bin/env python
"""保存した生の応答（<name>.raw.jsonl）から、今の run_gold_prompts.py の読み取りで <name>.tsv を作り直す（2026-10-12）。
読み取りの不具合を直したあと、判定をやり直さずに済ませるため。元の tsv は <name>.tsv.old に退避。
  python reparse_gold.py <outdir> <name> <family>
"""
import json
import os
import sys

from run_gold_prompts import final_text, parse_response

d, name, family = sys.argv[1:4]
tsv = os.path.join(d, f"{name}.tsv"); raw = os.path.join(d, f"{name}.raw.jsonl")
rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
if not os.path.exists(tsv + ".old"):
    os.rename(tsv, tsv + ".old")
bad = 0
with open(tsv, "w", encoding="utf-8") as f:
    for r in rows:
        t = final_text(r["raw"], family)
        lab = parse_response(t) if t is not None else None
        bad += lab not in (0, 1)
        f.write(f"{r['line_no']}\t{'はい' if lab == 1 else 'いいえ'}\n")
print(f"{name}: {len(rows)} 件 / 読めず {bad}")
