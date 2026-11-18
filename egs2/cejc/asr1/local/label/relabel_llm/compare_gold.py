#!/usr/bin/env python
"""確認 2：本番の仕組み（run_full.py --only-gold）で出し直した 922 件を、段階 1 の結果と比べる。

思考ありは GPU やバッチが違うと貪欲デコードでも思考が分かれ、個々の判定が 1 割ほど入れ替わる
（check_gold922/think_rerun_variance.md）。そのため件ごとの一致ではなく、
出し直した結果で組んだ 3 モデル多数決（今の Qwen3-8B ＋ Qwen3-32B-AWQ 思考なし ＋ 8B 思考あり）が
今の Qwen3-8B より良いこと（ブートストラップ 95% 区間の下限 > 0）を本番に進む基準にする。
922 件そろっていて基準を満たせば終了コード 0、それ以外は 1。

  python compare_gold.py
"""
import json
import random
import sys

R = "/autofs/diamond5/share/users/kobori/relabel"
NAME = "Qwen3-8B-think"


def read(p):
    return {k: (1 if v == "はい" else 0)
            for k, v in (l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if l.strip())}


P = {str(r["line_no"]): r for l in open(f"{R}/gold1000_prompts.jsonl", encoding="utf-8")
     if (r := json.loads(l))["route"] == "llm"}
gold = {k: r["gold"] for k, r in P.items()}
cur = {k: int(r["qwen"] == 1) for k, r in P.items()}
b32 = read(f"{R}/Qwen3-32B-AWQ-nothink.tsv")
old = read(f"{R}/{NAME}.tsv")
new = read(f"{R}/full/check_gold922/{NAME}.tsv")
bad = sum(1 for l in open(f"{R}/full/check_gold922/{NAME}.unparsed.tsv", encoding="utf-8") if l.strip())
diff = sorted(int(k) for k in old if new.get(k) != old[k])
if len(new) != 922:
    print(f"出し直し {len(new)} 件（922 件そろっていない）→ NG")
    sys.exit(1)

acc = lambda d: sum(d[k] == gold[k] for k in gold) / len(gold)
mv = {k: int(cur[k] + b32[k] + new[k] >= 2) for k in gold}
d = [(mv[k] == gold[k]) - (cur[k] == gold[k]) for k in gold]
rng = random.Random(0)
bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
lo, hi = bs[125], bs[4874]
ok = lo > 0
print(f"段階 1 と判定が違う: {len(diff)} / 922 件 / 読めなかった応答（出し直し）: {bad} 件")
print(f"人手正解との一致率: 8B 思考あり 段階1 {acc(old):.3f} / 出し直し {acc(new):.3f}")
print(f"3 モデル多数決（出し直しの 8B 思考あり）{acc(mv):.3f}、今の Qwen3-8B {acc(cur):.3f} との差 "
      f"{sum(d) / len(d):+.3f}（95% 区間 {lo:+.3f}〜{hi:+.3f}）→ {'OK' if ok else 'NG'}（基準: 下限 > 0）")
sys.exit(0 if ok else 1)
