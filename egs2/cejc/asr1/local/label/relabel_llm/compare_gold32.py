#!/usr/bin/env python
"""32B 分担の確認：silver13 で出し直した gold 922 件を、段階 1 の Qwen3-32B-AWQ-nothink.tsv と比べる。
思考なしは出し直しでほぼ揺れない（silver9 の出し直しは 1 件差）ので、違いが MAX_DIFF 件以下なら終了コード 0。
"""
import sys

R = "/autofs/diamond5/share/users/kobori/relabel"
MAX_DIFF = 10


def read(p):
    return dict(l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if l.strip())


old = read(f"{R}/Qwen3-32B-AWQ-nothink.tsv")
new = read(f"{R}/full/check_gold922/Qwen3-32B-AWQ-nothink.silver13.tsv")
bad = sum(1 for l in open(f"{R}/full/check_gold922/Qwen3-32B-AWQ-nothink.silver13.unparsed.tsv", encoding="utf-8") if l.strip())
diff = sorted(int(k) for k in old if new.get(k) != old[k])
ok = len(old) == len(new) == 922 and len(diff) <= MAX_DIFF
print(f"段階 1: {len(old)} 件 / silver13 で出し直し: {len(new)} 件 / 判定が違う: {len(diff)} 件 {diff} / "
      f"読めなかった応答: {bad} 件 → {'OK' if ok else 'NG'}（基準: 違い {MAX_DIFF} 件以下）")
sys.exit(0 if ok else 1)
