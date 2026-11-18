#!/usr/bin/env python
"""資料用：Qwen3-8B 思考ありの出し直しで判定が揺れる現象を集計する（think_rerun_variance.md の数字の出どころ）。

  A  段階 1        silver9 RTX 3090、run_gold.py、チャンク 64          relabel/Qwen3-8B-think.{tsv,raw.jsonl}
  B  出し直し 1    silver13 RTX 5090、run_full.py --only-gold、チャンク 512  full/check_gold922/Qwen3-8B-think.{tsv,ntok.tsv}
  C  出し直し 2    silver13 RTX 5090、run_gold.py、チャンク 512        full/check_gold922/rerun2/Qwen3-8B-think.{tsv,raw.jsonl}
いずれも vLLM 0.31.0、bf16、chat template（enable_thinking=True）、temperature 0、max_tokens 4096、同じプロンプト文字列。

出力: full/check_gold922/think_rerun_variance.tsv（LINE ごとの正解・3 回の判定・出力トークン数・A と C の分岐位置）
"""
import json
import random
import statistics as st

from transformers import AutoTokenizer

R = "/autofs/diamond5/share/users/kobori/relabel"
C = f"{R}/full/check_gold922"


def tsv(p):
    return {int(k): int(v == "はい") for k, v in (l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if l.strip())}


def raw(p):
    return {(r := json.loads(l))["line_no"]: r for l in open(p, encoding="utf-8")}


P = {r["line_no"]: r for l in open(f"{R}/gold1000_prompts.jsonl", encoding="utf-8") if (r := json.loads(l))["route"] == "llm"}
gold = {k: r["gold"] for k, r in P.items()}
cur = {k: int(r["qwen"] == 1) for k, r in P.items()}
b32 = tsv(f"{R}/Qwen3-32B-AWQ-nothink.tsv")
lab = {"A": tsv(f"{R}/Qwen3-8B-think.tsv"), "B": tsv(f"{C}/Qwen3-8B-think.tsv"), "C": tsv(f"{C}/rerun2/Qwen3-8B-think.tsv")}
rA, rC = raw(f"{R}/Qwen3-8B-think.raw.jsonl"), raw(f"{C}/rerun2/Qwen3-8B-think.raw.jsonl")
nB = {int(l.split("\t")[0]): int(l.split("\t")[1]) for l in open(f"{C}/Qwen3-8B-think.ntok.tsv", encoding="utf-8")}
nout = {"A": {k: r["n_out"] for k, r in rA.items()}, "B": nB, "C": {k: r["n_out"] for k, r in rC.items()}}
K = sorted(gold)

print("## 入力")
print(f"入力トークン数が A と C で同じ: {sum(rA[k]['n_in'] == rC[k]['n_in'] for k in K)} / {len(K)}")

print("\n## 判定と出力長の一致")
for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
    same = [k for k in K if lab[x][k] == lab[y][k]]
    print(f"{x}-{y}: 判定が違う {len(K) - len(same)} 件 / 出力トークン数が完全一致 {sum(nout[x][k] == nout[y][k] for k in K)} 件"
          f"（判定が同じ {len(same)} 件のうち {sum(nout[x][k] == nout[y][k] for k in same)} 件）")
flip = {k for k in K if len({lab['A'][k], lab['B'][k], lab['C'][k]}) > 1}
print(f"3 回のどこかで判定が割れた: {len(flip)} 件（{len(flip) / len(K):.1%}）")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
div = {}
for k in K:
    a, c = tok(rA[k]["raw"], add_special_tokens=False).input_ids, tok(rC[k]["raw"], add_special_tokens=False).input_ids
    i = next((i for i, (u, v) in enumerate(zip(a, c)) if u != v), min(len(a), len(c)))
    div[k] = i
print("\n## A と C で出力が最初に分かれる位置（トークン）")
dv = [div[k] for k in K if rA[k]["raw"] != rC[k]["raw"]]
print(f"出力が全く同じ: {len(K) - len(dv)} 件 / 分かれた {len(dv)} 件の分岐位置 中央値 {st.median(dv):.0f}、"
      f"25% {sorted(dv)[len(dv) // 4]}、75% {sorted(dv)[3 * len(dv) // 4]}、最初の 10 トークン以内 {sum(d < 10 for d in dv)} 件")

print("\n## 人手正解との一致率と 3 モデル多数決（今の Qwen3-8B 単独 %.3f）" % (sum(cur[k] == gold[k] for k in K) / len(K)))
for x in "ABC":
    mv = {k: int(cur[k] + b32[k] + lab[x][k] >= 2) for k in K}
    d = [(mv[k] == gold[k]) - (cur[k] == gold[k]) for k in K]
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(5000))
    print(f"{x}: 8B 思考あり単独 {sum(lab[x][k] == gold[k] for k in K) / len(K):.3f} / 多数決 {sum(mv[k] == gold[k] for k in K) / len(K):.3f}"
          f"（差 {sum(d) / len(d):+.3f}、95% 区間 {bs[125]:+.3f}〜{bs[4874]:+.3f}）")

print("\n## 例：A と C で判定が違う件の分岐点（前後の文字列）")
for k in [k for k in K if lab["A"][k] != lab["C"][k]][:3]:
    a = tok(rA[k]["raw"], add_special_tokens=False).input_ids
    c = tok(rC[k]["raw"], add_special_tokens=False).input_ids
    i = div[k]
    print(f"[LINE {k}] 正解 {'はい' if gold[k] else 'いいえ'} / A {'はい' if lab['A'][k] else 'いいえ'} / C {'はい' if lab['C'][k] else 'いいえ'}、分岐 {i} トークン目")
    print("  共通部分の末尾: " + repr(tok.decode(a[max(0, i - 25):i])))
    print("  A のその先:     " + repr(tok.decode(a[i:i + 25])))
    print("  C のその先:     " + repr(tok.decode(c[i:i + 25])))

with open(f"{C}/think_rerun_variance.tsv", "w", encoding="utf-8") as f:
    f.write("line_no\tgold\tA_stage1\tB_rerun1\tC_rerun2\tnout_A\tnout_B\tnout_C\tdiverge_tok_AC\n")
    for k in K:
        f.write(f"{k}\t{gold[k]}\t{lab['A'][k]}\t{lab['B'][k]}\t{lab['C'][k]}\t{nout['A'][k]}\t{nout['B'][k]}\t{nout['C'][k]}\t{div[k]}\n")
