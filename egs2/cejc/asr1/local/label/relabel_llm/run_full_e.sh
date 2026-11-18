#!/bin/bash
# (e) を加えたプロンプトで、Qwen3-8B 思考ありの票を全データ付け直す（2026-10-10）。
# 正本（relabel/full/Qwen3-8B-think.tsv）は変えず、relabel/full_e/ に出す。
# 確認：本番と同じ仕組みで gold 922 件を出し、gold 単独で出した (e) 版（prompt_check/out/Qwen3-8B-think.add_e.tsv）と
#       一致率 0.97 以上なら本番に入る。→ 実際は 0.868（思考ありは元のプロンプトでも出し直しで 0.869 しか一致しない）で止まった。
#       前回と同じ基準（人手正解での多数決の改善の下限 > 0）で silver11 で確かめ（+0.021 [+0.003, +0.039]），--skip-check で本番だけ流した（2026-10-10）。
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
R=/autofs/diamond5/share/users/kobori/relabel
D=$R/full_e
mkdir -p $D logs_e
M=/data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/
ARGS="--model $M --name Qwen3-8B-think.add_e --thinking --max-tokens 4096 --max-model-len 10240 --gpu-mem 0.85 --chunk 512 --system $R/prompt_check/system_add_e.txt"

if [ "${1:-}" != --skip-check ]; then
echo "=== $(date '+%F %T') 確認：gold 922 件"
$PY run_full.py $ARGS --only-gold --outdir $D/check_gold922 || { echo error > logs_e/state; exit 1; }
$PY - <<PYEOF || { echo gate_failed > logs_e/state; exit 10; }
def rd(p):
    return {int(l.split("\t")[0]): l.rstrip("\n").split("\t")[1] for l in open(p, encoding="utf-8") if "\t" in l}
a = rd("$D/check_gold922/Qwen3-8B-think.add_e.tsv"); b = rd("$R/prompt_check/out/Qwen3-8B-think.add_e.tsv")
k = sorted(set(a) & set(b)); agree = sum(a[x] == b[x] for x in k) / len(k)
print(f"gold 単独の (e) 版との一致: {len(k)} 件 / 一致率 {agree:.3f}")
raise SystemExit(0 if len(k) == 922 and agree >= 0.97 else 1)
PYEOF
fi

echo "=== $(date '+%F %T') 本番"
$PY run_full.py $ARGS --outdir $D || { echo error > logs_e/state; exit 1; }
echo done > logs_e/state
echo "=== $(date '+%F %T') 終了"
