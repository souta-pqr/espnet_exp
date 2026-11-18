#!/bin/bash
# 32B 思考なしの分担（silver13）：gold 922 件の確認を通ったら、silver9 の出力を読みながら後ろから全件を付け直す。
# watch32.sh から guard.py の下で起動される。確認で止まったら logs32/state に gate_failed、終われば done。
set -euo pipefail
trap 'echo error > logs32/state' ERR
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
D=/autofs/diamond5/share/users/kobori/relabel/full
ARGS="--model Qwen/Qwen3-32B-AWQ --name Qwen3-32B-AWQ-nothink.silver13 --max-tokens 64 --max-model-len 5120 --gpu-mem 0.85 --chunk 256"

echo "=== $(date '+%F %T') 確認：gold 922 件の出し直し"
$PY run_full.py $ARGS --only-gold --outdir $D/check_gold922
if ! $PY compare_gold32.py | tee logs32/check_gold.log; then
  echo gate_failed > logs32/state; exit 10
fi

echo "=== $(date '+%F %T') 本番（後ろから。silver9 の Qwen3-32B-AWQ-nothink.tsv にある LINE は飛ばす）"
$PY run_full.py $ARGS --reverse --other $D/Qwen3-32B-AWQ-nothink.tsv
echo done > logs32/state
echo "=== $(date '+%F %T') 終了"
