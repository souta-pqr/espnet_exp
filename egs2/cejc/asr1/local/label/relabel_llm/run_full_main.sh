#!/bin/bash
# 段階 2（silver13）：Qwen3-8B 思考ありで LLM 判定区間を全件付け直す。
# watch.sh から guard.py の下で起動される。確認 1・2 を通ったときだけ本番に入る。
# 確認で止まったら logs/state に gate_failed、全件終われば done を書く。再実行すると書き済みの分は飛ばす。
set -euo pipefail
trap 'echo error > logs/state' ERR
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
D=/autofs/diamond5/share/users/kobori/relabel/full
ARGS="--model Qwen/Qwen3-8B --name Qwen3-8B-think --thinking --max-tokens 4096 --max-model-len 10240 --gpu-mem 0.85 --chunk 512"

echo "=== $(date '+%F %T') 確認 1：対象数とプロンプトの一致"
$PY full_items.py --check 2>&1 | grep -v -i warn | tee logs/check1_prompts.log
if ! grep -q '経路の食い違い: 0 件' logs/check1_prompts.log \
   || ! grep -q '見つかった 922 / 文字列が完全一致 922 / 不一致 0' logs/check1_prompts.log; then
  echo gate_failed > logs/state; exit 10
fi

echo "=== $(date '+%F %T') 確認 2：gold 922 件の出し直し"
$PY run_full.py $ARGS --only-gold --outdir $D/check_gold922
if ! $PY compare_gold.py | tee logs/check2_gold.log; then
  echo gate_failed > logs/state; exit 10
fi

echo "=== $(date '+%F %T') 本番"
$PY run_full.py $ARGS
echo done > logs/state
echo "=== $(date '+%F %T') 終了"
