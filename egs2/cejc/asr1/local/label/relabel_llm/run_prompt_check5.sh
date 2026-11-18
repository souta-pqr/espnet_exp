#!/bin/bash
# 多数決の票を増やす・安定させる確認（2026-10-12）。人手正解 922 件だけ、(e) 版プロンプト。
#   1. Qwen3-8B 思考あり（3 票目）をあと 2 回（思考ありは実行ごとに判定が 1 割強変わるため、回をまたいだ多数決を見る）
#   2. 4・5 票目の候補（/data/huggingface/hub の共有キャッシュを読むだけ。ダウンロードしない）を思考なしで
# 出力：/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_v/
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
P=/autofs/diamond5/share/users/kobori/relabel/prompt_check/gold1000_prompts.add_e.jsonl
O=/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_v
H=/data/huggingface/hub
snap() { ls -d $H/models--$1/snapshots/*/ | head -1; }
mkdir -p $O
for r in 2 3; do
  echo "=== $(date "+%F %T") Qwen3-8B 思考あり 再実行 $r"
  $PY run_gold_prompts.py --prompts $P --model $(snap Qwen--Qwen3-8B) --name Qwen3-8B-think.add_e.r$r --thinking \
    --max-tokens 4096 --max-model-len 10240 --gpu-mem 0.85 --chunk 512 --outdir $O
done
while read m n mem; do
  echo "=== $(date "+%F %T") $n"
  $PY run_gold_prompts.py --prompts $P --model $(snap $m) --name $n.add_e \
    --max-tokens 64 --max-model-len 5120 --gpu-mem $mem --chunk 256 --outdir $O || echo "失敗: $n"
done <<LIST
tokyotech-llm--Qwen3-Swallow-8B-RL-v0.2 Qwen3-Swallow-8B-nothink 0.85
tokyotech-llm--Qwen3-Swallow-32B-RL-v0.2-AWQ-INT4 Qwen3-Swallow-32B-AWQ-nothink 0.85
RedHatAI--Qwen3.8-27B-INT4 Qwen3.8-27B-INT4-nothink 0.85
google--gemma-4-31B-it-qat-w4a16-ct gemma-4-31B-it 0.92
LIST
echo "=== $(date "+%F %T") 終了"
echo done > $O/DONE
