#!/bin/bash
# 基準 (e) を加えたプロンプトを Qwen3-8B 思考ありで（言い切りを「いいえ」にする誤りが思考ありで最も多いため）
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
~/anaconda3/envs/llm/bin/python run_gold_prompts.py --prompts /autofs/diamond5/share/users/kobori/relabel/prompt_check/gold1000_prompts.add_e.jsonl --model /data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/ --name Qwen3-8B-think.add_e --thinking \
  --max-tokens 4096 --max-model-len 10240 --gpu-mem 0.85 --chunk 512 --outdir /autofs/diamond5/share/users/kobori/relabel/prompt_check/out
echo done > /autofs/diamond5/share/users/kobori/relabel/prompt_check/out/DONE3
