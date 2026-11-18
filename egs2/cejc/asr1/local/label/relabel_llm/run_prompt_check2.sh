#!/bin/bash
# run_prompt_check.sh の後に、基準 (e) を加えたプロンプトを流す
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
until [ -e /autofs/diamond5/share/users/kobori/relabel/prompt_check/out/DONE ]; do sleep 30; done
~/anaconda3/envs/llm/bin/python run_gold_prompts.py --prompts /autofs/diamond5/share/users/kobori/relabel/prompt_check/gold1000_prompts.add_e.jsonl --model /data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/ --name Qwen3-8B-nothink.add_e \
  --max-tokens 64 --max-model-len 8192 --gpu-mem 0.85 --chunk 256 --outdir /autofs/diamond5/share/users/kobori/relabel/prompt_check/out
echo done > /autofs/diamond5/share/users/kobori/relabel/prompt_check/out/DONE2
