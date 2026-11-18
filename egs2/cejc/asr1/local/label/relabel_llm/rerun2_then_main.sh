#!/bin/bash
# 資料用：同じ RTX 5090・同じチャンク（512）で gold 922 件をもう一度出し直し（思考の全文つき）、そのあと本番に入る
set -euo pipefail
trap 'echo error > logs/state' ERR
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
echo "=== $(date '+%F %T') 資料用：同じ GPU での 2 回目の出し直し"
~/anaconda3/envs/llm/bin/python run_gold.py --model Qwen/Qwen3-8B --name Qwen3-8B-think --thinking \
  --max-tokens 4096 --max-model-len 10240 --gpu-mem 0.85 --chunk 512 \
  --outdir /autofs/diamond5/share/users/kobori/relabel/full/check_gold922/rerun2
exec ./run_full_main.sh
