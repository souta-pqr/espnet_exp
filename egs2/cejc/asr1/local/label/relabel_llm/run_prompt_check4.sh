#!/bin/bash
# 基準 (e) を加えたプロンプトを Qwen3-32B-AWQ 思考なし（多数決の 2 票目）で（2026-10-09）
# 32B は ~/.cache/huggingface/hub に一時的にダウンロード（logs_e32/download.log）。判定が終わったら消す。
# 設定は本番の 32B（run32_main.sh）と同じ：最大長 5120・GPU メモリ 0.85・出力 64 トークン
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
until grep -q "^DONE" logs_e32/download.log; do sleep 30; done
D=/autofs/diamond5/share/users/kobori/relabel/prompt_check
~/anaconda3/envs/llm/bin/python run_gold_prompts.py --prompts $D/gold1000_prompts.add_e.jsonl --model Qwen/Qwen3-32B-AWQ --name Qwen3-32B-AWQ-nothink.add_e \
  --max-tokens 64 --max-model-len 5120 --gpu-mem 0.85 --chunk 256 --outdir $D/out
echo done > $D/out/DONE4
