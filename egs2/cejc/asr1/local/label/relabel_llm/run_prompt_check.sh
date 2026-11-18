#!/bin/bash
# プロンプトの難しさの確認（2026-10-09）：簡単にしたプロンプト 3 通り × Qwen3-8B（思考なし）で人手正解 922 件を判定
# Qwen3-8B は /data の共有キャッシュを読むだけ（自分のキャッシュからは消えているため）
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
D=/autofs/diamond5/share/users/kobori/relabel/prompt_check
for v in def noovl def_noovl; do
  $PY run_gold_prompts.py --prompts $D/gold1000_prompts.$v.jsonl --model /data/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/ --name Qwen3-8B-nothink.$v \
    --max-tokens 64 --max-model-len 8192 --gpu-mem 0.85 --chunk 256 --outdir $D/out
done
echo done > $D/out/DONE
