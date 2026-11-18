#!/bin/bash
# run_check_f.sh の最初の 1 本（add_e.gold1000）が起動時の GPU メモリ不足で落ちたため，終わった後に流し直す（2026-10-17）。
set -uo pipefail
O=/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_route
while [ ! -f $O/DONE ]; do sleep 60; done
export PATH=$HOME/anaconda3/envs/llm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/autofs/diamond5/share/users/kobori/relabel
M=$(ls -d /data/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/ | head -1)
echo "=== $(date "+%F %T") add_e.gold1000 再実行"
python run_full.py --model $M --family gemma --thinking --max-tokens 2560 --max-model-len 7680 --gpu-mem 0.92 --chunk 256 \
    --routes "llm rule_nextdiff" --system $R/prompt_check/system_add_e.txt --name gemma-4-31B-it-think.add_e.gold1000 --outdir $O --only-gold || echo "失敗"
echo "=== $(date "+%F %T") 終了"; echo done > $O/DONE2
