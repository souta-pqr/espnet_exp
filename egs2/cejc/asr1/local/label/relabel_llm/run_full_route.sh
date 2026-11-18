#!/bin/bash
# 全データの付け直し（2026-10-16 準備）。run_check_f.sh の結果を見て，プロンプトの版と経路を決めてから流す。
#   V=add_e ROUTES=rule_nextdiff bash run_full_route.sh        … 規則で完了の 148,961 件だけ（約 82 時間）
#   V=add_f ROUTES="llm rule_nextdiff" bash run_full_route.sh  … 相槌以外の全 286,773 件（約 160 時間）
# 止めても続きから再開できる（run_full.py が出力済みの line_no を飛ばす）。setsid nohup で起動する。
set -uo pipefail
: "${V:?V=add_e または add_f}" "${ROUTES:?ROUTES を指定}"
export PATH=$HOME/anaconda3/envs/llm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/relabel_llm
R=/autofs/diamond5/share/users/kobori/relabel
M=$(ls -d /data/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/ | head -1)
N=gemma-4-31B-it-think.$V.$(echo $ROUTES | tr ' ' '+')
mkdir -p logs_route
echo running > logs_route/state.$N
echo "=== $(date '+%F %T') $N"
python run_full.py --model $M --name $N --family gemma --thinking --max-tokens 2560 --max-model-len 7680 \
  --gpu-mem 0.92 --chunk 256 --routes "$ROUTES" --system $R/prompt_check/system_$V.txt --outdir $R/full_route \
  || { echo error > logs_route/state.$N; exit 1; }
echo done > logs_route/state.$N
echo "=== $(date '+%F %T') 終了"
