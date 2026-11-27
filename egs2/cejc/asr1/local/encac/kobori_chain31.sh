#!/bin/bash
# 11/26：補助タスクの出力層（CTC）で評価データの CER を測る（cer_aux.py）．23:55 の見張りが止める（名前を kobori_chainNN にしてある）
A=~/2025/espnet/egs2/cejc/asr1; PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet ASR1=$A OMP_NUM_THREADS=2
cd ~/2026/work/encac_pool
for N in 1 30; do $PY cer_aux.py $A/exp/encac_espmk1gsp${N}_20261126 $N $A/data/eval_g hyp_eval_g_p$N.tsv; done
