#!/bin/bash
# 11/26：判定する時刻ごとの正解率（acc_by_off.py）を 3 つのモデルで出す．GPU は書き出しと共有するが，書き出しの計算時間は表から取るのでスコアは変わらない
A=~/2025/espnet/egs2/cejc/asr1; PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet ASR1=$A OMP_NUM_THREADS=2
cd ~/2026/work/encac_pool
$PY acc_by_off.py ~/2026/work/from_silver10_1120/2025/espnet/egs2/cejc/asr1/exp/encac_espmk1gs_20261119/best.pt 0
$PY acc_by_off.py $A/exp/encac_espmk1gsp1_20261126/best.pt 1
$PY acc_by_off.py $A/exp/encac_espmk1gsp30_20261126/best.pt 30
