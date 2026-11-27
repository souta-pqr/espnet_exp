#!/bin/bash
# 2026-11-20（silver9 は 11/20 の 1 日だけ使える）：今の学習データ（train_gxac．書き起こしの区間）と実時間と同じ切り方のデータ（train_gs）を合わせて，
# 提案手法（espmk1）を学び直す．合わせると 617,459 区間で多いので 1 エポック（更新 約 9,600 回．espmk1 の 11,737 回に近い）．bs 4・acc 16（実質のバッチは同じ 64）
A=~/2025/espnet/egs2/cejc/asr1
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_ALLOC_CONF=expandable_segments:True ASR1=$A
O=$A/exp/encac_espmk1gxacgs_20261120
cd ~/2026/work/encac_gs
echo "=== $(date) 学習"
MAXSEC=12 CTCW=1.0 ~/anaconda3/envs/espnet/bin/python train_ac.py $O --lr 5e-4 --train train_gxacgs --bs 4 --acc 16 --epochs 1 > $O.log 2>&1 && echo 終わり $(date) > $O.done
