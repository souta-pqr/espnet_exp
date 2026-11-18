#!/bin/bash
# 2026-11-14：学習の録音（1387 本・489 時間）の silero の確率を書き出す（CPU 4 並列・1 スレッドずつ）
D=~/2026/work/rt_eval_20261102; A=~/2025/espnet/egs2/cejc/asr1
cd ~/2026/work/exp-realtime-epd
for i in 0 1 2 3; do PYTHONPATH=. ~/anaconda3/envs/epd/bin/python $D/train_vad.py $A/data/train_gxac/wav.scp $A/tt_analysis/vad_silero/train_g $i 4 > $D/log_train_vad_$i.txt 2>&1 & done
wait
echo 終わり $(date) > $D/train_vad.done
