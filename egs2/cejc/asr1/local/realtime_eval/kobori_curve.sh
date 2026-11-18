#!/bin/bash
# 2026-11-14：cut-in rate と latency の曲線を方式ごとに作る（CPU 1 コアずつ）
A=~/2025/espnet/egs2/cejc/asr1; V=$A/tt_analysis/vad_silero; PY=~/anaconda3/envs/espnet/bin/python
for n in "$@"; do $PY rt_curve.py $A/data/dev_g $V/dev_g dump_dev_g_$n $A/data/eval_g $V/eval_g dump_eval_g_$n > curve_$n.txt 2>&1; done
echo 終わり > curve.done
