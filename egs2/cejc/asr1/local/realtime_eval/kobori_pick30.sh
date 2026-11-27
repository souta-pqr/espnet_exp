#!/bin/bash
R=~/2026/work/rt_eval_20261102; A=~/2025/espnet/egs2/cejc/asr1
until [ -e $R/dump_eval_g_espmk1gsp30.done ]; do sleep 60; done
cd ~/2026/work/from_silver10_1120/2026/work/rt_eval_20261102
~/anaconda3/envs/espnet/bin/python rt_pick.py $A/data/dev_g $A/tt_analysis/vad_silero/dev_g $R/dump_dev_g_espmk1gsp30 $A/data/eval_g $A/tt_analysis/vad_silero/eval_g $R/dump_eval_g_espmk1gsp30 > $R/pick_espmk1gsp30.txt 2>&1
echo 終わり > $R/pick_espmk1gsp30.done
