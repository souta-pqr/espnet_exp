#!/bin/bash
# 11/27（CPU だけ）：無音が 1.2 秒を超えた後のスコアを固定して選び直す（rt_pick_cap.py）．全フレームの平均・最後のフレーム・最後の約 1 秒の平均
A=~/2025/espnet/egs2/cejc/asr1; F=~/2026/work/from_silver10_1120/2026/work/rt_eval_20261102; R=~/2026/work/rt_eval_20261102
PY=~/anaconda3/envs/espnet/bin/python
cd ~/2026/work/rt_cap
for x in "mean $F espmk1gs" "p1 $R espmk1gsp1" "p30 $R espmk1gsp30"; do set -- $x
  CAP=1.2 OMP_NUM_THREADS=2 $PY rt_pick_cap.py $A/data/dev_g $A/tt_analysis/vad_silero/dev_g $2/dump_dev_g_$3 $A/data/eval_g $A/tt_analysis/vad_silero/eval_g $2/dump_eval_g_$3 > cap1.2_$1.txt 2>&1 &
done
wait; echo 終わり > cap.done
