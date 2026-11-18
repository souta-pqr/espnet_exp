#!/bin/bash
# espm の入力の範囲を変えた実時間評価（2026-11-12）： bash kobori_rt_ctx.sh <wav.scp> <データ名> <並列数> <範囲> ...
cd ~/2026/work/exp-realtime-epd
D=~/2026/work/rt_eval_20261102
C=1:0.01,2:0.011,5:0.0134,10:0.0259,20:0.0529,30:0.0785,45:0.1179,60:0.1549,90:0.2328
W=$1; N=$2; K=$3; shift 3
for v in "$@"; do
  X=""; [ $v != base ] && X="--encac_ctx $v"
  for i in $(seq 0 $((K - 1))); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ~/anaconda3/envs/espnet/bin/python $D/rt_eval.py --wav_scp $W --out $D/${N}_encac_$v --shard $i --nshard $K --model encac --device cuda --encac_cost $C $X > $D/log_${N}_encac_${v}_$i.txt 2>&1 &
  done
done
wait
echo 終わり $(date) > $D/${N}_encac_ctx.done
