#!/bin/bash
# 実時間のスコアの書き出し・encac 以外の方式（2026-11-13）： bash kobori_rt_dump2.sh <データ名> <並列数> <出力の名前> <rt_dump.py への引数（--model …）>
# CPU で動かし，計算時間はその場で測る（ほかの重いジョブと並べない）
cd ~/2026/work/exp-realtime-epd
D=~/2026/work/rt_eval_20261102
N=$1; K=$2; O=$3; shift 3
for i in $(seq 0 $((K - 1))); do
  OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_dump.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/$N/wav.scp --out $D/dump_${N}_$O --shard $i --nshard $K "$@" > $D/log_dump_${N}_${O}_$i.txt 2>&1 &
done
wait
echo 終わり $(date) > $D/dump_${N}_$O.done
