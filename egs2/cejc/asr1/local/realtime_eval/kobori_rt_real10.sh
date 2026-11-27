#!/bin/bash
# silver10 用の kobori_rt_real.sh（2026-11-20）．espnet の環境が無いので easyturn の環境＋pylib（typeguard・silero_vad など）で動かす．
# 計算の時間は表（C）から取るので，どの計算機で流しても同じスコアになる．
#   bash kobori_rt_real10.sh <データ名> <並列数> <出力の名前> <重み> <rt_eval.py への追加の引数…>（条件 2 を使わないときは --g2 99 --th2 0）
cd ~/2026/work/exp-realtime-epd
D=~/2026/work/rt_eval_20261102
C=1:0.01,2:0.011,5:0.0134,10:0.0259,20:0.0529,30:0.0785,45:0.1179,60:0.1549,90:0.2328
N=$1; K=$2; O=$3; Wt=$4; shift 4
for i in $(seq 0 $((K - 1))); do
  PYTHONPATH=$HOME/2025/espnet:$HOME/2026/work/encac_gs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ~/anaconda3/envs/easyturn/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/$N/wav.scp --out $D/${N}_$O --shard $i --nshard $K --model encac --device cuda --encac $Wt --encac_cost $C --encac_ctx p0 "$@" > $D/log_${N}_${O}_$i.txt 2>&1 &
done
wait
echo 終わり $(date) > $D/${N}_$O.done
