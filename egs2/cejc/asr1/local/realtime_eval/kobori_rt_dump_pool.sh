#!/bin/bash
# kobori_rt_dump.sh の複製（2026-11-26）．フレームの平均をやめたモデル用に，exp-realtime-epd の複製（exp-realtime-epd-pool．
# epd/models/encac.py が重みの "pool" を読む）で流す．exp-realtime-epd は silver11 からコミットしているので触らない．
# 途中で止められたとき（11/26 23:55 の見張り）に .done を書かないよう，全部の録音が書けたときだけ .done を書く．
#   bash kobori_rt_dump_pool.sh <データ名> <並列数> <出力の名前> <重み> [rt_dump.py への追加の引数]
cd ~/2026/work/exp-realtime-epd-pool
D=~/2026/work/rt_eval_20261102
C=1:0.01,2:0.011,5:0.0134,10:0.0259,20:0.0529,30:0.0785,45:0.1179,60:0.1549,90:0.2328
N=$1; K=$2; O=$3; Wt=$4; shift 4
S=~/2025/espnet/egs2/cejc/asr1/data/$N/wav.scp
for i in $(seq 0 $((K - 1))); do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ~/anaconda3/envs/espnet/bin/python $D/rt_dump.py --wav_scp $S --out $D/dump_${N}_$O --shard $i --nshard $K --model encac --device cuda --encac $Wt --encac_cost $C --encac_ctx p0 "$@" >> $D/log_dump_${N}_${O}_$i.txt 2>&1 &
done
wait
n=$(ls $D/dump_${N}_$O | grep -c "^[^.]*\.npz$")
if [ "$n" -eq "$(wc -l < $S)" ]; then echo 終わり $(date) > $D/dump_${N}_$O.done; else echo "書けたのは $n 録音．.done は書かない"; exit 1; fi
