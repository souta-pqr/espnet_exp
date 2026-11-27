#!/bin/bash
# 2026-11-21：Smart Turn（chain26 の後）．開発データ（chain24）で th1 0.70／0.80／0.89／0.95 → cut-in rate 9.1／7.5／5.5／4.0%．
# 7% 前後が無いので開発データで th1 0.85 を足し，評価データで th1 0.80（≤ 8%）・0.89（≤ 6%）を流す．ほかは chain24 と同じ
D=~/2026/work/rt_eval_20261102
until [ -e $D/chain26.done ]; do sleep 120; done
cd ~/2026/work/exp-realtime-epd
for x in dev_g:0.85 eval_g:0.80 eval_g:0.89; do
  d=${x%%:*}; th=${x##*:}; O=${d}_st_th$th
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  pids=()
  for i in $(seq 0 5); do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/$d/wav.scp --out $D/$O --shard $i --nshard 6 --model smart_turn_ft --g1 0.6 --th1 $th --S 3 > $D/log_${O}_$i.txt 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"; echo 終わり $(date) > $D/$O.done
done
echo 終わり $(date) > $D/chain27.done
