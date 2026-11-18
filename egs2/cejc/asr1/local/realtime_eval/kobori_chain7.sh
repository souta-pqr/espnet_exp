#!/bin/bash
# 2026-11-14：前回までの提案手法（proposed_new）を，しきい値 θ1 を変えて評価データで実際に流す（cut-in rate と latency の曲線の点を増やす）．
#             後から当てはめる採点はこの方式では実走と合わなかった（応答するたびに発話を区切り直すので，モデルに入る音声が決め方で変わる）．
#             chain6（GPU の学習と書き出し）が終わってから始める．CPU 6 並列．ほかの値は eval_g_pg1x（θ1 0.79）と同じ．
D=~/2026/work/rt_eval_20261102
until [ -e $D/chain6.done ]; do sleep 120; done
cd ~/2026/work/exp-realtime-epd
for v in 0.65 0.50; do
  echo "=== $(date) 実走 proposed_new θ1 $v"
  for i in 0 1 2 3 4 5; do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/eval_g/wav.scp --out $D/eval_g_pg1x_th$v --shard $i --nshard 6 --model proposed_new --no_past --g1 0.5 --th1 $v --g2 2.0 --th2 0.4 --S 3 --phi 0.5 > $D/log_eval_g_pg1x_th${v}_$i.txt 2>&1 &
  done
  wait
done
echo 終わり $(date) > $D/chain7.done
