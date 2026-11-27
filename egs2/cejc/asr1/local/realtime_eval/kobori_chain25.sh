#!/bin/bash
# エンコーダ固定＋Transformer の評価データの実走（2026-11-19）．開発データで cut-in rate の上限 7%・8% のときに選んだ
# g1 0.5・th1 0.70 と 0.60 を流す（0.79 は eval_g_pg1x がある）．ほかの値は eval_g_pg1x と同じ．chain24（Smart Turn の開発データ）の後に 1 本ずつ
D=~/2026/work/rt_eval_20261102; E=~/2026/work/exp-realtime-epd
until [ -e $D/chain24.done ]; do sleep 120; done
cd $E
for v in 0.70 0.60; do
  O=eval_g_pg1x_th$v
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  for i in 0 1 2 3 4 5; do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/eval_g/wav.scp --out $D/$O --shard $i --nshard 6 --model proposed_new --no_past --g1 0.5 --th1 $v --g2 2.0 --th2 0.4 --S 3 --phi 0.5 > $D/log_${O}_$i.txt 2>&1 &
  done
  wait; echo 終わり $(date) > $D/$O.done
done
echo 終わり $(date) > $D/chain25.done
