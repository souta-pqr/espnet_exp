#!/bin/bash
# エンコーダ固定＋Transformer（proposed_new --no_past）の応答の決め方を，開発データを実際に流して選ぶための実走（2026-11-18）．
# 応答するたびに発話を区切り直すので後から当てはめられない．g2 2.0・th2 0.40・S 3・phi 0.5 は eval_g_pg1x と同じにし，g1 と th1 を動かす．CPU 6 並列で 1 つずつ（計算の遅れが結果に入るので並べない）
D=~/2026/work/rt_eval_20261102
cd ~/2026/work/exp-realtime-epd
for g1 in 0.5 0.4; do for th in 0.79 0.70 0.60 0.50; do
  O=dev_g_pg1x_g${g1}_th$th
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  for i in $(seq 0 5); do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/dev_g/wav.scp --out $D/$O --shard $i --nshard 6 --model proposed_new --no_past --g1 $g1 --th1 $th --g2 2.0 --th2 0.4 --S 3 --phi 0.5 > $D/log_${O}_$i.txt 2>&1 &
  done
  wait; echo 終わり $(date) > $D/$O.done
done; done
echo 終わり $(date) > $D/chain22.done
