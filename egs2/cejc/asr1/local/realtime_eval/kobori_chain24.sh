#!/bin/bash
# Smart Turn（CEJC で学び直したもの）の応答の決め方を，開発データを実際に流して選ぶための実走（2026-11-18．chain23 の後）．
# 応答するたびに発話を区切り直すので後から当てはめられない．無音 0.6 秒・3 秒で応答は今の決め方（preset smart_turn_ft）と同じにし，閾値 th1 を動かす．
# CPU 6 並列で 1 つずつ（計算の遅れが結果に入るので並べない）
D=~/2026/work/rt_eval_20261102
until [ -e $D/chain23.done ]; do sleep 120; done
cd ~/2026/work/exp-realtime-epd
for th in 0.89 0.80 0.95 0.70; do
  O=dev_g_st_th$th
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  for i in $(seq 0 5); do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/dev_g/wav.scp --out $D/$O --shard $i --nshard 6 --model smart_turn_ft --g1 0.6 --th1 $th --S 3 > $D/log_${O}_$i.txt 2>&1 &
  done
  wait; echo 終わり $(date) > $D/$O.done
done
echo 終わり $(date) > $D/chain24.done
