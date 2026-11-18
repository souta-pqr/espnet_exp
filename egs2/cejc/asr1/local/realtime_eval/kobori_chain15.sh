#!/bin/bash
# 2026-11-16：Hazumi1712・Hazumi1902（Hazumi1911 と同じ作り方のユーザだけの音声）に実時間のスコアを書き出す．chain14（学習と書き出し）が終わってから
D=~/2026/work/rt_eval_20261102
H=/autofs/diamond5/share/users/kobori/hazumi_audio
C=1:0.01,2:0.011,5:0.0134,10:0.0259,20:0.0529,30:0.0785,45:0.1179,60:0.1549,90:0.2328
until [ -e $D/chain14.done ] || ! pgrep -f '[k]obori_chain14[.]sh' > /dev/null; do sleep 60; done
cd ~/2026/work/exp-realtime-epd
for O in "$@"; do
  [ -e weights/encac_$O.pt ] || { echo "重みなし $O"; continue; }
  for V in 1712 1902; do
    echo "=== $(date) Hazumi$V $O"
    for i in 0 1 2 3 4 5; do
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ~/anaconda3/envs/espnet/bin/python $D/rt_dump.py --wav_scp $H/Hazumi${V}_user.scp --out $D/dump_hazumi${V}_$O --shard $i --nshard 6 --model encac --device cuda --encac weights/encac_$O.pt --encac_cost $C --encac_ctx p0 > $D/log_dump_hazumi${V}_${O}_$i.txt 2>&1 &
    done
    wait
    ls $D/dump_hazumi${V}_$O | wc -l
  done
done
echo 終わり $(date) > $D/chain15.done
