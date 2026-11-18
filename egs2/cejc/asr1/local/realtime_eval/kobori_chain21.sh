#!/bin/bash
# 2026-11-18：waseda_soma（人がターンのラベルを付けた 90 対話，話者ごと 180 本）に実時間のスコアを書き出す．chain20（評価データの実走）が終わってから
D=~/2026/work/rt_eval_20261102
S=/autofs/diamond5/share/users/kobori/waseda_soma
C=1:0.01,2:0.011,5:0.0134,10:0.0259,20:0.0529,30:0.0785,45:0.1179,60:0.1549,90:0.2328
until [ -e $D/chain20.done ] || ! pgrep -f '[k]obori_chain20[.]sh' > /dev/null; do sleep 60; done
cd ~/2026/work/exp-realtime-epd
for O in "$@"; do
  echo "=== $(date) waseda_soma $O"
  for i in 0 1 2 3 4 5; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ~/anaconda3/envs/espnet/bin/python $D/rt_dump.py --wav_scp $S/data/wav.scp --out $D/dump_soma_$O --shard $i --nshard 6 --model encac --device cuda --encac weights/encac_$O.pt --encac_cost $C --encac_ctx p0 > $D/log_dump_soma_${O}_$i.txt 2>&1 &
  done
  wait
  ls $D/dump_soma_$O | wc -l
done
echo 終わり $(date) > $D/chain21.done
