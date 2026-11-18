#!/bin/bash
# 2026-11-14：提案手法（espm）を 4 エポックに延ばして学習（2 エポックの終わりでも検証の損失が下がり続けていたため）→ 実時間のスコア
#             chain5（比較手法の書き出し）が終わってから始める（書き出しと GPU の学習を並べない）
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -e $D/chain5.done ]; do sleep 120; done
N=espm4; O=$A/exp/encac_${N}_20261114
export MAXSEC=12
cd ~/2026/work/easy_turn/cejc
echo "=== $(date) 学習 $N"
$PY train_ac.py $O --epochs 4 --lr 5e-4 --train train_gxac --bs 8 --acc 8 > $O.log 2>&1 || { echo "学習 $N 失敗"; exit 1; }
$PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; exit 1; }
cd ~/2026/work/exp-realtime-epd; $PY tools/check_encac.py weights/encac_$N.pt --device cuda 2>&1 | tail -n 2
cd $D
for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
echo 終わり $(date) > $D/chain6.done
