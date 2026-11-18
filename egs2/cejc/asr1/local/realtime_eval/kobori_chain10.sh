#!/bin/bash
# 2026-11-15：silver9 で学習した espmvx（元の区間の頭と声のかたまりの頭の両方で学習．検証の損失 0.4212）の実時間のスコア．chain9 が終わってから
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAXSEC=12
until [ -e $D/chain9.done ]; do sleep 120; done
N=espmvx; O=$A/exp/encac_${N}_20261115
cd ~/2026/work/easy_turn/cejc
$PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; exit 1; }
cd ~/2026/work/exp-realtime-epd; $PY tools/check_encac.py weights/encac_$N.pt --device cuda 2>&1 | tail -n 2
cd $D
for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
V=$A/tt_analysis/vad_silero
$PY rt_pick.py $A/data/dev_g $V/dev_g dump_dev_g_$N $A/data/eval_g $V/eval_g dump_eval_g_$N > pick_$N.txt 2>&1
echo 終わり $(date) > $D/chain10.done
