#!/bin/bash
# espmpre（読み出しを区間末の 0.6 秒前〜2.0 秒後から選ぶ）の学習のやり直し（2026-11-13）．
# chain2 では --bs 16 --acc 4 でメモリ不足．11/11 の espm と同じ --bs 8 --acc 8 にする．
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAXSEC=12 PRE=0.6
N=espmpre; O=$A/exp/encac_${N}_20261112
cd ~/2026/work/easy_turn/cejc
mv $O.log $O.oom.log
echo "=== $(date) 学習 $N"
$PY train_ac.py $O --lr 5e-4 --train train_gxac --bs 8 --acc 8 > $O.log 2>&1 || { echo "学習 $N 失敗"; exit 1; }
$PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; exit 1; }
cd $D
for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
echo 終わり $(date) > $D/chain4.done
