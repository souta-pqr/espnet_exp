#!/bin/bash
# 2026-11-16：区間の文字を当てる追加の課題（CTC，重み 3.0）を足して学習する版（espmk3．CTCW=3.0．ほかは 11/11 の espm と同じ）→ 実時間のスコア（CEJC・Hazumi1911）
#             短い句の切れ端（「昔の人」「あそこの」など）でスコアが高く出るのが弱点で，文字の手がかりを contextual block conformer エンコーダに残すため
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=espmk3; O=$A/exp/encac_${N}_20261117
cd ~/2026/work/easy_turn/cejc
echo "=== $(date) 学習 $N"
MAXSEC=12 CTCW=3.0 $PY train_ac.py $O --lr 5e-4 --train train_gxac --bs 8 --acc 8 > $O.log 2>&1 || { echo "学習 $N 失敗"; exit 1; }
$PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; exit 1; }
cd ~/2026/work/exp-realtime-epd; $PY tools/check_encac.py weights/encac_$N.pt --device cuda 2>&1 | tail -n 2
cd $D
for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
rm -f $D/chain13.done $D/chain15.done; bash kobori_chain13.sh $N; touch $D/chain14.done; bash kobori_chain15.sh $N
echo 終わり $(date) > $D/chain18.done
