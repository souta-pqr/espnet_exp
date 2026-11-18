#!/bin/bash
# 2026-11-15：同じ話者の次の声までの時間を当てる課題を足して学習する版（espmn．NEXT=1．ほかは 11/11 の espm と同じ）→ 実時間のスコア
#             Next-Turn (arXiv 2606.18094) にならう．chain10 が終わってから始める．最後に silver9 で学習した espmcp の実時間のスコアも出す
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ -e $D/chain10.done ] || ! pgrep -f '[k]obori_chain10[.]sh' > /dev/null; do sleep 60; done
N=espmn; O=$A/exp/encac_${N}_20261115
cd ~/2026/work/easy_turn/cejc
echo "=== $(date) 学習 $N"
MAXSEC=12 NEXT=1 $PY train_ac.py $O --lr 5e-4 --train train_gxac --bs 8 --acc 8 > $O.log 2>&1 || { echo "学習 $N 失敗"; exit 1; }
for N in espmn espmcp; do
  O=$A/exp/encac_${N}_20261115
  [ -e $O/best.pt ] || { echo "$N の重みなし"; continue; }
  cd ~/2026/work/easy_turn/cejc
  $PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; continue; }
  cd ~/2026/work/exp-realtime-epd; $PY tools/check_encac.py weights/encac_$N.pt --device cuda 2>&1 | tail -n 2
  cd $D
  for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
  echo $N 終わり $(date) >> $D/chain11.done
done
