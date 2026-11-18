#!/bin/bash
# 2026-11-13：(1) 提案手法の改良版 espmt（フレームの平均＋最後の 30 フレーム（約 1 秒）の平均を読む．ほかは 11/11 の espm と同じ）の学習 → 実時間のスコア
#             (2) 比較手法（前回までの提案手法 proposed_new・Smart Turn）の実時間のスコアの書き出し（応答の決め方を開発データの実時間で選ぶため）
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=espmt; O=$A/exp/encac_${N}_20261113
(
  export MAXSEC=12 TAIL=30
  cd ~/2026/work/easy_turn/cejc
  echo "=== $(date) 学習 $N"
  $PY train_ac.py $O --lr 5e-4 --train train_gxac --bs 8 --acc 8 > $O.log 2>&1 || { echo "学習 $N 失敗"; exit 1; }
  $PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; exit 1; }
  cd ~/2026/work/exp-realtime-epd; $PY tools/check_encac.py weights/encac_$N.pt --device cuda 2>&1 | tail -n 2
  cd $D
  for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
)
echo 終わり $(date) > $D/chain5a.done
cd $D
for m in proposed_new smart_turn_ft; do
  for d in eval_g dev_g; do echo "=== $(date) 実時間のスコア $m $d"; bash kobori_rt_dump2.sh $d 6 $m --model $m; done
done
echo 終わり $(date) > $D/chain5.done
