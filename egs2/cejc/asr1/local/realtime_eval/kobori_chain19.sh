#!/bin/bash
# 2026-11-17：文字を当てる課題（CTC，重み 1.0）を足した版（espmk1）を提案手法の候補として，工夫を 1 つずつ外す比較を作り直す → 実時間のスコア（CEJC）
#   espk1    ：前の区間とつないだ音声なし（学習データ train_gx）
#   espmk1pre：読み出しを区間末の 0.6 秒前〜2.0 秒後から選ぶ（PRE=0.6）
#   （エンコーダを固定する版は，文字を当てる課題を足してもエンコーダが変わらないので作り直さない）
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
one() {   # <名前> <学習データ> <PRE>
  N=$1; O=$A/exp/encac_${N}_20261117
  cd ~/2026/work/easy_turn/cejc
  echo "=== $(date) 学習 $N"
  MAXSEC=12 CTCW=1.0 PRE=$3 $PY train_ac.py $O --lr 5e-4 --train $2 --bs 8 --acc 8 > $O.log 2>&1 || { echo "学習 $N 失敗"; return 1; }
  $PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; return 1; }
  cd $D
  for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
}
one espk1 train_gx 0
one espmk1pre train_gxac 0.6
echo 終わり $(date) > $D/chain19.done
