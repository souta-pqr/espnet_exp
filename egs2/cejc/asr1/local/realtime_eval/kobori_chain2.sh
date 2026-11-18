#!/bin/bash
# アブレーションの学習 → デモ用の重みの書き出し → 実時間のスコアの書き出し（2026-11-12．chain1 の後に流す）
#   espmfz ：エンコーダを固定（つないだ音声あり）
#   espmpre：読み出しを区間末の 0.6 秒前〜2.0 秒後から選ぶ（エンコーダも学習・つないだ音声あり）
D=~/2026/work/rt_eval_20261102
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAXSEC=12
until [ -e $D/chain1.done ]; do sleep 60; done
one() {   # <名前> <train_ac.py の引数>
  N=$1; shift
  O=$A/exp/encac_${N}_20261112
  cd ~/2026/work/easy_turn/cejc
  echo "=== $(date) 学習 $N"
  $PY train_ac.py $O --lr 5e-4 --train train_gxac "$@" > $O.log 2>&1 || { echo "学習 $N 失敗"; return 1; }
  $PY ~/2026/work/exp-realtime-epd/tools/export_encac.py $O/best.pt ~/2026/work/exp-realtime-epd/weights/encac_$N.pt || { echo "書き出し $N 失敗"; return 1; }
  cd $D
  for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア $N $d"; bash kobori_rt_dump.sh $d 6 $N weights/encac_$N.pt; done
}
one espmfz --freeze_enc
PRE=0.6 one espmpre
echo 終わり $(date) > $D/chain2.done
