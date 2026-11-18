#!/bin/bash
# エンコーダを取り替えた音響だけのモデル（enc_plug.py．2026-11-04，silver13）：学習 → 開発・評価データのダンプ（NOEND）
#   bash run_enc.sh <名前> <ENC> <エンコーダの lr>
cd ~/2026/work/easy_turn/cejc
A=~/2025/espnet/egs2/cejc/asr1
N=$1; export ENC=$2 PYTHONPATH=$HOME/2025/espnet PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=$A/exp/encac_${N}_20261104
PY=~/anaconda3/envs/espnet/bin/python
echo "=== $(date) 学習 $N $ENC lr $3"
$PY train_ac.py $O --lr $3 ${TRAIN_ARGS} > $O.log 2>&1 || { echo "学習 失敗"; exit 1; }
for d in dev_g eval_g; do
  echo "=== $(date) ダンプ $d"
  AC=1 NOEND=1 $PY et_dump.py $O/best.pt $d $A/tt_preds/encac_${N}_${d}_post5.0_noend.tsv > $A/exp/encac_${N}_dump_$d.log 2>&1 || { echo "ダンプ $d 失敗"; exit 1; }
done
touch $A/tt_preds/encac_${N}_dump.done
echo "=== $(date) 完了"
