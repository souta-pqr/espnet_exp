#!/bin/bash
# 11/26：CER の「区間末のラベルで学習する前」の基準．元の音声認識モデルのエンコーダを固めたまま，補助タスクの出力層（CTC）と 3 クラスの線形層だけを train_gs で学習する．
# ほかは提案手法と同じ（CTCW=1.0・MAXSEC=12・2 エポック・bs 4・acc 16）．23:55 の見張りが止める（止まったら 11/28 に流し直す）
A=~/2025/espnet/egs2/cejc/asr1; PY=~/anaconda3/envs/espnet/bin/python
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_ALLOC_CONF=expandable_segments:True ASR1=$A
O=$A/exp/encac_espmk1gsfz_20261126
cd ~/2026/work/encac_pool
echo "=== $(date) 学習（エンコーダを固める）"
MAXSEC=12 CTCW=1.0 $PY train_ac.py $O --lr 5e-4 --train train_gs --bs 4 --acc 16 --freeze_enc > $O.log 2>&1 || { echo "=== $(date) 止まった"; exit 1; }
echo 終わり $(date) > $O.done
OMP_NUM_THREADS=2 $PY cer_aux.py $O 0 $A/data/eval_g hyp_eval_g_fz.tsv
