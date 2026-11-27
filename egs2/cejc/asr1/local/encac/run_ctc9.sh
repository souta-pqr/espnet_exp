#!/bin/bash
# 2026-11-27（silver9．本来の予約の外で借りている．持ち主が使うときは止める）
# 音声認識（補助タスクの CTC）だけでエンコーダを学習する（train_ac.py --ctc_only）．区間末のラベルの学習が CER を下げるかの公平な基準．
# 設定は提案手法と同じ（train_gs・CTCW=1.0・MAXSEC=12・lr 5e-4・2 エポック・bs 2・acc 32）．最後に CER を測る．
A=~/2025/espnet/egs2/cejc/asr1; S=~/2026/work/asr1_s13
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ASR1=$S
O=$A/exp/encac_espmk1gsctc_20261127
cd ~/2026/work/encac_pool
echo "=== $(date) 学習"
POOL=0 MAXSEC=12 CTCW=1.0 ~/anaconda3/envs/espnet/bin/python train_ac.py $O --lr 5e-4 --train train_gs --bs 2 --acc 32 --ctc_only > $O.log 2>&1 || { echo "=== $(date) 止まった"; exit 1; }
echo 終わり $(date) > $O.done
echo "=== $(date) CER"
OMP_NUM_THREADS=4 ~/anaconda3/envs/espnet/bin/python cer_aux.py $O 0 $S/data/eval_g hyp_eval_g_ctc.tsv
echo "=== $(date) 終わり"
