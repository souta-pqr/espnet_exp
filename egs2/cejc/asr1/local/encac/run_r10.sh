#!/bin/bash
# 2026-11-27（silver10．本来の予約の外で借りている．持ち主が使うときは止める）
# 提案手法（espmk1gs．全フレームの平均）を，補助タスクの出力層（best_aux.pt）も残す形で学び直す．CER は silver13 で測る（ここには editdistance が無い）．
# 設定は元の run_gs10.sh と同じ（train_gs・CTCW=1.0・MAXSEC=12・lr 5e-4・2 エポック・bs 2・acc 32．環境も同じ easyturn＋pylib）．
# データは silver13 から写したもの（~/2026/work/asr1_s13/data）．
A=~/2025/espnet/egs2/cejc/asr1
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet:$HOME/2026/work/encac_pool/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ASR1=$HOME/2026/work/asr1_s13
O=$A/exp/encac_espmk1gsr_20261127
cd ~/2026/work/encac_pool
echo "=== $(date) 学習"
POOL=0 MAXSEC=12 CTCW=1.0 ~/anaconda3/envs/easyturn/bin/python train_ac.py $O --lr 5e-4 --train train_gs --bs 2 --acc 32 > $O.log 2>&1 && echo 終わり $(date) > $O.done
echo "=== $(date) 終わり"
