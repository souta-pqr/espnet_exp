#!/bin/bash
# 2026-11-19：実時間と同じ切り方の学習データ（train_gs．silero-vad の音声区間の頭 〜 0.4 秒以上の無音の頭）で，提案手法（espmk1）を学び直す．
# 学習データのほかは espmk1 と同じ（CTCW=1.0・MAXSEC=12・lr 5e-4・2 エポック）．GPU が 10 GB なので bs 2・acc 32（実質のバッチは同じ 64）．
# 11/19 20 時：torch 2.7 は PYTORCH_ALLOC_CONF を読まず，メモリの断片化で 2,100 回目の更新で落ちたので PYTORCH_CUDA_ALLOC_CONF に直して流し直す
# silver10 には espnet の環境が無いので，easyturn の環境に silver9 の espnet 環境から typeguard などを写した pylib を足して動かす
A=~/2025/espnet/egs2/cejc/asr1
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet:$HOME/2026/work/encac_gs/pylib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ASR1=$A
O=$A/exp/encac_espmk1gs_20261119
cd ~/2026/work/encac_gs
echo "=== $(date) 学習"
MAXSEC=12 CTCW=1.0 ~/anaconda3/envs/easyturn/bin/python train_ac.py $O --lr 5e-4 --train train_gs --bs 2 --acc 32 > $O.log 2>&1 && echo 終わり $(date) > $O.done
