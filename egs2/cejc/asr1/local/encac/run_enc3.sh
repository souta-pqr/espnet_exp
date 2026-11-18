#!/bin/bash
# 切り分け A（2026-11-05，silver13）：単純な仕組み（平均＋線形層）＋提案手法のエンコーダを固める（espfz）。run_enc2.sh が終わってから始める
cd ~/2026/work/easy_turn/cejc
while pgrep -f "bash run_enc2.sh" > /dev/null; do sleep 300; done
TRAIN_ARGS="--freeze_enc" bash run_enc.sh espfz espnet:$HOME/2025/espnet/egs2/cejc/asr1/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth 5e-4
