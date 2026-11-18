#!/bin/bash
# 層の重み付き和（2026-11-06，silver13）：esp4（run_enc4.log）が終わってから，エンコーダを固めて全部の層の重み付き和を読む
cd ~/2026/work/easy_turn/cejc
until grep -q "完了\|失敗" run_enc4.log; do sleep 300; done
WSUM=1 TRAIN_ARGS="--freeze_enc" bash run_enc.sh espfzws espnet:$HOME/2025/espnet/egs2/cejc/asr1/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth 5e-4
