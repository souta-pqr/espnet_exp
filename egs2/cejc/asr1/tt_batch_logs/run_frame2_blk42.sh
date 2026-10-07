#!/usr/bin/env bash
# Phase 3: Stage2（frame2 = フレーム単位・pool=attn・same N=5・tail）を
#          新 Stage1（20260921-pureasr-blk42-sp, CER 20.7）の上で再学習する。
#
# 旧 frame2 は Stage1=20260713-pureasr（CER 23.3, block_size 18）の上で学習され、
# 音声のみ macro-F1 0.7306 / 融合 1-best 0.7429 / N-best 0.7461 を出していた。
# エンコーダが変わる（block_size 18->42）ため Stage2 は再学習が必要。
#
# 変更点は config の block_size 18->42 と pureasr_tag のみ。Stage1 の重みは
# 旧と同じ 471 キー・同一形状・同一 normalize 統計なので差し替え可能。
#
# cron から起動する（user@1609.service の外で走らせて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
# 1回目は CUDA OOM で落ちた（reserved-but-unallocated が 6.84GiB＝断片化が大きい）。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== [$(date '+%m-%d %H:%M')] Phase3: frame2 blk42 学習開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

tail=1 tag_suffix=-frame2-blk42 \
    asr_tag=20260924-turntaking-xfmr-pool_attn-frame2-blk42-same_n5_p30 \
    base_cfg=myconf/train_asr_turntaking_xfmr_pool_attn_frame2_blk42.yaml \
    pool=attn scope=same N=5 max_past=30 \
    pureasr_tag=20260921-pureasr-blk42-sp \
    ./run_xfmr_pool.sh \
  && echo "===== [$(date '+%m-%d %H:%M')] 学習 完了 =====" \
  || { echo "===== [$(date '+%m-%d %H:%M')] 学習 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
