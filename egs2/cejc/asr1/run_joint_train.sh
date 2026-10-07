#!/usr/bin/env bash
# 音声 ＋ 認識結果テキストの共同学習。
#   nohup ./run_joint_train.sh > tt_batch_logs/joint_train.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
echo "python3 = $(command -v python3)"

echo "===== [$(date '+%m-%d %H:%M')] 1. 学習 ====="
if ! ls -d exp/asr_*-turntaking-xfmr-pool_attn-joint-same_n5_p30 >/dev/null 2>&1; then
    tail=1 tag_suffix=-joint \
        asr_stats_dir=exp/asr_stats_raw_jp_word_pool_attn_same_n5_p30_joint \
        base_cfg=myconf/train_asr_turntaking_xfmr_pool_attn_joint.yaml \
        pool=attn scope=same N=5 max_past=30 pureasr_tag=20260713-pureasr \
        ./run_xfmr_pool.sh || { echo "学習に失敗"; exit 1; }
else
    echo "学習済みなので飛ばす"
fi

echo "===== [$(date '+%m-%d %H:%M')] 2. フレーム単位ダンプ ====="
tag_suffix=-joint sets="eval train_dev" ./run_frame_dump.sh || echo "ダンプに失敗"

echo "===== [$(date '+%m-%d %H:%M')] 3. 比較 ====="
mkdir -p tt_analysis
$PY local/turntaking/frame_rules.py --step 0.05 \
    --stems frame_attn-frame2_same_n5 frame_attn-joint_same_n5 \
    > tt_analysis/frame_rules_joint.txt 2>&1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
