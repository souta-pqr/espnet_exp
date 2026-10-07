#!/usr/bin/env bash
# フレーム単位学習（検証条件も揃えた版）。
#   nohup ./run_frame2_train.sh > tt_batch_logs/frame2_train.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
echo "python3 = $(command -v python3)"

echo "===== [$(date '+%m-%d %H:%M')] 1. 学習 ====="
if ! ls -d exp/asr_*-turntaking-xfmr-pool_attn-frame2-same_n5_p30 >/dev/null 2>&1; then
    tail=1 tag_suffix=-frame2 \
        base_cfg=myconf/train_asr_turntaking_xfmr_pool_attn_frame2.yaml \
        pool=attn scope=same N=5 max_past=30 pureasr_tag=20260713-pureasr \
        ./run_xfmr_pool.sh || { echo "学習に失敗"; exit 1; }
else
    echo "学習済みなので飛ばす"
fi

echo "===== [$(date '+%m-%d %H:%M')] 2. フレーム単位ダンプ ====="
tag_suffix=-frame2 sets="eval train_dev" ./run_frame_dump.sh || echo "ダンプに失敗"

echo "===== [$(date '+%m-%d %H:%M')] 3. 比較 ====="
mkdir -p tt_analysis
$PY local/turntaking/frame_rules.py --grid_points "0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0" \
    --stems frame_attn-trunc-new_same_n5 frame_attn-frame_same_n5 frame_attn-frame2_same_n5 \
    > tt_analysis/frame_rules_frame2.txt 2>&1
$PY local/turntaking/frame_rules.py --step 0.05 \
    --stems frame_attn-trunc-new_same_n5 frame_attn-frame_same_n5 frame_attn-frame2_same_n5 \
    >> tt_analysis/frame_rules_frame2.txt 2>&1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
