#!/usr/bin/env bash
# フレーム単位学習：波形を切らず、tail を足して 1 回符号化し、窓内 8 か所で読む。
#   nohup ./run_frame_train.sh > tt_batch_logs/frame_train.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
echo "python3 = $(command -v python3)"

echo "===== [$(date '+%m-%d %H:%M')] 1. 学習 ====="
if ! ls -d exp/asr_*-turntaking-xfmr-pool_attn-frame-same_n5_p30 >/dev/null 2>&1; then
    tail=1 tag_suffix=-frame \
        base_cfg=myconf/train_asr_turntaking_xfmr_pool_attn_frame.yaml \
        pool=attn scope=same N=5 max_past=30 pureasr_tag=20260713-pureasr \
        ./run_xfmr_pool.sh || { echo "学習に失敗"; exit 1; }
else
    echo "学習済みなので飛ばす"
fi

echo "===== [$(date '+%m-%d %H:%M')] 2. フレーム単位ダンプ ====="
tag_suffix=-frame sets="eval train_dev" ./run_frame_dump.sh || echo "ダンプに失敗"

echo "===== [$(date '+%m-%d %H:%M')] 3. 比較 ====="
mkdir -p tt_analysis
$PY local/turntaking/frame_vs_grid.py --stem pool_attn-frame_same_n5 \
    > tt_analysis/frame_train_vs_grid.txt 2>&1 || echo "  （グリッドが無いので一部は出ない）"
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
