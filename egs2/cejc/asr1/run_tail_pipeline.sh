#!/usr/bin/env bash
# 無音込み ＋ 先読みヘッド（Endpoint Anticipation 型）の一括実行。
#   1. tail_speech.scp（local/build_tail_audio.py）が揃うまで待つ
#   2. 学習: attn / N=5 / warmup 2500 / 60ep ＋ tail 1.0 s ＋ horizons [0.2,0.4,0.6,0.8]
#   3. 前向きグリッド（無音込み・細かい格子）を eval / train_dev でダンプ
#   4. 停止規則の比較表を tt_analysis/anticipate_*.txt に出す
#
#   nohup ./run_tail_pipeline.sh > tt_batch_logs/tail_pipeline.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
# asr.sh は素の python3 を呼ぶので、espnet の conda 環境を PATH の先頭に置く
# （切り離して起動すると conda activate が効いておらず /opt/anaconda3 の python3 が使われた）
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
echo "python3 = $(command -v python3)"
export pool=attn scope=same N=5 max_past=30 tag_suffix=-tail
grid=${grid:-"0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 1.75 2.0 2.25 2.5 2.75 3.0"}

echo "===== [$(date '+%m-%d %H:%M')] 1. tail 音声を待つ ====="
while [ ! -s dump/raw/train_nodup/tail_speech.scp ] || [ ! -s dump/raw/train_dev/tail_speech.scp ]; do
    sleep 120
done
echo "tail 音声: $(wc -l < dump/raw/train_nodup/tail_speech.scp) / $(wc -l < dump/raw/train_dev/tail_speech.scp)"

echo "===== [$(date '+%m-%d %H:%M')] 2. 学習 ====="
if ! ls -d exp/asr_*-turntaking-xfmr-pool_attn-tail-same_n5_p30 >/dev/null 2>&1; then
    tail=1 base_cfg=myconf/train_asr_turntaking_xfmr_pool_attn_tail.yaml \
        pureasr_tag=20260713-pureasr ./run_xfmr_pool.sh || { echo "学習に失敗"; exit 1; }
else
    echo "学習済みモデルがあるので飛ばす"
fi

echo "===== [$(date '+%m-%d %H:%M')] 3. 前向きグリッド（無音込み） ====="
withsil=1 grid="${grid}" sets="eval train_dev" ./run_forward_grid.sh || echo "グリッドに失敗した点がある"

echo "===== [$(date '+%m-%d %H:%M')] 4. 停止規則の比較 ====="
mkdir -p tt_analysis
for mw in 0.25 0.5; do
    $PY local/turntaking/anticipate_commit.py --stem pool_attn-tail_same_n5 --grid "${grid}" \
        --max_wait "${mw}" > "tt_analysis/anticipate_tail_eval_mw${mw}.txt" 2>&1
    $PY local/turntaking/anticipate_commit.py --stem pool_attn-tail_same_n5 --grid "${grid}" \
        --max_wait "${mw}" --dev > "tt_analysis/anticipate_tail_dev_mw${mw}.txt" 2>&1
done
# 比較相手（打ち切り学習のみ・無音なし学習）は既存の 9 点格子ダンプで
$PY local/turntaking/anticipate_commit.py --stem pool_attn-trunc03_same_n5 --max_wait 0.25 \
    > tt_analysis/anticipate_trunc03_eval_mw0.25.txt 2>&1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
