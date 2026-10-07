#!/usr/bin/env bash
# 無音込み学習の切り分け実験。
#
# 先の tail 実験は 4 つを同時に変えてしまった（tail 音声・打ち切り範囲 0.3→1.0 秒・
# 確率 0.5→0.7・先読みヘッド）。資料 6 節が「範囲を広げると発話全体付近が落ちる」と
# 報告しているので、素の精度が低かったのは範囲のせいかもしれず、
# 「無音込み学習は効かない」と断定できない。
#
# 本条件は trunc03 との差を「発話後の無音を見せるか」だけに絞る：
#   trunc03 … prob 0.5 で [−0.3s, 0] から一様、残り半分は発話末ちょうど
#   本条件  … prob 1.0 で [−0.3s, +0.3s] から一様（負側の分布は完全に一致）
#
# 先に走っている seed 検証（run_seed_stoprules.sh）の GPU 使用が終わるのを待つ。
#
#   nohup ./run_tail03_ablation.sh > tt_batch_logs/tail03_ablation.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
echo "python3 = $(command -v python3)"

echo "===== [$(date '+%m-%d %H:%M')] 0. 先行ジョブの GPU 解放を待つ ====="
while pgrep -f run_seed_stoprules.sh >/dev/null; do sleep 120; done
echo "解放された"

echo "===== [$(date '+%m-%d %H:%M')] 1. 学習（tail 単独） ====="
if ! ls -d exp/asr_*-turntaking-xfmr-pool_attn-tail03-same_n5_p30 >/dev/null 2>&1; then
    tail=1 tag_suffix=-tail03 \
        base_cfg=myconf/train_asr_turntaking_xfmr_pool_attn_tail03.yaml \
        pool=attn scope=same N=5 max_past=30 pureasr_tag=20260713-pureasr \
        ./run_xfmr_pool.sh || { echo "学習に失敗"; exit 1; }
else
    echo "学習済みなので飛ばす"
fi

echo "===== [$(date '+%m-%d %H:%M')] 2. 前向きグリッド（無音込み） ====="
withsil=1 tag_suffix=-tail03 pool=attn scope=same N=5 sets="eval train_dev" \
    ./run_forward_grid.sh || echo "失敗した点がある"

echo "===== [$(date '+%m-%d %H:%M')] 3. 比較 ====="
mkdir -p tt_analysis
$PY local/turntaking/anticipate_commit.py --stem pool_attn-tail03_same_n5 --max_wait 0.25 \
    > tt_analysis/anticipate-tail03_eval_mw0.25.txt 2>&1
$PY local/turntaking/response_tradeoff.py --max_wait 0.25 \
    --stems pool_attn-trunc03_same_n5 pool_attn-tail03_same_n5 pool_attn-tail_same_n5 \
    > tt_analysis/tradeoff_tail_ablation.txt 2>&1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
