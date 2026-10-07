#!/usr/bin/env bash
# 停止規則を 3 シードで確かめる。
#
# 「完了確率 τ で確定する」規則が資料 11 節の推奨（単一 τ ＋ 経過時間ゲート）を
# 上回ることは seed30 の 1 本でしか測っていない（資料 11 節の但し書きと同じ状況）。
# 学習済みの seed1 / seed2 について無音込み前向きグリッドのダンプを取り、
# 3 シードで比較表を作る。追加学習は無く、推論とオフライン解析だけ。
#
#   nohup ./run_seed_stoprules.sh > tt_batch_logs/seed_stoprules.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
# 切り離して起動すると conda activate が効かないので PATH を明示する
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
echo "python3 = $(command -v python3)"

for sfx in -trunc-new -trunc-new-seed1 -trunc-new-seed2; do
    echo "===== [$(date '+%m-%d %H:%M')] グリッド ${sfx} ====="
    withsil=1 tag_suffix="${sfx}" pool=attn scope=same N=5 \
        sets="eval train_dev" ./run_forward_grid.sh || echo "  失敗した点がある: ${sfx}"
done

echo "===== [$(date '+%m-%d %H:%M')] 比較表 ====="
mkdir -p tt_analysis
for sfx in -trunc-new -trunc-new-seed1 -trunc-new-seed2 -trunc03 -tail; do
    stem="pool_attn${sfx}_same_n5"
    g="0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0"
    [ "${sfx}" = "-tail" ] && g="0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 1.75 2.0 2.25 2.5 2.75 3.0"
    $PY local/turntaking/anticipate_commit.py --stem "${stem}" --grid "${g}" --max_wait 0.25 \
        > "tt_analysis/anticipate${sfx}_eval_mw0.25.txt" 2>&1 \
        && echo "  OK ${stem}" || echo "  失敗 ${stem}"
done
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
