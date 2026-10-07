#!/usr/bin/env bash
# Phase 3（再々投入）: stage 11 から直接始める。
#
# 経緯:
#   1回目 06:08 起動 → collect-stats 63分 → 07:11 学習開始 → 07:20 CUDA OOM で死亡。
#   2回目 11:48 起動 → batch_bins を半減して再投入したが、run_xfmr_pool.sh は
#          stage 10 決め打ちのため同じ collect-stats を再実行し始めたので 12:11 に打ち切り。
#   collect-stats の出力は入力特徴量の形状と統計だけで block_size / batch_bins に依存しない。
#   1回目が残した exp/asr_stats_raw_jp_word_pool_attn_same_n5_p30_tail/{train,valid}（07:11）が
#   そのまま使えるので、stage 10 を飛ばして stage 11 から始める。
#
#   config は run_xfmr_pool.sh が 11:48 に生成した .gen_*.yaml をそのまま使う
#   （block_size 42 / init_param=新Stage1 / batch_bins 350000 / accum_grad 16 を確認済み）。
#
# cron から起動する（user@1609.service の外で走らせて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
# 1回目は CUDA OOM（reserved-but-unallocated が 6.84GiB ＝断片化が大きい）。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=myconf/.gen_pool_attn-frame2-blk42_same_n5_p30.yaml
TAG=20260924-turntaking-xfmr-pool_attn-frame2-blk42-same_n5_p30
STATS=exp/asr_stats_raw_jp_word_pool_attn_same_n5_p30_tail

echo "===== [$(date '+%m-%d %H:%M')] Phase3-s11: frame2 blk42 学習開始（stage 11 から）====="
echo "cgroup: $(cat /proc/self/cgroup)"
grep -E "^batch_bins|^accum_grad|block_size: 42" "$CFG" | sed 's/#.*//'

./asr.sh --stage 11 --stop_stage 11 \
    --asr_stats_dir "$STATS" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml \
    --asr_config "$CFG" --asr_tag "$TAG" \
    --inference_config myconf/decode_cbs_transducer_bounded.yaml \
    --train_set train_nodup --valid_set train_dev --test_sets eval \
    --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth \
  && echo "===== [$(date '+%m-%d %H:%M')] 学習 完了 =====" \
  || { echo "===== [$(date '+%m-%d %H:%M')] 学習 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
