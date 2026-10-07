#!/usr/bin/env bash
# Phase 1: 新 Stage1（20260921-pureasr-blk42-sp, CER 20.7）で train_dev_dec をデコードする。
#
# 目的: 融合（local/turntaking/frame_fusion.py）は dev で重みを選び eval で報告するため、
#       eval だけでなく dev の認識結果も新 Stage1 のものに差し替える必要がある。
#       eval は学習ジョブの stage 12-13 で作成済み。
#
# cron から起動する（user@1609.service の外で走らせて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH

TAG=20260921-pureasr-blk42-sp
STATS=exp/asr_stats_raw_jp_word_pureasr_sp

echo "===== [$(date '+%m-%d %H:%M')] Phase1: train_dev_dec デコード開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

./asr.sh --stage 12 --stop_stage 13 \
    --asr_stats_dir "$STATS" --ngpu 1 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --use_context_inputs false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml \
    --asr_config myconf/train_asr_pureasr_conformer_blk42_sp_ext.yaml \
    --asr_tag "$TAG" \
    --inference_config myconf/decode_cbs_transducer_bounded.yaml \
    --train_set train_nodup --valid_set train_dev --test_sets train_dev_dec \
    --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth \
  && echo "===== [$(date '+%m-%d %H:%M')] デコード 完了 =====" \
  || { echo "===== [$(date '+%m-%d %H:%M')] デコード 失敗 ====="; exit 1; }

D=exp/asr_$TAG/decode_cbs_transducer_bounded_asr_model_valid.loss.ave/train_dev_dec
echo "--- 新 Stage1 train_dev_dec: $(wc -l < $D/text) 発話 ---"
grep -A4 "^### CER" exp/asr_$TAG/RESULTS.md 2>/dev/null | tail -2
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
