#!/usr/bin/env bash
# 実験1: block_size 18 -> 42（hop/look は 3/3 据え置き）
#
# 狙い: contextual_block_conformer の左文脈は past = block - hop - look で決まる
#       （contextual_block_conformer_encoder.py:254）。18/3/3 では 12 フレーム
#       ＝約 400ms しかない。42 にすると 36 フレーム ≒ 1.2 秒に伸びる一方、
#       look_ahead を触らないのでレイテンシ 100ms は変わらない。
# ベース: 20260918-pureasr-ext の 69ep（valid loss 13.051、CER 22.5）の重み。
#         block_size は重みの形状に影響しないのでそのまま流用できる。
#         ファインチューニングなので lr 0.002->0.0005、warmup 25000->5000。
#
# cron から起動する（user@1609.service の外で走らせて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH

TAG=20260920-pureasr-blk42
INIT=exp/asr_20260918-pureasr-ext/valid.loss.best.pth

common=(--asr_stats_dir exp/asr_stats_raw_jp_word_pureasr --ngpu 1
        --use_streaming false --use_disfluency_detection false
        --use_turntaking_detection true --use_multitask_transducer false
        --use_context_inputs false
        --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word
        --lm_config conf/train_lm.yaml
        --asr_config myconf/train_asr_pureasr_conformer_blk42.yaml
        --asr_tag "$TAG"
        --inference_config myconf/decode_cbs_transducer_bounded.yaml
        --train_set train_nodup --valid_set train_dev --test_sets eval
        --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false
        --inference_asr_model valid.loss.ave.pth)

echo "===== [$(date '+%m-%d %H:%M')] 実験1 block42 学習開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"
./asr.sh --stage 11 --stop_stage 11 --pretrained_model "$INIT" "${common[@]}" \
  && echo "===== [$(date '+%m-%d %H:%M')] 学習 完了 =====" \
  || { echo "===== [$(date '+%m-%d %H:%M')] 学習 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] eval をデコード ====="
./asr.sh --stage 12 --stop_stage 13 "${common[@]}" \
  && echo "===== [$(date '+%m-%d %H:%M')] デコード 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] デコード 失敗 ====="

echo "--- ベースライン CER 22.5 との比較 ---"
grep -A4 "^### CER" exp/asr_$TAG/RESULTS.md 2>/dev/null | tail -1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
