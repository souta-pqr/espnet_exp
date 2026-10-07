#!/usr/bin/env bash
# eval をデコードして CER を測る（stage 12-13）。
# 学習は 80ep で patience 打ち切り済み。valid.loss.ave.pth は生成されている。
# （tag_acc の平均だけ 49epoch.pth 欠落で落ちたが、純粋 ASR では使わない）
#
# cron から起動する。user@1609.service の外で走るので systemd-oomd に殺されない。
# 最初に自分の crontab を消して一度きりにする。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH

echo "===== [$(date '+%m-%d %H:%M')] eval をデコードして CER を測る ====="
echo "cgroup: $(cat /proc/self/cgroup)"
./asr.sh --stage 12 --stop_stage 13 \
    --asr_stats_dir exp/asr_stats_raw_jp_word_pureasr --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --use_context_inputs false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml \
    --asr_config myconf/train_asr_pureasr_conformer_ext.yaml \
    --asr_tag 20260918-pureasr-ext \
    --inference_config myconf/decode_cbs_transducer_bounded.yaml \
    --train_set train_nodup --valid_set train_dev --test_sets eval \
    --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth \
  && echo "===== [$(date '+%m-%d %H:%M')] デコード 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] デコード 失敗 ====="

echo "--- 旧 CER 23.3 との比較 ---"
grep -A 6 "^### CER" exp/asr_20260918-pureasr-ext/RESULTS.md 2>/dev/null | tail -2
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
