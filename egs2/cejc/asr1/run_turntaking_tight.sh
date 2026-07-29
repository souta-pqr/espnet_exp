#!/usr/bin/env bash
# tight（区間末ぴったり・未来音声なし・trailing なし）の turn-taking マルチタスク学習。
# 既存の tight dump（dump/raw/{train_nodup,train_dev,eval}、tag 付き）を使い、
# collect-stats(stage10) → 学習(11) → デコード(12) → スコアリング(13/result.txt) を実行する。
# 音声の再 dump はしない。未来VA は config 側で off（va_weight=0）。
set -e; set -u; set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval"

asr_config=${asr_config:-myconf/train_asr_turntaking_conformer.yaml}
asr_tag=${asr_tag:-20260623-turntaking-tight}
asr_stats_dir=${asr_stats_dir:-exp/asr_stats_raw_jp_word_tight}   # tight 用 stats（trailing と分ける）
inference_config=myconf/decode_cbs_transducer.yaml
lm_config=conf/train_lm.yaml

./asr.sh                                               \
    --stage 10                                         \
    --asr_stats_dir "${asr_stats_dir}"                 \
    --ngpu 1                                           \
    --use_streaming false                              \
    --use_disfluency_detection false                   \
    --use_turntaking_detection true                    \
    --use_multitask_transducer false                   \
    --nj 16 --inference_nj 16                           \
    --lang jp --feats_type raw --token_type word        \
    --lm_config "${lm_config}"                          \
    --asr_config "${asr_config}"                        \
    --asr_tag "${asr_tag}"                              \
    --inference_config "${inference_config}"            \
    --train_set "${train_set}"                          \
    --valid_set "${valid_set}"                          \
    --test_sets "${test_sets}"                          \
    --lm_train_text "data/${train_set}/text"            \
    --use_lm false --use_word_lm false                  \
    --inference_asr_model valid.loss.ave.pth            \
    "$@"
