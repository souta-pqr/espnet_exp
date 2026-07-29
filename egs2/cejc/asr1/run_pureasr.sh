#!/usr/bin/env bash
# Stage1: 現発話のみで純粋ASR(transducer)を学習（区間末損失なし＝turntaking_weight:0）。
# ここで得たエンコーダを Stage2 で凍結して使う。
set -e; set -u; set -o pipefail
train_set=train_nodup; valid_set=train_dev; test_sets="eval"
asr_config=myconf/train_asr_pureasr_conformer.yaml
asr_tag=${asr_tag:-$(date +%Y%m%d)-pureasr}
asr_stats_dir=exp/asr_stats_raw_jp_word_pureasr
inference_config=${inference_config:-myconf/decode_cbs_transducer_bounded.yaml}
# past_speech/ctx_vec は使わない（誤配線防止）
for d in "${train_set}" "${valid_set}" ${test_sets}; do
  for f in past_speech ctx_vec past_vec; do
    [ -f "dump/raw/${d}/${f}.scp" ] && mv "dump/raw/${d}/${f}.scp" "dump/raw/${d}/${f}.scp.off" || true
  done
done
./asr.sh --stage 10 --asr_stats_dir "${asr_stats_dir}" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml --asr_config "${asr_config}" --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth "$@"
