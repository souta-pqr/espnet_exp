#!/usr/bin/env bash
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval"

asr_config=myconf/train_cif_transformer.yaml
inference_config=myconf/decode_transformer.yaml

# expdir
asr_tag=0525-cif-tag-transformer-cejc-epoch150-slave-mlp

lm_config=conf/train_lm.yaml
use_lm=false

./asr.sh                                               \
    --ngpu 1                                           \
    --use_streaming false                              \
    --use_multitask_transducer false                   \
    --use_disfluency_detection false                   \
    --use_cif_detection true                           \
    --nj 16                                            \
    --inference_nj 16                                  \
    --lang jp                                          \
    --feats_type raw                                   \
    --token_type word                                  \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --asr_tag "${asr_tag}"                             \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --lm_train_text "data/${train_set}/text"           \
    --use_lm false                                     \
    --use_word_lm false                                \
    --inference_asr_model valid.cer.best.pth           \
    "$@"
