#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval"

asr_config=myconf/train_asr_transducer_conformer.yaml
inference_config=myconf/decode_cbs_transducer.yaml

asr_tag=20260612-streaming-conformer

# LM settings
lm_config=conf/train_lm.yaml
use_lm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
# speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --ngpu 1                                           \
    --use_streaming false                              \
    --use_disfluency_detection false                   \
    --use_multitask_transducer false                   \
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
    --inference_asr_model valid.cer_transducer.best.pth           \
    "$@"