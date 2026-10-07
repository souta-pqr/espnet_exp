#!/usr/bin/env bash
# N-best 認識結果を作る（eval と dev）。
#
# 融合の天井（正解書き起こし 0.7829）と実測（0.7429）の差は、認識結果が
# 意味的に間違っていることに起因する。門番で捨てても重みで加減しても
# 回収できない（門番の上限は +0.0025）。正しい候補が 2 位以下にいる場合を
# 拾えるのは N-best だけ。
#
#   nohup ./run_asr_nbest.sh > tt_batch_logs/asr_nbest.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
echo "python3 = $(command -v python3)"

echo "===== [$(date '+%m-%d %H:%M')] N-best デコード開始 ====="
./asr.sh --stage 12 --stop_stage 12 \
    --asr_stats_dir exp/asr_stats_raw_jp_word_pureasr --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml \
    --asr_config myconf/train_asr_pureasr_conformer.yaml \
    --asr_tag 20260713-pureasr \
    --inference_config myconf/decode_cbs_transducer_bounded_nbest5.yaml \
    --train_set train_nodup --valid_set train_dev \
    --test_sets "eval train_dev_dec" \
    --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth \
  && echo "===== [$(date '+%m-%d %H:%M')] 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] 失敗 ====="

for d in exp/asr_20260713-pureasr/decode_cbs_transducer_bounded_nbest5_*/*/; do
  for n in 1 2 3 4 5; do
    f="$d/${n}best_recog/text"
    [ -f "$f" ] && echo "  $(wc -l < "$f") 行  $f"
  done
done
