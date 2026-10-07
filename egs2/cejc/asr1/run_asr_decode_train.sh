#!/usr/bin/env bash
# 学習データ側の ASR 認識結果を作る。
#
# 音声＋テキスト併用を「後段の確率平均」でなく「共同学習」で行うには、
# 学習データにも認識結果が要る。あわせて
#   ・BERT を認識結果で学習し直せる（現在は正解書き起こしで学習して認識結果に当てている）
#   ・混合重みを dev で選べる（現在は eval 上で掃引していたため等重みを報告している）
# が可能になる。
#
#   nohup ./run_asr_decode_train.sh > tt_batch_logs/asr_decode_train.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
echo "python3 = $(command -v python3)"

# past_* はデコードに不要（誤配線防止）
for d in train_nodup train_dev; do
  for f in past_speech past_bounds past_vec ctx_vec tail_speech; do
    [ -f "dump/raw/${d}/${f}.scp" ] && mv "dump/raw/${d}/${f}.scp" "dump/raw/${d}/${f}.scp.off" || true
  done
done

echo "===== [$(date '+%m-%d %H:%M')] デコード開始 ====="
./asr.sh --stage 12 --stop_stage 12 \
    --asr_stats_dir exp/asr_stats_raw_jp_word_pureasr --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml \
    --asr_config myconf/train_asr_pureasr_conformer.yaml \
    --asr_tag 20260713-pureasr \
    --inference_config myconf/decode_cbs_transducer_bounded.yaml \
    --train_set train_nodup --valid_set train_dev \
    --test_sets "train_dev_dec train_nodup_dec" \
    --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth \
  && echo "===== [$(date '+%m-%d %H:%M')] 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] 失敗 ====="
for d in exp/asr_20260713-pureasr/decode_*/train_dev_dec exp/asr_20260713-pureasr/decode_*/train_nodup_dec; do
  [ -f "$d/text" ] && echo "  $(wc -l < "$d/text") 行  $d/text"
done
