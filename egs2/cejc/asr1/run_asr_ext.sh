#!/usr/bin/env bash
# Stage1 ASR の学習を延長する（50 → 80 エポック）。
#
# 20260713-pureasr は max_epoch: 50 の上限に当たって止まっており、収束していない。
#   ・最良モデルの更新が 46・47・49 エポックと最後まで続いている
#   ・patience: 10 は一度も発動していない
#   ・valid loss は 20ep 14.381 → 47ep 13.269 と下降中
# 1 エポック 11 分 20 秒なので、30 エポック追加で約 6 時間。
#
# エンコーダは contextual_block_conformer（block 18 / hop 3 / look_ahead 3）のまま。
# CER を下げるために非ストリーミングの Conformer に替えることはしない。
#
# 既存の 20260713-pureasr は壊さない（Stage2 の init_param が参照しているため）。
# checkpoint.pth と config.yaml を新タグに複製済みで、そこから resume する。
#
# past_speech / tail_speech / text_vec などの追加入力は、このモデル
# (turntaking_weight=0 の純粋 ASR) では一切使われないのに 197,135 件分の
# flac を毎エポック読むだけで、iter_time 0.7 秒 / forward 0.019 秒（GPU ほぼ遊び）
# = 1 エポック 2 時間 50 分になっていた。--use_context_inputs false で切る。
# 旧 20260713-pureasr と同じ入力構成に戻るので 1 エポック 11 分 20 秒に戻る。
#
#   setsid nohup ./run_asr_ext.sh >> tt_batch_logs/asr_ext.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
echo "python3 = $(command -v python3)"

echo "===== [$(date '+%m-%d %H:%M')] 学習の延長を開始 ====="
./asr.sh --stage 11 --stop_stage 11 \
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
  && echo "===== [$(date '+%m-%d %H:%M')] 学習 完了 =====" \
  || { echo "===== [$(date '+%m-%d %H:%M')] 学習 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] eval をデコードして CER を測る ====="
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
