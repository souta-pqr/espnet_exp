#!/usr/bin/env bash
# 実験2b: 実験2（speed perturbation）の延長。
#
# 実験2 は max_epoch 20 で完走したが、最終 20ep で valid loss の最良を更新して
# 終わった（11.582）。まだ下げ余地があるとみて max_epoch を 40 に上げ、
# exp/asr_20260921-pureasr-blk42-sp/checkpoint.pth（20ep）から resume する。
# 打ち切りは patience 10 任せ。
#
# dump（train_nodup_sp）と stats（asr_stats_raw_jp_word_pureasr_sp）は作成済みなので
# stage 11 から流す。20ep 時点の結果（CER 20.8 / WER 27.6）は RESULTS.md.20ep-bak に退避済み。
#
# cron から起動する（user@1609.service の外で走らせて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH

TAG=20260921-pureasr-blk42-sp
STATS=exp/asr_stats_raw_jp_word_pureasr_sp

echo "===== [$(date '+%m-%d %H:%M')] 実験2b sp 延長（20ep から resume）開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

common=(--asr_stats_dir "$STATS" --ngpu 1
        --speed_perturb_factors "0.9 1.0 1.1"
        --use_streaming false --use_disfluency_detection false
        --use_turntaking_detection true --use_multitask_transducer false
        --use_context_inputs false
        --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word
        --lm_config conf/train_lm.yaml
        --asr_config myconf/train_asr_pureasr_conformer_blk42_sp_ext.yaml
        --asr_tag "$TAG"
        --inference_config myconf/decode_cbs_transducer_bounded.yaml
        --train_set train_nodup --valid_set train_dev --test_sets eval
        --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false
        --inference_asr_model valid.loss.ave.pth)

echo "===== [$(date '+%m-%d %H:%M')] stage 11: 学習（resume）====="
./asr.sh --stage 11 --stop_stage 11 "${common[@]}" \
  || { echo "===== [$(date '+%m-%d %H:%M')] 学習 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] stage 12-13: eval デコード ====="
./asr.sh --stage 12 --stop_stage 13 "${common[@]}" \
  && echo "===== [$(date '+%m-%d %H:%M')] デコード 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] デコード 失敗 ====="

echo "--- 20ep 時点 CER 20.8 / WER 27.6 との比較 ---"
grep -A4 "^### CER" exp/asr_$TAG/RESULTS.md 2>/dev/null | tail -1
grep -A4 "^### WER" exp/asr_$TAG/RESULTS.md 2>/dev/null | tail -1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
