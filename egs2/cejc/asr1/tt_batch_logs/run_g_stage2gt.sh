#!/usr/bin/env bash
# 判定器の学習範囲を「区間末 +0.75 秒」から「+2 秒」に広げて学び直す（gt，2026-10-24）。
#   ねらい：長く待たせる区間の多くは，無音はすぐ来たのに P(完了) が 0.5〜0.75 で止まる（tt_analysis/longlat_20261024）。
#           学習で区間末の後 0.75 秒までしか見せていないため，「無音が長く続くほど完了らしい」を学べていない。
#   ラベル・入力・凍結部は gm と同じ．gm の重みから続けて学習する（gm は 12 エポック目以降，開発セットで良くなっていない）。
#
#  1. 区間末の後の音声（tail）を 2 秒で作り直す（train_gx・dev_g．0.75 秒のファイルを上書き．gm 以前のモデルは先頭 0.75 秒だけを読む）
#  2. Stage2 学習（myconf/.gen_pool_attn-frame2-blk42xh2-gt_same_n5_p30.yaml）
#  3. フレームダンプ（区間の長さ + 5 秒．eval_g・dev_g）
# 評価（eot_eval・two_stage・eval_all）は終わってから 1 本ずつ流す。
# 起動：tmux の外から setsid nohup bash tt_batch_logs/run_g_stage2gt.sh >> tt_batch_logs/g_stage2gt.log 2>&1 &
set -u
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kobori/.conda/envs/espnet/bin/python
M=tt_batch_logs/g_stage2gt
CFG=myconf/.gen_pool_attn-frame2-blk42xh2-gt_same_n5_p30.yaml
TAG=20261024-turntaking-xfmr-pool_attn-frame2-blk42xh2-gt-same_n5_p30
STATS=exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail
TR=train_gx
P=tt_preds/frame_attn-frame2-blk42xh2

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "G_STAGE2GT 開始（tail 2 秒）"
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 1. tail を 2 秒で作り直す ----
if ! done_ tail; then
  for ds in dev_g "$TR"; do
    log "tail 2 秒（$ds）"
    cp "dump/raw/$ds/tail_speech.scp" "dump/raw/$ds/tail_speech.scp.bak_075"
    nice -n 10 /usr/bin/time -f "最大メモリ %M KB  %e 秒" $PY local/build_tail_audio.py --dset "$ds" --tail_sec 2.0 --nj 8 \
      || { echo "tail の作成に失敗: $ds"; exit 1; }
    cmp -s "dump/raw/$ds/tail_speech.scp" "dump/raw/$ds/tail_speech.scp.bak_075" || { echo "tail_speech.scp の中身が変わった: $ds"; exit 1; }
  done
  mark tail
fi

# ---- 2. Stage2 学習 ----
tt_args=(--asr_stats_dir "$STATS" --ngpu 1
         --use_streaming false --use_disfluency_detection false
         --use_turntaking_detection true --use_multitask_transducer false
         --nj 16 --inference_nj 16 --lang jp_g --feats_type raw --token_type word
         --lm_config conf/train_lm.yaml
         --inference_config myconf/decode_cbs_transducer_bounded.yaml
         --train_set "$TR" --valid_set dev_g --test_sets eval_g
         --lm_train_text "data/$TR/text" --use_lm false --use_word_lm false
         --inference_asr_model valid.loss.ave.pth
         --asr_config "$CFG" --asr_tag "$TAG")

if ! done_ train; then
  log "Stage2 学習（$TAG）"
  ./asr.sh --stage 11 --stop_stage 11 "${tt_args[@]}" \
    && [ -s "exp/asr_$TAG/valid.loss.ave.pth" ] || { echo "Stage2 学習に失敗"; exit 1; }
  mark train
fi

# ---- 3. フレームダンプ（区間の長さ + 5 秒） ----
if ! done_ dump; then
  log "フレームダンプ（eval_g / dev_g，区間の長さ + 5 秒）"
  tag_suffix=-frame2-blk42xh2-gt post=5.0 sets="eval_g dev_g" ./run_frame_dump.sh
  for ds in eval_g dev_g; do
    [ -s "${P}-gt_same_n5_${ds}_valid.loss.ave_post5.0.tsv" ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  mark dump
fi

log "G_STAGE2GT 完了"
