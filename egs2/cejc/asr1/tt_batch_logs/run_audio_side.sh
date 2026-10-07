#!/usr/bin/env bash
# 音声側の強化 3 本を順に回す（SESSION_HANDOFF_20260930 §3 の 2 番）。GPU 1 枚なので直列。
#
#  A. frame2-blk42 を 30 → 40 エポックに延長（30ep の checkpoint から再開）
#  B. ヘッドを Transformer 2 層に（Stage1 blk42 から学習し直し、30ep）
#  C. block_size 42 → 60（C1: Stage1 を学習し直し → eval/train_dev_dec をデコード、
#                          C2: Stage2 frame2 を学習）
#
# 各モデルは フレームダンプ → 停止規則（音声のみ）→ 融合 まで通して判断する
# （代理指標の valid loss で判断しない）。比較相手は frame2-blk42（音声のみ 0.7354 / 融合 0.7537）。
#
# 各段の完了時に tt_batch_logs/audio_side_<段>.done を置く。途中で落ちたら直して再投入すれば
# 済んだ段は飛ばす。学習が CUDA OOM で落ちたら batch_bins 半分・accum_grad 倍（実効バッチ同一）で
# 最大 2 回やり直す。
#
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kobori/.conda/envs/espnet/bin/python
M=tt_batch_logs/audio_side

BASE_TAG=20260924-turntaking-xfmr-pool_attn-frame2-blk42-same_n5_p30
BASE_STEM=frame_attn-frame2-blk42_same_n5
ASR42=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
S1_60=20260930-pureasr-blk60-sp
ASR60=exp/asr_$S1_60/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
TT_STATS=exp/asr_stats_raw_jp_word_pool_attn_same_n5_p30_tail

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "AUDIO_SIDE 開始"
echo "cgroup: $(cat /proc/self/cgroup)"

tt_args=(--asr_stats_dir "$TT_STATS" --ngpu 1
         --use_streaming false --use_disfluency_detection false
         --use_turntaking_detection true --use_multitask_transducer false
         --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word
         --lm_config conf/train_lm.yaml
         --inference_config myconf/decode_cbs_transducer_bounded.yaml
         --train_set train_nodup --valid_set train_dev --test_sets eval
         --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false
         --inference_asr_model valid.loss.ave.pth)
s1_args=(--asr_stats_dir exp/asr_stats_raw_jp_word_pureasr_sp --ngpu 1
         --speed_perturb_factors "0.9 1.0 1.1"
         --use_streaming false --use_disfluency_detection false
         --use_turntaking_detection true --use_multitask_transducer false
         --use_context_inputs false
         --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word
         --lm_config conf/train_lm.yaml
         --inference_config myconf/decode_cbs_transducer_bounded.yaml
         --train_set train_nodup --valid_set train_dev --test_sets eval
         --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false
         --inference_asr_model valid.loss.ave.pth)

# 学習（stage 11）。OOM なら batch_bins 半分・accum_grad 倍で最大 2 回やり直す。
#   train <config> <tag> <args 配列名>
train() {
  local cfg=$1 tag=$2; local -n a=$3
  local try
  for try in 0 1 2; do
    ./asr.sh --stage 11 --stop_stage 11 "${a[@]}" --asr_config "$cfg" --asr_tag "$tag" \
      && [ -s "exp/asr_$tag/valid.loss.ave.pth" ] && return 0
    if tail -n 300 "exp/asr_$tag/train.log" 2>/dev/null | grep -q "CUDA out of memory"; then
      [ "$try" = 2 ] && break
      local bb ag
      bb=$(awk '/^batch_bins:/{print $2}' "$cfg"); ag=$(awk '/^accum_grad:/{print $2}' "$cfg")
      echo "CUDA OOM → batch_bins $bb→$((bb / 2)) / accum_grad $ag→$((ag * 2)) でやり直す"
      sed -i -E "s/^batch_bins: $bb/batch_bins: $((bb / 2))/; s/^accum_grad: $ag/accum_grad: $((ag * 2))/" "$cfg"
      mv "exp/asr_$tag" "exp/asr_$tag.oom-bak$try"
    else
      break
    fi
  done
  echo "学習失敗: $tag"; return 1
}

# Stage2 の評価。 evaluate <段> <tag_suffix> <stem> <ASR デコード dir>
evaluate() {
  local step=$1 suf=$2 stem=$3 asr=$4
  done_ "${step}_eval" && { echo "評価済み: $step"; return 0; }
  log "$step: フレームダンプ"
  tag_suffix=$suf pool=attn scope=same N=5 max_past=30 sets="eval train_dev" ./run_frame_dump.sh
  for ds in eval train_dev; do
    [ -s "tt_preds/${stem}_${ds}_valid.loss.ave.tsv" ] || { echo "ダンプ出力なし: $stem $ds"; return 1; }
  done
  log "$step: 停止規則（音声のみ）"
  $PY local/turntaking/frame_rules.py --step 0.05 --stems "$BASE_STEM" "$stem" \
      > "tt_analysis/frame_rules_${step}.txt" 2>&1 || { echo "停止規則に失敗"; return 1; }
  log "$step: 融合"
  $PY local/turntaking/frame_fusion.py --stem "$stem" --bert exp/text_bert_asr \
      --asr_text "$asr/eval/text" --dev_asr_text "$asr/train_dev_dec/text" \
      > "tt_analysis/frame_fusion_${step}.txt" 2>&1 || { echo "融合に失敗"; return 1; }
  grep -E "音声のみ|融合（dev 選択" "tt_analysis/frame_fusion_${step}.txt" | grep "区間検出まで待つ"
  mark "${step}_eval"
}

# ---- A. 30 → 40 エポック延長 ----
A_TAG=20260930-turntaking-xfmr-pool_attn-frame2-blk42ep40-same_n5_p30
if ! done_ A_train; then
  log "A: 40 エポックに延長（30ep から再開）"
  [ -d "exp/asr_$A_TAG" ] || cp -a "exp/asr_$BASE_TAG" "exp/asr_$A_TAG" || exit 1
  train myconf/.gen_pool_attn-frame2-blk42ep40_same_n5_p30.yaml "$A_TAG" tt_args || exit 1
  mark A_train
fi
evaluate A -frame2-blk42ep40 frame_attn-frame2-blk42ep40_same_n5 "$ASR42" || exit 1

# ---- B. Transformer 2 層ヘッド ----
B_TAG=20260930-turntaking-xfmr-pool_attn-frame2-blk42xh2-same_n5_p30
if ! done_ B_train; then
  log "B: ヘッドを Transformer 2 層に"
  train myconf/.gen_pool_attn-frame2-blk42xh2_same_n5_p30.yaml "$B_TAG" tt_args || exit 1
  mark B_train
fi
evaluate B -frame2-blk42xh2 frame_attn-frame2-blk42xh2_same_n5 "$ASR42" || exit 1

# ---- B40. B を 30 → 40 エポックに延長（2026-10-03 追加。A で延長が +0.0037 効いたため）----
# exp は B の複製。過去エポックの Nepoch.pth はハードリンク（共用サーバなので容量を増やさない）。
B40_TAG=20260930-turntaking-xfmr-pool_attn-frame2-blk42xh2ep40-same_n5_p30
if ! done_ B40_train; then
  log "B40: B を 40 エポックに延長（30ep から再開）"
  [ -d "exp/asr_$B40_TAG" ] || { echo "exp/asr_$B40_TAG が無い"; exit 1; }
  train myconf/.gen_pool_attn-frame2-blk42xh2ep40_same_n5_p30.yaml "$B40_TAG" tt_args || exit 1
  mark B40_train
fi
evaluate B40 -frame2-blk42xh2ep40 frame_attn-frame2-blk42xh2ep40_same_n5 "$ASR42" || exit 1

# ---- C1. Stage1 blk60 ----
if ! done_ C1_train; then
  log "C1: Stage1 blk60 学習"
  train myconf/train_asr_pureasr_conformer_blk60_sp.yaml "$S1_60" s1_args || exit 1
  mark C1_train
fi
if ! done_ C1_decode; then
  log "C1: eval / train_dev_dec デコード"
  ./asr.sh --stage 12 --stop_stage 13 "${s1_args[@]}" \
      --asr_config myconf/train_asr_pureasr_conformer_blk60_sp.yaml --asr_tag "$S1_60" \
      --test_sets "eval train_dev_dec" || { echo "デコード失敗"; exit 1; }
  for ds in eval train_dev_dec; do [ -s "$ASR60/$ds/text" ] || { echo "デコード出力なし $ds"; exit 1; }; done
  echo "--- blk42-sp: CER 20.7 / WER 27.4 との比較 ---"
  grep -A4 -E "^### (CER|WER)" "exp/asr_$S1_60/RESULTS.md" | grep -E "eval|###"
  mark C1_decode
fi

# ---- C2. Stage2 blk60 ----
C2_TAG=20260930-turntaking-xfmr-pool_attn-frame2-blk60-same_n5_p30
if ! done_ C2_train; then
  log "C2: Stage2 frame2 blk60 学習"
  train myconf/.gen_pool_attn-frame2-blk60_same_n5_p30.yaml "$C2_TAG" tt_args || exit 1
  mark C2_train
fi
evaluate C2 -frame2-blk60 frame_attn-frame2-blk60_same_n5 "$ASR60" || exit 1

log "AUDIO_SIDE 完了"
