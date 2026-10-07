#!/usr/bin/env bash
# 新しい分割（train_g / dev_g / eval_g、docs/plan.md ⓪）の Stage1 を学習してデコードする（plan ⓪-3）。
#
#  0. データ準備（tt_batch_logs/run_gsplit_data.sh）の完了を待つ
#  1. Stage1 学習：block_size 42 ＋ 速度摂動（旧 blk42-sp と同じ設定。blk60 は CER を改善しなかった）
#  2. dump/raw/train_g を作る：train_g_sp の 1.0 倍速の行だけを抜き出す（音声は複製しない）
#  3. dev_g / eval_g / train_g をデコード（CER と、BERT・融合に使う認識結果）
#
# 各段の完了時に tt_batch_logs/g_stage1_<段>.done を置く。落ちたら直して再投入すれば済んだ段は飛ばす。
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=tt_batch_logs/g_stage1
TAG=20261006-pureasr-g-blk42-sp
CFG=myconf/train_asr_pureasr_conformer_blk42_sp_g.yaml

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "G_STAGE1 開始"
echo "cgroup: $(cat /proc/self/cgroup)"

common=(--ngpu 1 --speed_perturb_factors "0.9 1.0 1.1"
        --use_streaming false --use_disfluency_detection false
        --use_turntaking_detection true --use_multitask_transducer false
        --use_context_inputs false
        --nj 16 --inference_nj 16 --lang jp_g --feats_type raw --token_type word
        --lm_config conf/train_lm.yaml
        --inference_config myconf/decode_cbs_transducer_bounded.yaml
        --train_set train_g --valid_set dev_g --test_sets eval_g
        --lm_train_text data/train_g/text --use_lm false --use_word_lm false
        --asr_stats_dir exp/asr_stats_raw_jp_g_word_sp
        --inference_asr_model valid.loss.ave.pth)

# ---- 0. データ準備の完了を待つ ----
while [ ! -e tt_batch_logs/gsplit_stage10.done ]; do
  pgrep -f "bash $HERE/tt_batch_logs/run_gsplit_dat[a].sh" >/dev/null \
    || { echo "データ準備が完了印なしで止まっている。中止"; exit 1; }
  sleep 600
done
log "データ準備の完了を確認"
for f in dump/raw/train_g_sp/tag dump/raw/dev_g/tag dump/raw/eval_g/tag data/jp_g_token_list/word/tokens.txt; do
  [ -s "$f" ] || { echo "無い: $f"; exit 1; }
done
echo "train_g_sp $(wc -l < dump/raw/train_g_sp/wav.scp) / dev_g $(wc -l < dump/raw/dev_g/wav.scp) / eval_g $(wc -l < dump/raw/eval_g/wav.scp) 発話"

# ---- 1. Stage1 学習（OOM なら batch_bins 半分・accum_grad 倍で最大 2 回）----
if ! done_ train; then
  log "Stage1 学習（$TAG）"
  ok=0
  for try in 0 1 2; do
    ./asr.sh --stage 11 --stop_stage 11 "${common[@]}" --asr_config "$CFG" --asr_tag "$TAG" \
      && [ -s "exp/asr_$TAG/valid.loss.ave.pth" ] && { ok=1; break; }
    if tail -n 300 "exp/asr_$TAG/train.log" 2>/dev/null | grep -q "CUDA out of memory" && [ "$try" != 2 ]; then
      bb=$(awk '/^batch_bins:/{print $2}' "$CFG"); ag=$(awk '/^accum_grad:/{print $2}' "$CFG")
      echo "CUDA OOM → batch_bins $bb→$((bb / 2)) / accum_grad $ag→$((ag * 2)) でやり直す"
      sed -i -E "s/^batch_bins: $bb/batch_bins: $((bb / 2))/; s/^accum_grad: $ag/accum_grad: $((ag * 2))/" "$CFG"
      mv "exp/asr_$TAG" "exp/asr_$TAG.oom-bak$try"
    else
      break
    fi
  done
  [ "$ok" = 1 ] || { echo "Stage1 学習に失敗"; exit 1; }
  mark train
fi

# ---- 2. dump/raw/train_g（1.0 倍速だけ。音声ファイルは train_g_sp のものを指す）----
if ! done_ train_g_dir; then
  log "dump/raw/train_g を作る"
  S=dump/raw/train_g_sp; D=dump/raw/train_g
  mkdir -p "$D"
  for f in wav.scp text tag utt2num_samples utt2spk; do
    grep -vE '^sp[0-9.]+-' "$S/$f" > "$D/$f"
  done
  cp "$S/feats_type" "$D/"
  [ -f "$S/audio_format" ] && cp "$S/audio_format" "$D/"
  utils/utt2spk_to_spk2utt.pl < "$D/utt2spk" > "$D/spk2utt"
  n=$(wc -l < "$D/wav.scp"); m=$(wc -l < data/train_g/segments)
  echo "train_g: $n 発話（data/train_g は $m。差は長さで除外された分）"
  for f in text tag utt2num_samples utt2spk; do
    [ "$(wc -l < "$D/$f")" = "$n" ] || { echo "行数が合わない: $f"; exit 1; }
  done
  mark train_g_dir
fi

# ---- 3. デコード（dev_g・eval_g は CER、train_g は BERT の学習用）----
if ! done_ decode; then
  log "デコード（dev_g eval_g train_g）"
  ./asr.sh --stage 12 --stop_stage 13 "${common[@]}" --asr_config "$CFG" --asr_tag "$TAG" \
      --test_sets "dev_g eval_g train_g" || { echo "デコード失敗"; exit 1; }
  DEC=exp/asr_$TAG/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
  for ds in dev_g eval_g train_g; do [ -s "$DEC/$ds/text" ] || { echo "デコード出力なし $ds"; exit 1; }; done
  grep -A6 -E "^### (CER|WER)" "exp/asr_$TAG/RESULTS.md" | grep -E "###|_g\|"
  mark decode
fi

log "G_STAGE1 完了"
