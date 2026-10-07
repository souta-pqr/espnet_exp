#!/usr/bin/env bash
# 話者が重ならない新しい分割（data/{train_g,dev_g,eval_g}、local/cejc_groupsplit.py）のデータ準備（2026-10-05）。
#
#  0. 古い分割の C1（Stage1 blk60）の学習・デコードが終わり、C2 が止められるのを待つ
#     （dump と GPU 学習を並行させると I/O で遅くなるため）
#  1. asr.sh stage 2〜5: 速度摂動 → dump → 長さで絞る → トークン一覧
#     トークン一覧は --lang jp_g で data/jp_g_token_list に作る（旧モデルの data/jp_token_list を上書きしない）
#  2. Stage1 の統計（stage 10）
#
# 共用サーバなので容量に注意。見込みは dump 約 25〜30 GB（train_g_sp に 3 速度ぶん・dev_g・eval_g）。
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
M=tt_batch_logs/gsplit
log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "GSPLIT_DATA 開始"
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 0. 古い分割の C1 が終わるのを待つ ----
while pgrep -f "bash /home/kobori/2025/espnet/egs2/cejc/asr1/tt_batch_logs/run_audio_sid[e].sh" >/dev/null \
   || pgrep -f "turntaking_asr_trai[n]" >/dev/null || pgrep -f "asr_inferenc[e]" >/dev/null; do
  sleep 300
done
log "古い分割のジョブが終わったのを確認"
ls tt_batch_logs/audio_side_C1_*.done 2>/dev/null

common=(--ngpu 1 --speed_perturb_factors "0.9 1.0 1.1"
        --use_streaming false --use_disfluency_detection false
        --use_turntaking_detection true --use_multitask_transducer false
        --use_context_inputs false
        --nj 16 --inference_nj 16 --lang jp_g --feats_type raw --token_type word
        --lm_config conf/train_lm.yaml
        --asr_config myconf/train_asr_pureasr_conformer_blk42_sp_ext.yaml
        --inference_config myconf/decode_cbs_transducer_bounded.yaml
        --train_set train_g --valid_set dev_g --test_sets eval_g
        --lm_train_text data/train_g/text --use_lm false --use_word_lm false
        --asr_stats_dir exp/asr_stats_raw_jp_g_word_sp)

# ---- 1. 速度摂動・dump・絞り込み・トークン一覧 ----
if ! done_ stage2to5; then
  log "asr.sh stage 2〜5"
  ./asr.sh --stage 2 --stop_stage 5 "${common[@]}" || { echo "stage 2〜5 失敗"; exit 1; }
  for d in train_g_sp dev_g eval_g; do
    [ -s dump/raw/$d/wav.scp ] || { echo "dump が無い: $d"; exit 1; }
    echo "  $d: $(wc -l < dump/raw/$d/text) 発話"
  done
  [ -s data/jp_g_token_list/word/tokens.txt ] || { echo "トークン一覧が無い"; exit 1; }
  echo "  トークン数 $(wc -l < data/jp_g_token_list/word/tokens.txt)（旧 $(wc -l < data/jp_token_list/word/tokens.txt)）"
  du -sh dump/raw/org/train_g_sp dump/raw/org/dev_g dump/raw/org/eval_g
  mark stage2to5
fi

# ---- 2. Stage1 の統計 ----
if ! done_ stage10; then
  log "asr.sh stage 10（Stage1 の統計）"
  ./asr.sh --stage 10 --stop_stage 10 "${common[@]}" || { echo "stage 10 失敗"; exit 1; }
  mark stage10
fi

log "GSPLIT_DATA 完了"
