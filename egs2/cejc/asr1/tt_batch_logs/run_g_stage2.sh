#!/usr/bin/env bash
# 新しい分割（train_g / dev_g / eval_g、docs/plan.md ⓪）の ラベル更新・BERT・Stage2・評価（plan ⓪-4〜6）。
# 2026-10-07 改訂（旧版は run_g_stage2.sh.v1）:
#   - ラベルを 3 モデルの多数決に差し替える（人手正解で一致率 0.762 → 0.810）。
#     票 2・3 は silver9（Qwen3-32B-AWQ 思考なし）・silver13（Qwen3-8B 思考あり）で全件に付けたもの。
#   - 漏れていた 8 会話（data/train_g_extra、4,307 発話・1.86 時間、すべて学習側）を BERT・Stage2 の学習に足す。
#     学習集合は train_gx = train_g ＋ train_g_extra（train_g そのものは書き換えない）。Stage1 には足さない。
#   - クラス重みは多数決後の train_gx の割合（完了÷各クラス）で決める。
#
#  0. Stage1（学習とデコード）と、多数決の票 2・3 がそろうのを待つ
#  1. 多数決でラベルを作り直す（data/{train_g,train_g_extra,dev_g,eval_g}/tag。元は tag.qwen）→ クラス重み
#  2. train_g_extra の音声を書き出し（0.1〜20 秒）→ Stage1 でデコード → train_gx を組む
#  3. BERT（認識結果で学習・正解書き起こしで学習）
#  4. Stage2 の入力（過去発話の音声・後続音声 0.75 秒）→ 統計 → 学習（Transformer 2 層ヘッド）
#  5. 評価：フレームダンプ → 停止規則 → 融合 → 早期確定の規則（dev 選択・持続 3）
#
# 各段の完了時に tt_batch_logs/g_stage2_<段>.done を置く。落ちたら直して再投入すれば済んだ段は飛ばす。
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kobori/.conda/envs/espnet/bin/python
M=tt_batch_logs/g_stage2
S1TAG=20261006-pureasr-g-blk42-sp
S1=exp/asr_$S1TAG
DEC=$S1/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
CFG=myconf/.gen_pool_attn-frame2-blk42xh2-g_same_n5_p30.yaml
TAG=20261006-turntaking-xfmr-pool_attn-frame2-blk42xh2-g-same_n5_p30
STEM=frame_attn-frame2-blk42xh2-g_same_n5
STATS=exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail
BERT=exp/text_bert_g_asr
BERT_REF=exp/text_bert_g_ref
VOTE=/autofs/diamond5/share/users/kobori/relabel/full
TR=train_gx

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "G_STAGE2 開始（多数決ラベル・train_gx 版）"
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 0. Stage1 と多数決の票を待つ ----
while [ ! -e tt_batch_logs/g_stage1_decode.done ]; do
  pgrep -f "bash $HERE/tt_batch_logs/run_g_stage[1].sh" >/dev/null \
    || { echo "Stage1 の処理が完了印なしで止まっている。中止"; exit 1; }
  sleep 600
done
log "Stage1 の完了を確認"
for ds in dev_g eval_g train_g; do [ -s "$DEC/$ds/text" ] || { echo "認識結果が無い: $ds"; exit 1; }; done
# 票は全件（LLM が判定する 137,812 件。silver9 の README で確認）がそろうまで待つ。
# README は実行中から自動で書かれるので、完了の合図には使わない（LINE の重複を除いた件数で判断する）。
NEED=137812
nlines() { [ -f "$1" ] && cut -f1 "$1" | sort -u | wc -l || echo 0; }
while :; do
  a=$(nlines "$VOTE/Qwen3-32B-AWQ-nothink.tsv"); b=$(nlines "$VOTE/Qwen3-8B-think.tsv")
  [ "$a" -ge "$NEED" ] && [ "$b" -ge "$NEED" ] && break
  echo "[$(date '+%m-%d %H:%M')] 票の待機中: 32B $a / 8B 思考あり $b（必要 $NEED）"
  sleep 1800
done
echo "票 32B: $a 件 / 8B 思考あり: $b 件"
log "多数決の票がそろったのを確認"

# ---- 1. 多数決でラベルを作り直す → クラス重み ----
if ! done_ vote; then
  log "多数決でラベルを作り直す"
  $PY local/turntaking/vote_tags.py --v32b "$VOTE/Qwen3-32B-AWQ-nothink.tsv" \
      --v8think "$VOTE/Qwen3-8B-think.tsv" --write || { echo "多数決に失敗"; exit 1; }
  # dump 側の tag も揃える（dev_g / eval_g は長さで絞っていないので行がそのまま対応する）
  for ds in dev_g eval_g; do
    $PY - "$ds" <<'EOF' || exit 1
import sys
ds = sys.argv[1]
t = dict(l.split() for l in open(f"data/{ds}/tag"))
rows = [l.split()[0] for l in open(f"dump/raw/{ds}/tag")]
assert all(u in t for u in rows), "dump にあって data に無い発話がある"
open(f"dump/raw/{ds}/tag", "w").writelines(f"{u} {t[u]}\n" for u in rows)
print(ds, "dump の tag を更新", len(rows))
EOF
  done
  mark vote
fi
W=$($PY -c "
import collections
c = collections.Counter(l.split()[1] for s in ('train_g', 'train_g_extra') for l in open(f'data/{s}/tag'))
print(f\"{c['1']/c['0']:.2f} 1.0 {c['1']/c['2']:.2f}\")")
echo "クラス重み [継続 完了 相槌] = $W"
sed -i -E "s/^(\s+)tag_class_weight: \[[0-9., ]+\]/\1tag_class_weight: [$(echo $W | sed 's/ /, /g')]/" "$CFG"
grep -n "tag_class_weight" "$CFG"

s1_args=(--ngpu 1 --speed_perturb_factors "0.9 1.0 1.1"
         --use_streaming false --use_disfluency_detection false
         --use_turntaking_detection true --use_multitask_transducer false
         --use_context_inputs false
         --nj 16 --inference_nj 16 --lang jp_g --feats_type raw --token_type word
         --lm_config conf/train_lm.yaml
         --inference_config myconf/decode_cbs_transducer_bounded.yaml
         --train_set train_g --valid_set dev_g
         --lm_train_text data/train_g/text --use_lm false --use_word_lm false
         --asr_stats_dir exp/asr_stats_raw_jp_g_word_sp
         --inference_asr_model valid.loss.ave.pth
         --asr_config myconf/train_asr_pureasr_conformer_blk42_sp_g.yaml --asr_tag "$S1TAG")

# ---- 2. train_g_extra の音声 → デコード → train_gx ----
if ! done_ extra; then
  log "train_g_extra の音声を書き出す"
  O=dump/raw/org/train_g_extra; D=dump/raw/train_g_extra
  scripts/audio/format_wav_scp.sh --nj 8 --cmd run.pl --audio-format flac --fs 16k \
      --segments data/train_g_extra/segments --multi-columns-input false --multi-columns-output false \
      data/train_g_extra/wav.scp "$O" || { echo "音声の書き出しに失敗"; exit 1; }
  mkdir -p "$D"
  # 学習側と同じ長さの条件（0.1〜20 秒）で絞る
  awk -v lo=1600 -v hi=320000 '$2 >= lo && $2 <= hi {print $1}' "$O/utt2num_samples" | sort > "$D/.keep"
  for f in wav.scp utt2num_samples; do utils/filter_scp.pl "$D/.keep" "$O/$f" > "$D/$f"; done
  for f in text tag utt2spk; do utils/filter_scp.pl "$D/.keep" "data/train_g_extra/$f" > "$D/$f"; done
  utils/utt2spk_to_spk2utt.pl < "$D/utt2spk" > "$D/spk2utt"
  cp dump/raw/train_g/feats_type "$D/"; [ -f dump/raw/train_g/audio_format ] && cp dump/raw/train_g/audio_format "$D/"
  echo "train_g_extra: $(wc -l < "$D/wav.scp") 発話（data は $(wc -l < data/train_g_extra/segments)）"
  log "train_g_extra を Stage1 でデコード"
  ./asr.sh --stage 12 --stop_stage 12 "${s1_args[@]}" --test_sets train_g_extra \
    || { echo "デコードに失敗"; exit 1; }
  [ -s "$DEC/train_g_extra/text" ] || { echo "デコード出力なし"; exit 1; }
  mark extra
fi
if ! done_ gx; then
  log "train_gx（train_g ＋ train_g_extra）を組む"
  mkdir -p "data/$TR" "dump/raw/$TR" "$DEC/$TR"
  for f in text tag segments utt2spk; do cat data/train_g/$f data/train_g_extra/$f | sort > "data/$TR/$f"; done
  cat data/train_g/wav.scp data/train_g_extra/wav.scp | sort -u > "data/$TR/wav.scp"
  utils/utt2spk_to_spk2utt.pl < "data/$TR/utt2spk" > "data/$TR/spk2utt"
  for f in wav.scp text utt2num_samples utt2spk; do cat dump/raw/train_g/$f dump/raw/train_g_extra/$f | sort > "dump/raw/$TR/$f"; done
  # tag は多数決後の data から取り直す（dump/raw/train_g の tag は多数決前）
  utils/filter_scp.pl "dump/raw/$TR/wav.scp" "data/$TR/tag" > "dump/raw/$TR/tag"
  utils/utt2spk_to_spk2utt.pl < "dump/raw/$TR/utt2spk" > "dump/raw/$TR/spk2utt"
  cp dump/raw/train_g/feats_type "dump/raw/$TR/"; [ -f dump/raw/train_g/audio_format ] && cp dump/raw/train_g/audio_format "dump/raw/$TR/"
  cat "$DEC/train_g/text" "$DEC/train_g_extra/text" | sort > "$DEC/$TR/text"
  n=$(wc -l < "dump/raw/$TR/wav.scp")
  for f in text tag utt2num_samples utt2spk; do
    [ "$(wc -l < "dump/raw/$TR/$f")" = "$n" ] || { echo "行数が合わない: $f"; exit 1; }
  done
  echo "$TR: dump $n 発話 / 認識結果 $(wc -l < "$DEC/$TR/text") 発話"
  mark gx
fi

# ---- 3. BERT ----
if ! done_ bert_asr; then
  log "BERT（認識結果で学習）"
  $PY local/turntaking/text_ceiling.py --seed 0 --epochs 2 --class_weight "$W" \
      --train_set "$TR" --dev_set dev_g --eval_set eval_g \
      --train_text "$DEC/$TR/text" --dev_text "$DEC/dev_g/text" --eval_text "$DEC/eval_g/text" \
      --save "$BERT" --out tt_analysis/text_bert_g_asr.json || { echo "BERT 学習に失敗"; exit 1; }
  [ -s "$BERT/model.safetensors" ] || { echo "BERT の重みが無い"; exit 1; }
  mark bert_asr
fi
if ! done_ bert_ref; then
  log "BERT（正解書き起こしで学習・天井用）"
  $PY local/turntaking/text_ceiling.py --seed 0 --epochs 2 --class_weight "$W" \
      --train_set "$TR" --dev_set dev_g --eval_set eval_g \
      --save "$BERT_REF" --out tt_analysis/text_bert_g_ref.json || { echo "BERT(ref) 学習に失敗"; exit 1; }
  [ -s "$BERT_REF/model.safetensors" ] || { echo "BERT(ref) の重みが無い"; exit 1; }
  mark bert_ref
fi

# ---- 4. Stage2 の入力・統計・学習 ----
if ! done_ inputs; then
  for ds in "$TR" dev_g eval_g; do
    log "Stage2 の入力: $ds"
    $PY local/build_past_audio.py --seg_dir "data/$ds" --dump_dir "dump/raw/$ds" \
        --scope same --n_past 5 --max_past_sec 30 --out_name past_speech_same_n5_p30 \
      || { echo "過去発話の音声に失敗: $ds"; exit 1; }
    cp "dump/raw/$ds/past_speech_same_n5_p30.scp" "dump/raw/$ds/past_speech.scp"
    cp "dump/raw/$ds/past_speech_same_n5_p30_bounds.scp" "dump/raw/$ds/past_bounds.scp"
    $PY local/build_tail_audio.py --dset "$ds" --tail_sec 0.75 --nj 16 \
      || { echo "後続音声に失敗: $ds"; exit 1; }
    n=$(wc -l < "dump/raw/$ds/wav.scp")
    for f in past_speech.scp past_bounds.scp tail_speech.scp; do
      echo "  $f $(wc -l < "dump/raw/$ds/$f") 行（wav.scp $n 行）"
    done
  done
  du -sh "dump/raw/org/$TR/tail" dump/raw/org/dev_g/tail dump/raw/org/eval_g/tail
  mark inputs
fi

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

if ! done_ stats; then
  log "Stage2 の統計（stage 10）"
  ./asr.sh --stage 10 --stop_stage 10 "${tt_args[@]}" || { echo "統計に失敗"; exit 1; }
  mark stats
fi
if ! done_ train; then
  log "Stage2 学習（$TAG）"
  ok=0
  for try in 0 1 2; do
    ./asr.sh --stage 11 --stop_stage 11 "${tt_args[@]}" \
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
  [ "$ok" = 1 ] || { echo "Stage2 学習に失敗"; exit 1; }
  mark train
fi

# ---- 5. 評価 ----
if ! done_ eval; then
  log "フレームダンプ（eval_g / dev_g）"
  tag_suffix=-frame2-blk42xh2-g pool=attn scope=same N=5 max_past=30 sets="eval_g dev_g" ./run_frame_dump.sh
  for ds in eval_g dev_g; do
    [ -s "tt_preds/${STEM}_${ds}_valid.loss.ave.tsv" ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  log "停止規則（音声のみ）"
  $PY local/turntaking/frame_rules.py --step 0.05 --dset eval_g --stems "$STEM" \
      > tt_analysis/frame_rules_g.txt 2>&1 || { echo "停止規則に失敗"; exit 1; }
  log "融合"
  $PY local/turntaking/frame_fusion.py --stem "$STEM" --bert "$BERT" --bert_ref "$BERT_REF" \
      --dev_set dev_g --eval_set eval_g \
      --asr_text "$DEC/eval_g/text" --dev_asr_text "$DEC/dev_g/text" \
      > tt_analysis/frame_fusion_g.txt 2>&1 || { echo "融合に失敗"; exit 1; }
  grep -E "音声のみ|融合（dev 選択|正解書き起こし" tt_analysis/frame_fusion_g.txt | grep "区間検出まで待つ"
  log "早期確定の規則（dev 選択・持続 3）"
  $PY local/turntaking/commit_rules.py --persist 3 --stems "$STEM" \
      --bert "$BERT" --asr_dir "$DEC" --dev_set dev_g --dev_asr_name dev_g --eval_set eval_g \
      > tt_analysis/commit_rules_g.txt 2>&1 || echo "早期確定の規則に失敗"
  mark eval
fi

log "G_STAGE2 完了"
