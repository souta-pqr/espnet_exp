#!/usr/bin/env bash
# (e) 版の多数決ラベルで BERT・Stage2 を学習し直して評価する（2026-10-12）。
#   票 1: 元の Qwen3-8B 思考なし（変えない）
#   票 2・3: (e) を足したプロンプトで silver13 が全件を付け直したもの（relabel/full_e/）
# 人手正解 922 件で多数決の一致率 0.794 → 0.836（docs/label_vote_20261007.md の追記）。
# 旧多数決版（run_g_stage2.sh）の学習・評価が終わってから始める（同じ data/*_g/tag を読むため）。
#
#  0. run_g_stage2.sh の終了を待つ
#  1. 今の tag（旧多数決）を tag.vote_old に退避 → (e) 版の多数決で書き換え → dump と train_gx の tag も揃える
#  2. BERT（認識結果・正解書き起こし）を学び直す（exp/text_bert_g_{asr,ref}_e）
#  3. Stage2 を学び直す（設定は旧版の複製でクラス重みだけ変える。統計・入力は旧版のものを使う）
#  4. 評価（旧版と同じ手順。出力は *_ge）
#
# 旧版の結果（exp/asr_..-g-..、tt_preds の ..-g_.. 、tt_analysis/*_g.txt）は上書きしない。
# 各段の完了時に tt_batch_logs/g_stage2e_<段>.done を置く。cron から起動。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kobori/.conda/envs/espnet/bin/python
M=tt_batch_logs/g_stage2e
S1=exp/asr_20261006-pureasr-g-blk42-sp
DEC=$S1/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
CFG0=myconf/.gen_pool_attn-frame2-blk42xh2-g_same_n5_p30.yaml
CFG=myconf/.gen_pool_attn-frame2-blk42xh2-ge_same_n5_p30.yaml
TAG=20261012-turntaking-xfmr-pool_attn-frame2-blk42xh2-ge-same_n5_p30
STEM=frame_attn-frame2-blk42xh2-ge_same_n5
STATS=exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail
BERT=exp/text_bert_g_asr_e
BERT_REF=exp/text_bert_g_ref_e
VOTE=/autofs/diamond5/share/users/kobori/relabel/full_e
TR=train_gx

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "G_STAGE2E 開始（(e) 版の多数決ラベル）"
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 0. 旧多数決版の終了を待つ ----
while pgrep -f "bash $HERE/tt_batch_logs/run_g_stage[2].sh" >/dev/null; do sleep 600; done
[ -e tt_batch_logs/g_stage2_eval.done ] || { echo "旧多数決版が評価まで終わっていない。中止"; exit 1; }
log "旧多数決版の終了を確認"

# ---- 1. (e) 版の多数決でラベルを書き換える ----
if ! done_ vote; then
  log "(e) 版の多数決でラベルを書き換える"
  for s in train_g train_g_extra dev_g eval_g; do
    [ -e "data/$s/tag.vote_old" ] || cp "data/$s/tag" "data/$s/tag.vote_old"
  done
  $PY local/turntaking/vote_tags.py --v32b "$VOTE/Qwen3-32B-AWQ-nothink.add_e.tsv" \
      --v8think "$VOTE/Qwen3-8B-think.add_e.tsv" --write || { echo "多数決に失敗"; exit 1; }
  for ds in dev_g eval_g; do
    [ -e "dump/raw/$ds/tag.vote_old" ] || cp "dump/raw/$ds/tag" "dump/raw/$ds/tag.vote_old"
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
  for f in "data/$TR/tag" "dump/raw/$TR/tag"; do [ -e "$f.vote_old" ] || cp "$f" "$f.vote_old"; done
  cat data/train_g/tag data/train_g_extra/tag | sort > "data/$TR/tag"
  utils/filter_scp.pl "dump/raw/$TR/wav.scp" "data/$TR/tag" > "dump/raw/$TR/tag"
  [ "$(wc -l < "dump/raw/$TR/tag")" = "$(wc -l < "dump/raw/$TR/wav.scp")" ] || { echo "train_gx の tag の行数が合わない"; exit 1; }
  for ds in "$TR" dev_g eval_g; do
    echo "$ds: 旧多数決から変わった発話 $(paste -d' ' "dump/raw/$ds/tag.vote_old" "dump/raw/$ds/tag" | awk '$2 != $4' | wc -l)"
  done
  mark vote
fi
W=$($PY -c "
import collections
c = collections.Counter(l.split()[1] for s in ('train_g', 'train_g_extra') for l in open(f'data/{s}/tag'))
print(f\"{c['1']/c['0']:.2f} 1.0 {c['1']/c['2']:.2f}\")")
echo "クラス重み [継続 完了 相槌] = $W"
if [ ! -e "$CFG" ]; then
  cp "$CFG0" "$CFG"
  sed -i -E "s/^(\s+)tag_class_weight: \[[0-9., ]+\]/\1tag_class_weight: [$(echo $W | sed 's/ /, /g')]/" "$CFG"
fi
grep -n "tag_class_weight" "$CFG"

# ---- 2. BERT ----
if ! done_ bert_asr; then
  log "BERT（認識結果で学習）"
  $PY local/turntaking/text_ceiling.py --seed 0 --epochs 2 --class_weight "$W" \
      --train_set "$TR" --dev_set dev_g --eval_set eval_g \
      --train_text "$DEC/$TR/text" --dev_text "$DEC/dev_g/text" --eval_text "$DEC/eval_g/text" \
      --save "$BERT" --out tt_analysis/text_bert_ge_asr.json || { echo "BERT 学習に失敗"; exit 1; }
  [ -s "$BERT/model.safetensors" ] || { echo "BERT の重みが無い"; exit 1; }
  mark bert_asr
fi
if ! done_ bert_ref; then
  log "BERT（正解書き起こしで学習・天井用）"
  $PY local/turntaking/text_ceiling.py --seed 0 --epochs 2 --class_weight "$W" \
      --train_set "$TR" --dev_set dev_g --eval_set eval_g \
      --save "$BERT_REF" --out tt_analysis/text_bert_ge_ref.json || { echo "BERT(ref) 学習に失敗"; exit 1; }
  [ -s "$BERT_REF/model.safetensors" ] || { echo "BERT(ref) の重みが無い"; exit 1; }
  mark bert_ref
fi

# ---- 3. Stage2 学習 ----
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

# ---- 4. 評価 ----
if ! done_ eval; then
  log "フレームダンプ（eval_g / dev_g）"
  tag_suffix=-frame2-blk42xh2-ge pool=attn scope=same N=5 max_past=30 sets="eval_g dev_g" ./run_frame_dump.sh
  for ds in eval_g dev_g; do
    [ -s "tt_preds/${STEM}_${ds}_valid.loss.ave.tsv" ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  log "停止規則（音声のみ）"
  $PY local/turntaking/frame_rules.py --step 0.05 --dset eval_g --stems "$STEM" \
      > tt_analysis/frame_rules_ge.txt 2>&1 || { echo "停止規則に失敗"; exit 1; }
  log "融合"
  $PY local/turntaking/frame_fusion.py --stem "$STEM" --bert "$BERT" --bert_ref "$BERT_REF" \
      --dev_set dev_g --eval_set eval_g \
      --asr_text "$DEC/eval_g/text" --dev_asr_text "$DEC/dev_g/text" \
      > tt_analysis/frame_fusion_ge.txt 2>&1 || { echo "融合に失敗"; exit 1; }
  grep -E "音声のみ|融合（dev 選択|正解書き起こし" tt_analysis/frame_fusion_ge.txt | grep "区間検出まで待つ"
  log "早期確定の規則（dev 選択・持続 3）"
  $PY local/turntaking/commit_rules.py --persist 3 --stems "$STEM" \
      --bert "$BERT" --asr_dir "$DEC" --dev_set dev_g --dev_asr_name dev_g --eval_set eval_g \
      > tt_analysis/commit_rules_ge.txt 2>&1 || echo "早期確定の規則に失敗"
  mark eval
fi

log "G_STAGE2E 完了"
