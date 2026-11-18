#!/usr/bin/env bash
# gemma-4-31B（思考あり，(e) 版）のラベルで Stage2 を学習し直して，逐次の評価をする（2026-10-21）。
#   完了・継続：gemma（LLM の区間 137,812 と，直後に別の話者で「完了」にしていた区間 148,961）
#   相槌：今の規則＋一覧の語の並び（2,781 区間を移す）
# 人手正解 1000 件との一致率：(e) 版の多数決＋規則 0.832 → gemma 0.874。
#
#  1. 今の tag（(e) 版の多数決）を tag.vote_e に退避 → gemma で書き換え → dump と train_gx の tag も揃える
#  2. Stage2 を学び直す（設定は ge の複製でクラス重みだけ変える）
#  3. フレームダンプ（区間の長さ + 5 秒．eval_g・dev_g）
#  4. 逐次の評価（eot_eval.py）：g・ge・gm と silero-vad を新しいラベルで測る
#     判定器側の無音 g の候補に 1.2・1.5 秒を足す（前回は候補の端の 1.0 秒が選ばれたため）
#
# BERT・融合・早期確定の規則は流さない（融合のテキスト側は打ち止め）。
# 旧版の結果（exp/asr_..-g-.. ・-ge-..，tt_preds，tt_analysis）は上書きしない。
# 各段の完了時に tt_batch_logs/g_stage2gm_<段>.done を置く。
# 起動：tmux の外から setsid nohup bash tt_batch_logs/run_g_stage2gm.sh >> tt_batch_logs/g_stage2gm.log 2>&1 &
set -u
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kobori/.conda/envs/espnet/bin/python
M=tt_batch_logs/g_stage2gm
CFG0=myconf/.gen_pool_attn-frame2-blk42xh2-ge_same_n5_p30.yaml
CFG=myconf/.gen_pool_attn-frame2-blk42xh2-gm_same_n5_p30.yaml
TAG=20261021-turntaking-xfmr-pool_attn-frame2-blk42xh2-gm-same_n5_p30
STATS=exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail
TR=train_gx
P=tt_preds/frame_attn-frame2-blk42xh2

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }

echo; log "G_STAGE2GM 開始（gemma のラベル）"
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 1. gemma でラベルを書き換える ----
if ! done_ tag; then
  log "gemma でラベルを書き換える"
  $PY local/turntaking/gemma_tags.py --write || { echo "書き換えに失敗"; exit 1; }
  for ds in dev_g eval_g; do
    [ -e "dump/raw/$ds/tag.vote_e" ] || cp "dump/raw/$ds/tag" "dump/raw/$ds/tag.vote_e"
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
  for f in "data/$TR/tag" "dump/raw/$TR/tag"; do [ -e "$f.vote_e" ] || cp "$f" "$f.vote_e"; done
  cat data/train_g/tag data/train_g_extra/tag | sort > "data/$TR/tag"
  utils/filter_scp.pl "dump/raw/$TR/wav.scp" "data/$TR/tag" > "dump/raw/$TR/tag"
  [ "$(wc -l < "dump/raw/$TR/tag")" = "$(wc -l < "dump/raw/$TR/wav.scp")" ] || { echo "train_gx の tag の行数が合わない"; exit 1; }
  for ds in "$TR" dev_g eval_g; do
    echo "$ds: (e) 版の多数決から変わった発話 $(paste -d' ' "dump/raw/$ds/tag.vote_e" "dump/raw/$ds/tag" | awk '$2 != $4' | wc -l)"
  done
  mark tag
fi
W=$($PY -c "
import collections
c = collections.Counter(l.split()[1] for s in ('train_g', 'train_g_extra') for l in open(f'data/{s}/tag'))
print(f\"{c['1']/c['0']:.2f} 1.0 {c['1']/c['2']:.2f}\")")
echo "クラス重み [継続 完了 相槌] = $W"
if [ ! -e "$CFG" ]; then
  cp "$CFG0" "$CFG"
  sed -i -E "s/^(\s+)tag_class_weight: \[[0-9., ]+\].*/\1tag_class_weight: [$(echo $W | sed 's/ /, /g')]   # [継続, 完了, 相槌]。gemma のラベルの train_g + train_g_extra の割合（完了÷各クラス）（2026-10-21）/" "$CFG"
fi
grep -n "tag_class_weight" "$CFG"

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
  tag_suffix=-frame2-blk42xh2-gm post=5.0 sets="eval_g dev_g" ./run_frame_dump.sh
  for ds in eval_g dev_g; do
    [ -s "${P}-gm_same_n5_${ds}_valid.loss.ave_post5.0.tsv" ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  mark dump
fi

# ---- 4. 逐次の評価（新しいラベルで，g・ge・gm と silero-vad） ----
if ! done_ eot; then
  common=(--name g ge gm --window 5 --free_sec 2 --sustain 0
          --gates "0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.8 1.0 1.2 1.5" --fb_gates "0.6 0.8 1.0 1.2 1.5"
          --fallback "1.0 1.5 2.0 2.5 3.0" --fb_bc "1.01 0.3 0.1")
  dumps() { echo "${P}-g_same_n5_$1_valid.loss.ave_post5.0.tsv ${P}-ge_same_n5_$1_valid.loss.ave_post5.0.tsv ${P}-gm_same_n5_$1_valid.loss.ave_post5.0.tsv"; }
  log "逐次の評価（dev_g：曲線）"
  $PY local/turntaking/eot_eval.py --data data/dev_g --dump $(dumps dev_g) --silero tt_analysis/vad_silero/dev_g \
      "${common[@]}" --out tt_analysis/eot_eval_dev_gm_w5 > tt_analysis/eot_eval_dev_gm_w5.log 2>&1 || { echo "dev の評価に失敗"; exit 1; }
  log "逐次の評価（eval_g：dev で選んだ点．cut-in rate 5% 以下・miss rate 10% 以下）"
  $PY local/turntaking/eot_eval.py --data data/eval_g --dump $(dumps eval_g) --silero tt_analysis/vad_silero/eval_g \
      "${common[@]}" --out tt_analysis/eot_eval_eval_gm_w5_miss10 --pick_from tt_analysis/eot_eval_dev_gm_w5 \
      --pick_miss 0.10 --pair gm+silero+fb silero > tt_analysis/eot_eval_eval_gm_w5_miss10.log 2>&1 || { echo "eval の評価に失敗"; exit 1; }
  cat tt_analysis/eot_eval_eval_gm_w5_miss10/pick.txt
  mark eot
fi

log "G_STAGE2GM 完了"
