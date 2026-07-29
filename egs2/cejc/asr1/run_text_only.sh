#!/usr/bin/env bash
# テキストのみで発話区間末を学習・評価する（音声版 Stage2 との対照実験）。
#
# 音声版と同じヘッド(TurnTakingXfmrHead)・同じクラス重み・同じ最適化設定を
# myconf/train_asr_turntaking_xfmr.yaml から読み込むので、差分は入力modalityだけになる。
#
#   ./run_text_only.sh                       # 既定: 学習埋め込み・N=0（音声版 n0 と対応）
#   embed=frozen ./run_text_only.sh          # Stage1 ASR の decoder 埋め込みを凍結して使う
#   N=2 ./run_text_only.sh                   # 過去2発話つき（音声版 same_n2 と対応）
#   strip_tags=1 ./run_text_only.sh          # (F …)(D …) を落とす
#
# 比較対象（音声・凍結エンコーダ・N=0）: tt_eval_logs/n0_valid.tag_acc.ave.log
#   3クラス acc 0.730 / macro-F1 0.719 / 継続-終了 二値分離 0.651
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}

embed=${embed:-trainable}       # trainable | frozen
N=${N:-0}
scope=${scope:-same}
max_past=${max_past:-30}
strip_tags=${strip_tags:-0}
config=${config:-myconf/train_asr_turntaking_xfmr.yaml}
pureasr_tag=${pureasr_tag:-20260713-pureasr}

variant="textonly_${embed}_n${N}"
[ "$N" -gt 0 ] && variant="textonly_${embed}_${scope}_n${N}_p${max_past}"
[ "$strip_tags" = "1" ] && variant="${variant}_notags"
outdir=${outdir:-exp/${variant}}
LOG=tt_eval_logs; mkdir -p "$LOG" "$outdir"

opts=(--config "$config" --embed "$embed" --n_past "$N" --scope "$scope"
      --max_past_sec "$max_past" --outdir "$outdir"
      --stage1_model "exp/asr_${pureasr_tag}/valid.loss.ave.pth")
[ "$strip_tags" = "1" ] && opts+=(--strip_tags)

out="${LOG}/${variant}.log"
echo "===== テキストのみ学習: ${variant} -> ${out} ====="
$PY local/turntaking/train_text_only.py "${opts[@]}" > "$out" 2>&1

echo
sed -n '/上位 .* エポックの valid/,$p' "$out"
