#!/usr/bin/env bash
# 学習済み Stage2(xfmr) モデルのタグ性能を N ごとに評価し、表にまとめる。
#   scope=same    NS="1 2 3 4 5" ckpt=valid.tag_acc.ave.pth ./eval_xfmr_series.sh
#   scope=session NS="1 2 3 4 5" ckpt=valid.tag_acc.ave.pth ./eval_xfmr_series.sh
#   LoRA 版: tag_suffix=-lora scope=same NS="1 2 3 4 5" ./eval_xfmr_series.sh
# 学習は不要（評価のみ）。ログは tt_eval_logs/ に残る。
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
scope=${scope:-same}; max_past=${max_past:-30}; ckpt=${ckpt:-valid.tag_acc.ave.pth}
tag_suffix=${tag_suffix:-}   # LoRA 版なら "-lora"
NS=${NS:-1 2 3 4 5}
LOG=tt_eval_logs; mkdir -p "$LOG"

for n in $NS; do
    variant="${scope}_n${n}_p${max_past}_xfmr${tag_suffix}"
    d=$(ls -d exp/asr_*-turntaking-xfmr${tag_suffix}-${scope}_n${n}_p${max_past} 2>/dev/null | tail -1 || true)
    if [ -z "$d" ]; then echo "[skip] N=${n}: 学習済みモデルなし"; continue; fi
    if [ ! -f "$d/$ckpt" ]; then echo "[skip] N=${n}: $d/$ckpt なし"; continue; fi
    pv="dump/raw/eval/past_vec_${variant}.scp"
    if [ ! -f "$pv" ]; then echo "[skip] N=${n}: $pv なし"; continue; fi

    out="${LOG}/${scope}${tag_suffix}_n${n}_${ckpt%.pth}.log"
    echo "===== eval ${scope} N=${n}  (${d} / ${ckpt}) -> ${out} ====="
    $PY local/turntaking/evaluate_multitask.py \
        --config "$d/config.yaml" --model "$d/$ckpt" \
        --data_dir dump/raw/eval --xfmr --past_vec_scp "$pv" \
        > "$out" 2>&1
    grep -E "macro-F1|<継続>|<終了>|<相槌>" "$out" | sed 's/^/    /'
done

echo
echo "================ まとめ (scope=${scope}${tag_suffix}, ckpt=${ckpt}) ================"
printf "%-4s %10s %10s %10s %10s %8s\n" "N" "<継続>F1" "<完了>F1" "<相槌>F1" "macro-F1" "CER(%)"
for n in $NS; do
    out="${LOG}/${scope}${tag_suffix}_n${n}_${ckpt%.pth}.log"
    [ -f "$out" ] || continue
    ke=$(sed -n 's/.*<継続>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out" | head -1)
    kn=$(sed -n 's/.*<終了>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out" | head -1)
    ai=$(sed -n 's/.*<相槌>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out" | head -1)
    mf=$(sed -n 's/.*macro-F1 = \([0-9.]*\).*/\1/p' "$out" | head -1)
    d=$(ls -d exp/asr_*-turntaking-xfmr${tag_suffix}-${scope}_n${n}_p${max_past} 2>/dev/null | tail -1 || true)
    # RESULTS.md の |decode.../eval| 行のうち Wrd 最大（=単語単位）の Err 列を採る
    cer=$(awk -F'|' '/^\|decode/ && /\/eval\|/ {if ($4+0 > w) {w=$4+0; e=$9}} END{if (e!="") print e}' "$d/RESULTS.md" 2>/dev/null || true)
    printf "%-4s %10s %10s %10s %10s %8s\n" "$n" "${ke:--}" "${kn:--}" "${ai:--}" "${mf:--}" "${cer:--}"
done
