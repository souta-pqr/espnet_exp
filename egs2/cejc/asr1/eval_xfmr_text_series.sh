#!/usr/bin/env bash
#   scope=same NS="1 2 3 4 5" ./eval_xfmr_text_series.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
scope=${scope:-same}; max_past=${max_past:-30}; ckpt=${ckpt:-valid.tag_acc.ave.pth}; NS=${NS:-1 2 3 4 5}
mkdir -p tt_eval_logs
for n in $NS; do
    variant="${scope}_n${n}_p${max_past}"
    d=$(ls -d exp/asr_*-turntaking-xfmr-text-${scope}_n${n}_p${max_past} 2>/dev/null | tail -1 || true)
    ps="dump/raw/eval/past_text_${variant}.scp"
    { [ -z "$d" ] || [ ! -f "$d/$ckpt" ] || [ ! -f "$ps" ]; } && { echo "[skip] N=${n}"; continue; }
    out="tt_eval_logs/text_${scope}_n${n}_${ckpt%.pth}.log"
    if [ ! -f "$out" ] || ! grep -q macro-F1 "$out"; then
        echo "===== eval text ${scope} N=${n} ====="
        $PY local/turntaking/evaluate_multitask.py --config "$d/config.yaml" --model "$d/$ckpt" \
            --data_dir dump/raw/eval --past_text --past_text_scp "$ps" > "$out" 2>&1
    fi
done
echo; echo "================ text / scope=${scope} ================"
echo "| N | 継続F1 | 完了F1 | 相槌F1 | macro-F1 | **二値分離** | CER |"
echo "| --- | --- | --- | --- | --- | --- | --- |"
for n in $NS; do
    out="tt_eval_logs/text_${scope}_n${n}_${ckpt%.pth}.log"; [ -f "$out" ] || continue
    ke=$(sed -n 's/.*<継続>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    kn=$(sed -n 's/.*<終了>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    ai=$(sed -n 's/.*<相槌>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    mf=$(sed -n 's/.*macro-F1 = \([0-9.]*\).*/\1/p' "$out"|head -1)
    bs=$(sed -n 's/.*二値分離 acc = \([0-9.]*\).*/\1/p' "$out"|head -1)
    d=$(ls -d exp/asr_*-turntaking-xfmr-text-${scope}_n${n}_p${max_past} 2>/dev/null | tail -1)
    cer=$(awk -F'|' '/^\|decode/ && /\/eval\|/ {if ($4+0>w){w=$4+0;e=$9}} END{if(e!="")print e}' "$d/RESULTS.md" 2>/dev/null)
    echo "| $n | ${ke:--} | ${kn:--} | ${ai:--} | ${mf:--} | **${bs:--}** | ${cer:--} |"
done
