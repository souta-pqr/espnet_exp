#!/usr/bin/env bash
# 適応強さ掃引の結果をまとめる（タグ性能＋CER）。
#   RATIOS="0.05 0.1 0.2 0.4" ./eval_adapter_ratio_sweep.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
ckpt=${ckpt:-valid.tag_acc.ave.pth}
for s in ${RATIOS:-0.05 0.1 0.2 0.4}; do
    key=$(echo "$s" | sed 's/0\.//; s/\.//')
    d=$(ls -d exp/asr_*-turntaking-xfmr-adapter_r${key}-n0 2>/dev/null | tail -1 || true)
    if [ -z "$d" ] || [ ! -f "$d/$ckpt" ]; then echo "[skip] s=${s}: 未学習"; continue; fi
    out="tt_eval_logs/n0-adapter_r${key}_${ckpt%.pth}.log"
    if [ ! -f "$out" ]; then
        echo "===== eval s=${s} ($d) ====="
        tag_suffix="-adapter_r${key}" ckpt="$ckpt" ./eval_xfmr_n0.sh >/dev/null
    fi
done

echo
printf "%-10s %8s %8s %8s %9s %10s %7s\n" "適応強さ" "継続F1" "完了F1" "相槌F1" "macroF1" "継続/完了" "CER"
for s in ${RATIOS:-0.05 0.1 0.2 0.4}; do
    key=$(echo "$s" | sed 's/0\.//; s/\.//')
    out="tt_eval_logs/n0-adapter_r${key}_${ckpt%.pth}.log"
    [ -f "$out" ] || continue
    ke=$(sed -n 's/.*<継続>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out" | head -1)
    kn=$(sed -n 's/.*<終了>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out" | head -1)
    ai=$(sed -n 's/.*<相槌>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out" | head -1)
    mf=$(sed -n 's/.*macro-F1 = \([0-9.]*\).*/\1/p' "$out" | head -1)
    bs=$(sed -n 's/.*二値分離 acc = \([0-9.]*\).*/\1/p' "$out" | head -1)
    d=$(ls -d exp/asr_*-turntaking-xfmr-adapter_r${key}-n0 2>/dev/null | tail -1)
    cer=$(awk -F'|' '/^\|decode/ && /\/eval\|/ {if ($4+0>w){w=$4+0;e=$9}} END{if(e!="")print e}' "$d/RESULTS.md" 2>/dev/null)
    printf "%-10s %8s %8s %8s %9s %10s %7s\n" "$s" "${ke:--}" "${kn:--}" "${ai:--}" "${mf:--}" "${bs:--}" "${cer:--}"
done
echo
echo "参考) 凍結     : 0.500 0.740 0.918   0.719   0.651   23.3"
echo "参考) LoRA     : 0.539 0.723 0.914   0.725   0.658   23.8"
echo "参考) Adapter無制限: 0.534 0.738 0.924   0.732   0.663   36.8"
