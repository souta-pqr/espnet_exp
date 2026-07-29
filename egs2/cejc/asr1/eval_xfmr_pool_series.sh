#!/usr/bin/env bash
# 過去発話プーリング（max/attn）を N ごとに評価し表にまとめる。
#   pool=attn scope=same NS="1 2 3 4 5" ./eval_xfmr_pool_series.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
pool=${pool:?pool=max または attn}; scope=${scope:-same}; max_past=${max_past:-30}
ratio=${ratio:-8}; slots=${slots:-8}; uslots=${uslots:-1}   # conv圧縮率 / query,topkスロット / uttスロット
ckpt=${ckpt:-valid.tag_acc.ave.pth}; NS=${NS:-1 2 3 4 5}
case "${pool}" in
    conv)        pooltag="conv_c${ratio}"    ;;
    query|topk)  pooltag="${pool}_m${slots}" ;;
    utt)         pooltag="utt_k${uslots}"    ;;
    *)           pooltag="${pool}"           ;;
esac
mkdir -p tt_eval_logs
for n in $NS; do
    variant="${scope}_n${n}_p${max_past}"
    d=$(ls -d exp/asr_*-turntaking-xfmr-pool_${pooltag}-${scope}_n${n}_p${max_past} 2>/dev/null | tail -1 || true)
    ps="dump/raw/eval/past_speech_${variant}.scp"
    if [ -z "$d" ] || [ ! -f "$d/$ckpt" ]; then echo "[skip] N=${n}: 未学習"; continue; fi
    if [ ! -f "$ps" ]; then echo "[skip] N=${n}: $ps なし"; continue; fi
    out="tt_eval_logs/pool_${pooltag}_${scope}_n${n}_${ckpt%.pth}.log"
    if [ ! -f "$out" ] || ! grep -q "macro-F1" "$out"; then
        echo "===== eval pool=${pool} ${scope} N=${n} ====="
        $PY local/turntaking/evaluate_multitask.py --config "$d/config.yaml" --model "$d/$ckpt" \
            --data_dir dump/raw/eval --past_pool --past_speech_scp "$ps" > "$out" 2>&1
    fi
done
echo
echo "================ pool=${pooltag} / scope=${scope} ================"
echo "| N | 継続F1 | 完了F1 | 相槌F1 | macro-F1 | **二値分離** | CER |"
echo "| --- | --- | --- | --- | --- | --- | --- |"
for n in $NS; do
    out="tt_eval_logs/pool_${pooltag}_${scope}_n${n}_${ckpt%.pth}.log"
    [ -f "$out" ] || continue
    ke=$(sed -n 's/.*<継続>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    kn=$(sed -n 's/.*<終了>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    ai=$(sed -n 's/.*<相槌>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    mf=$(sed -n 's/.*macro-F1 = \([0-9.]*\).*/\1/p' "$out"|head -1)
    bs=$(sed -n 's/.*二値分離 acc = \([0-9.]*\).*/\1/p' "$out"|head -1)
    d=$(ls -d exp/asr_*-turntaking-xfmr-pool_${pooltag}-${scope}_n${n}_p${max_past} 2>/dev/null | tail -1)
    cer=$(awk -F'|' '/^\|decode/ && /\/eval\|/ {if ($4+0>w){w=$4+0;e=$9}} END{if(e!="")print e}' "$d/RESULTS.md" 2>/dev/null)
    echo "| $n | ${ke:--} | ${kn:--} | ${ai:--} | ${mf:--} | **${bs:--}** | ${cer:--} |"
done
