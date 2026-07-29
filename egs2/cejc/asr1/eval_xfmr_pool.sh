#!/usr/bin/env bash
# 過去音声プーリング版（max/attn）のタグ性能を評価。
#   pool=attn scope=same N=1 max_past=30 ./eval_xfmr_pool.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
pool=${pool:?pool=max または attn}; N=${N:-1}; max_past=${max_past:-30}; scope=${scope:-same}
ratio=${ratio:-8}; slots=${slots:-8}; uslots=${uslots:-1}   # conv圧縮率 / query,topkスロット / uttスロット
ckpt=${ckpt:-valid.tag_acc.ave.pth}
case "${pool}" in
    conv)        pooltag="conv_c${ratio}"    ;;
    query|topk)  pooltag="${pool}_m${slots}" ;;
    utt)         pooltag="utt_k${uslots}"    ;;
    *)           pooltag="${pool}"           ;;
esac
variant="${scope}_n${N}_p${max_past}"
d=$(ls -d exp/asr_*-turntaking-xfmr-pool_${pooltag}-${scope}_n${N}_p${max_past} 2>/dev/null | tail -1 || true)
[ -n "$d" ] || { echo "学習済みモデルが見つかりません"; exit 1; }
ps="dump/raw/eval/past_speech_${variant}.scp"
[ -f "$ps" ] || { echo "$ps がありません"; exit 1; }
out="tt_eval_logs/pool_${pooltag}_${scope}_n${N}_${ckpt%.pth}.log"; mkdir -p tt_eval_logs
echo "===== eval pool=${pooltag} ${scope} N=${N} (${d}) -> ${out} ====="
$PY local/turntaking/evaluate_multitask.py \
    --config "$d/config.yaml" --model "$d/$ckpt" \
    --data_dir dump/raw/eval --past_pool --past_speech_scp "$ps" \
    > "$out" 2>&1
grep -E "macro-F1|<継続>|<終了>|<相槌>|二値分離" "$out" | sed 's/^/    /'
