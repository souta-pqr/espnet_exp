#!/usr/bin/env bash
# LoRA / Adapter を「過去枝にも勾配を通す」設定で回し直す。
#   pureasr_tag=20260713-pureasr ./run_batch_peft_pastgrad.sh
#
# 何が変わったか:
#   espnet2/asr/espnet_model_turntaking_xfmr.py の _encode_past() を新設し、
#   過去音声の符号化を tt_adapt に従わせた。tt_adapt=="none"（凍結手法）は従来どおり
#   勾配なしで、PEFT だけ現発話と同じく勾配ありで通る。凍結手法の結果は不変。
#
#   逆伝播するフレーム数が 現発話 1.47 秒 → 現発話+過去 8.21 秒（約 5.6 倍）になるため、
#   PEFT の config は batch_bins 700000→175000 / accum_grad 8→32 にしてある。
#   実効バッチサイズと 1 エポックあたりの最適化ステップ数は凍結手法の run と一致するので、
#   学習条件はそろったまま、所要時間だけが増える（実測 48 分/epoch から 2〜3 倍の見込み）。
#
# asr_tag に "z-pastgrad" を挟むのは 2 つの理由:
#   (1) 旧設定の exp/asr_20260801-…-pool_attn-lora-same_n5_p30 と同じ名前になると
#       asr.sh が checkpoint.pth から resume してしまい、旧学習の続きになる
#   (2) eval_xfmr_pool.sh は同名 glob を ls | tail -1 で選ぶので、旧 run より後に
#       ソートされる必要がある（'z' > '-'）
#
# 並列にしないこと。両方とも dump/raw/<set>/past_speech.scp を上書きコピーして使う。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export N=${N:-5}
export scope=${scope:-same}
export max_past=${max_past:-30}

tag_pre="$(date +%Y%m%d)z-pastgrad"
variant="${scope}_n${N}_p${max_past}"

awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}
grep -q "_encode_past" ../../../espnet2/asr/espnet_model_turntaking_xfmr.py || {
    echo "[中止] espnet_model_turntaking_xfmr.py に _encode_past がありません（修正が入っていない）。" >&2
    exit 1
}

# "手法名:tag_suffix" の並び。tag_suffix は run_peft_*.sh 側の値に合わせる
runs=("run_peft_lora.sh:-lora" "run_peft_adapter.sh:-adapter")

# 出力先が既にあると asr.sh が resume してしまうので、先に全部チェックしてから走り出す
for r in "${runs[@]}"; do
    d="exp/asr_${tag_pre}-turntaking-xfmr-pool_attn${r#*:}-${variant}"
    [ -d "$d" ] && { echo "[中止] ${d} が既にあります。消すか asr_tag を変えてください。" >&2; exit 1; }
done

log_dir=tt_batch_logs; mkdir -p "$log_dir"
stamp=$(date +%Y%m%d-%H%M%S)
status=()

echo "===== batch 開始 ${stamp}  N=${N} scope=${scope}（過去枝に勾配あり）====="
printf '        %s\n' "${runs[@]%%:*}"

for r in "${runs[@]}"; do
    s="${r%%:*}"; suf="${r#*:}"
    tag="${tag_pre}-turntaking-xfmr-pool_attn${suf}-${variant}"
    log="${log_dir}/${stamp}_${s%.sh}_pastgrad_${scope}_n${N}.log"
    t0=$(date +%s)
    echo
    echo "----- [$(date +%H:%M:%S)] ${s} 開始  tag=${tag}"
    echo "      -> ${log}"
    asr_tag="${tag}" ./"${s}" > "${log}" 2>&1
    rc=$?
    el=$(( $(date +%s) - t0 ))
    if [ "${rc}" -eq 0 ]; then
        echo "----- [$(date +%H:%M:%S)] ${s} 完了 ($((el/3600))h$(((el%3600)/60))m)"
        status+=("OK   ${s}  $((el/3600))h$(((el%3600)/60))m")
    else
        echo "----- [$(date +%H:%M:%S)] ${s} 失敗 rc=${rc} ($((el/3600))h$(((el%3600)/60))m) -- 末尾20行:"
        tail -20 "${log}" | sed 's/^/       /'
        status+=("FAIL ${s}  rc=${rc}  ${log}")
    fi
done

echo
echo "===== batch 終了 ====="
printf '  %s\n' "${status[@]}"
echo
echo "OOM で落ちていたら batch_bins を半分・accum_grad を倍にして再実行:"
echo "  myconf/train_asr_turntaking_xfmr_peft_{lora,adapter}.yaml"
echo "  batch_bins: 175000 -> 87500 / accum_grad: 32 -> 64"
echo
echo "評価:"
echo "  pool=attn scope=${scope} N=${N} tag_suffix=-lora    ./eval_xfmr_pool.sh"
echo "  pool=attn scope=${scope} N=${N} tag_suffix=-adapter ./eval_xfmr_pool.sh"

printf '%s\n' "${status[@]}" | grep -q '^FAIL' && exit 1
exit 0
