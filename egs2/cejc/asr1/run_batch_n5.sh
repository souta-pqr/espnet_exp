#!/usr/bin/env bash
# N=5 / scope=same の 4 手法（max / attn / LoRA / Adapter）を 1 本ずつ順番に流す。
#   pureasr_tag=20260713-pureasr ./run_batch_n5.sh
#   pureasr_tag=20260713-pureasr N=1 scope=session ./run_batch_n5.sh
#
# 並列にしないこと。attn / LoRA / Adapter は asr_stats_dir
# （exp/asr_stats_raw_jp_word_pool_attn_${scope}_n${N}_p${max_past}）を共有し、
# さらに 4 本とも dump/raw/<set>/past_speech.scp を上書きコピーして使うため、
# 同時に走らせると stats と scp が競合する。
#
# 1 本が失敗しても残りは流す（夜間に全部止まるのを避ける）。最後に一覧を出し、
# 1 本でも失敗していれば終了コード 1 を返す。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export N=${N:-5}
export scope=${scope:-same}
export max_past=${max_past:-30}

log_dir=tt_batch_logs; mkdir -p "$log_dir"
stamp=$(date +%Y%m%d-%H%M%S)
scripts=(run_pool_max.sh run_pool_attn.sh run_peft_lora.sh run_peft_adapter.sh)
status=()

echo "===== batch 開始 ${stamp}  N=${N} scope=${scope} max_past=${max_past} ====="
echo "      ${#scripts[@]} 本を逐次実行: ${scripts[*]}"

for s in "${scripts[@]}"; do
    log="${log_dir}/${stamp}_${s%.sh}_${scope}_n${N}.log"
    t0=$(date +%s)
    echo
    echo "----- [$(date +%H:%M:%S)] ${s} 開始 -> ${log}"
    ./"${s}" > "${log}" 2>&1
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
echo "評価は学習後に個別に:"
echo "  pool=max  scope=${scope} N=${N} ./eval_xfmr_pool.sh"
echo "  pool=attn scope=${scope} N=${N} ./eval_xfmr_pool.sh"
echo "  pool=attn scope=${scope} N=${N} tag_suffix=-lora    ./eval_xfmr_pool.sh"
echo "  pool=attn scope=${scope} N=${N} tag_suffix=-adapter ./eval_xfmr_pool.sh"

printf '%s\n' "${status[@]}" | grep -q '^FAIL' && exit 1
exit 0
