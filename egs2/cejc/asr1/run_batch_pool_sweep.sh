#!/usr/bin/env bash
# 圧縮手法（query / topk / conv / utt）の 6 本を 1 本ずつ順番に流す。
#   pureasr_tag=20260713-pureasr ./run_batch_pool_sweep.sh
#
# N=1 の conv は実行済みのため含めない（exp/asr_20260730-…-pool_conv_c6-same_n1_p30）。
# N=5 の conv は ratio 未指定なので run_pool_conv.sh が平均 Tp から c を決める（≈204/8 → c=26）。
# N=3 の utt は past_speech_same_n3_p30_bounds.scp が無いため、run_xfmr_pool.sh が
# past データを作り直してから学習に入る（utt2num_samples を読むだけなので数秒）。
#
# 並列にしないこと。4 本とも dump/raw/<set>/past_speech.scp を上書きコピーして使うため、
# 同時に走らせると別の N のデータで学習してしまう。
#
# 1 本が失敗しても残りは流す。最後に一覧を出し、1 本でも失敗していれば終了コード 1 を返す。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export scope=${scope:-same}
export max_past=${max_past:-30}

cmds=(
    "N=1 slots=8 ./run_pool_query.sh"
    "N=1 slots=8 ./run_pool_topk.sh"
    "N=5 ./run_pool_conv.sh"
    "N=5 slots=8 ./run_pool_query.sh"
    "N=5 slots=8 ./run_pool_topk.sh"
    "N=5 ./run_pool_utt.sh"
    "pool=frames N=5 ./run_xfmr_pool.sh"
)

# frames は run_pool_*.sh のラッパを経由しないので、配線チェックをここで 1 回だけ掛ける
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}

log_dir=tt_batch_logs; mkdir -p "$log_dir"
stamp=$(date +%Y%m%d-%H%M%S)
status=()

echo "===== batch 開始 ${stamp}  scope=${scope} max_past=${max_past} ====="
echo "      ${#cmds[@]} 本を逐次実行:"
printf '        %s\n' "${cmds[@]}"

for c in "${cmds[@]}"; do
    slug=$(echo "$c" | sed 's#\./##; s#\.sh##; s#[^A-Za-z0-9]\+#_#g')
    log="${log_dir}/${stamp}_${slug}.log"
    t0=$(date +%s)
    echo
    echo "----- [$(date +%H:%M:%S)] ${c} 開始 -> ${log}"
    eval "$c" > "${log}" 2>&1
    rc=$?
    el=$(( $(date +%s) - t0 ))
    if [ "${rc}" -eq 0 ]; then
        echo "----- [$(date +%H:%M:%S)] ${c} 完了 ($((el/3600))h$(((el%3600)/60))m)"
        status+=("OK   ${c}  $((el/3600))h$(((el%3600)/60))m")
    else
        echo "----- [$(date +%H:%M:%S)] ${c} 失敗 rc=${rc} ($((el/3600))h$(((el%3600)/60))m) -- 末尾20行:"
        tail -20 "${log}" | sed 's/^/       /'
        status+=("FAIL ${c}  rc=${rc}  ${log}")
    fi
done

echo
echo "===== batch 終了 ====="
printf '  %s\n' "${status[@]}"
echo
echo "評価は学習後に個別に（conv の ratio は上のログで確定した値を使う）:"
echo "  pool=query slots=8 scope=${scope} N=1 ./eval_xfmr_pool.sh"
echo "  pool=topk  slots=8 scope=${scope} N=1 ./eval_xfmr_pool.sh"
echo "  pool=conv  ratio=26 scope=${scope} N=5 ./eval_xfmr_pool.sh"
echo "  pool=query slots=8 scope=${scope} N=5 ./eval_xfmr_pool.sh"
echo "  pool=topk  slots=8 scope=${scope} N=5 ./eval_xfmr_pool.sh"
echo "  pool=utt   uslots=1 scope=${scope} N=5 ./eval_xfmr_pool.sh"
echo "  pool=frames        scope=${scope} N=5 ./eval_xfmr_pool.sh"
echo "  まとめて表に: budget=8 N=5 scope=${scope} ./eval_xfmr_pool_compare.sh"

printf '%s\n' "${status[@]}" | grep -q '^FAIL' && exit 1
exit 0
