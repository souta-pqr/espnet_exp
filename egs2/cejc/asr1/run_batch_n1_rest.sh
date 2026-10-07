#!/usr/bin/env bash
# N=1 の欠けている手法を埋める。配線バグ修正前（〜07-29）の run は config.yaml に
# past_speech が 1 行も無く過去が学習に届いていないため、attn / frames は作り直しになる。
#   pureasr_tag=20260713-pureasr ./run_batch_n1_rest.sh
#
# N=1 で既に有効なもの: conv_c6（07-30）／ query_m8・topk_m8（run_batch_pool_sweep.sh）
#
# max は N=5 で最良だった手法なので、N=1 を取ると「N=5 が効いたのか max が効いたのか」を
# 分けられる。不要なら cmds から 1 行消せばよい。
# utt は N=1 だと有効発話数 1 × k=1 ＝ 1 スロットで attn とほぼ同粒度になる。
# 比較表（eval_xfmr_pool_compare.sh）の列を埋めるためのもので、単独の知見は薄い。
#
# 並列にしないこと。全手法が dump/raw/<set>/past_speech.scp を上書きコピーして使う。
# 1 本が失敗しても残りは流す。1 本でも失敗していれば終了コード 1 を返す。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export scope=${scope:-same}
export max_past=${max_past:-30}

cmds=(
    "N=1 ./run_pool_max.sh"
    "N=1 ./run_pool_attn.sh"
    "pool=frames N=1 ./run_xfmr_pool.sh"
    "N=1 ./run_pool_utt.sh"
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
echo "評価:"
echo "  pool=max            scope=${scope} N=1 ./eval_xfmr_pool.sh"
echo "  pool=attn           scope=${scope} N=1 ./eval_xfmr_pool.sh"
echo "  pool=frames         scope=${scope} N=1 ./eval_xfmr_pool.sh"
echo "  pool=utt  uslots=1  scope=${scope} N=1 ./eval_xfmr_pool.sh"
echo "  まとめて表に: budget=8 N=1 scope=${scope} ./eval_xfmr_pool_compare.sh"

printf '%s\n' "${status[@]}" | grep -q '^FAIL' && exit 1
exit 0
