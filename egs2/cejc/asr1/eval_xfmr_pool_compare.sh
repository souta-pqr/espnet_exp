#!/usr/bin/env bash
# 過去要約の比較結果を macro-F1 の高い順に並べ、どれが有効かを 1 枚の表にする。
#   budget=8 N=1 scope=same ./eval_xfmr_pool_compare.sh
#   POOLS="conv query topk" budget=8 N=5 scope=same ./eval_xfmr_pool_compare.sh
# 未評価のものは評価してから集計する（run_exp_xfmr_pool_compare.sh と同じ予算の解釈を使う）。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
scope=${scope:-same}; N=${N:-1}; max_past=${max_past:-30}
budget=${budget:-8}
ckpt=${ckpt:-valid.tag_acc.ave.pth}
POOLS=${POOLS:-"attn frames conv query topk utt"}
train_set=${train_set:-train_nodup}
variant="${scope}_n${N}_p${max_past}"
mkdir -p tt_eval_logs

# conv の圧縮率は run 側と同じ式で復元する（フレーム = 528 サンプル）
bounds="dump/raw/${train_set}/past_speech_${variant}_bounds.scp"
[ -f "${bounds}" ] || { echo "${bounds} がありません。先に run_exp_xfmr_pool_compare.sh を実行してください。" >&2; exit 1; }
read -r mean_tp conv_c < <(
    awk -v m="${budget}" '{s=0; for(i=2;i<=NF;i++) s+=$i; t+=s; n++}
        END{tp=t/n/528; c=int(tp/m+0.5); if(c<1)c=1; printf "%.0f %d\n", tp, c}' "${bounds}"
)

rows=$(mktemp); trap 'rm -f "${rows}"' EXIT

for pool in ${POOLS}; do
    case "${pool}" in
        conv)       pooltag="conv_c${conv_c}";    slotinfo="≈$((mean_tp / conv_c))"; kv="ratio=${conv_c}" ;;
        query|topk) pooltag="${pool}_m${budget}"; slotinfo="${budget}";              kv="slots=${budget}" ;;
        utt)        pooltag="utt_k1";             slotinfo="≤${N}";                  kv="uslots=1" ;;
        max|attn)   pooltag="${pool}";            slotinfo="1";                      kv="slots=${budget}" ;;
        frames)     pooltag="frames";             slotinfo="≈${mean_tp}";            kv="slots=${budget}" ;;
        *) echo "[skip] 未知の pool: ${pool}" >&2; continue ;;
    esac

    out="tt_eval_logs/pool_${pooltag}_${scope}_n${N}_${ckpt%.pth}.log"
    d=$(ls -d exp/asr_*-turntaking-xfmr-pool_"${pooltag}"-"${variant}" 2>/dev/null | tail -1 || true)
    if [ ! -f "$out" ] || ! grep -q "macro-F1" "$out"; then
        if [ -z "${d}" ] || [ ! -f "${d}/${ckpt}" ]; then echo "[skip] ${pool}: 未学習"; continue; fi
        echo "===== eval ${pool} (${pooltag}) ====="
        env "${kv%%=*}=${kv#*=}" pool="${pool}" scope="${scope}" N="${N}" max_past="${max_past}" \
            ckpt="${ckpt}" ./eval_xfmr_pool.sh >/dev/null
    fi
    [ -f "$out" ] || continue

    ke=$(sed -n 's/.*<継続>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    kn=$(sed -n 's/.*<終了>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    ai=$(sed -n 's/.*<相槌>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    mf=$(sed -n 's/.*macro-F1 = \([0-9.]*\).*/\1/p' "$out"|head -1)
    bs=$(sed -n 's/.*二値分離 acc = \([0-9.]*\).*/\1/p' "$out"|head -1)
    cer=$(awk -F'|' '/^\|decode/ && /\/eval\|/ {if ($4+0>w){w=$4+0;e=$9}} END{if(e!="")print e}' "${d}/RESULTS.md" 2>/dev/null || true)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${mf:-0}" "${pool}" "${pooltag}" "${slotinfo}" "${ke:--}" "${kn:--}" "${ai:--}" "${bs:--}|${cer:--}" >> "${rows}"
done

echo
echo "============ 過去要約の比較  scope=${scope} N=${N} 予算 M=${budget} / ckpt=${ckpt} ============"
echo "平均 Tp ≈ ${mean_tp} フレーム（conv は c=${conv_c}）"
echo
echo "| 手法 | 分類 | スロット数 | 継続F1 | 完了F1 | 相槌F1 | macro-F1 | 二値分離 | CER |"
echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
best=""
sort -gr "${rows}" | while IFS=$'\t' read -r mf pool pooltag slots ke kn ai rest; do
    bs="${rest%%|*}"; cer="${rest##*|}"
    case "${pool}" in
        conv)   kind="固定率・soft" ;;
        query)  kind="固定長・soft" ;;
        topk)   kind="固定長・hard" ;;
        utt)    kind="会話単位"     ;;
        attn)   kind="既存 M=1"     ;;
        max)    kind="既存 M=1・非学習" ;;
        frames) kind="圧縮なし"     ;;
        *)      kind="-"            ;;
    esac
    mark=""
    if [ -z "${best}" ]; then mark=" ★"; best="${pool}"; fi
    echo "| ${pool}${mark} | ${kind} | ${slots} | ${ke} | ${kn} | ${ai} | **${mf}** | ${bs} | ${cer} |"
done
echo
echo "★ = この条件で macro-F1 が最も高い手法。CER は凍結条件により全手法で同一のはず（ずれていたら凍結が壊れている）。"
