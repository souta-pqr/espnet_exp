#!/usr/bin/env bash
# 圧縮つまみ掃引の結果を表にまとめる（未評価のものは評価してから集計）。
#   pool=conv  VALS="1 8 16 32"  N=1 scope=same ./eval_xfmr_pool_compress_sweep.sh
#   pool=query VALS="1 2 4 8 16" N=1 scope=same ./eval_xfmr_pool_compress_sweep.sh
#   pool=topk  VALS="4 8 16"     N=1 scope=same ./eval_xfmr_pool_compress_sweep.sh
# 表の末尾に既存手法（frames / max / attn）の行を、評価ログがあれば参考として並べる。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
pool=${pool:?pool=conv / query / topk のいずれかを指定}
scope=${scope:-same}; N=${N:-1}; max_past=${max_past:-30}
ckpt=${ckpt:-valid.tag_acc.ave.pth}

case "${pool}" in
    conv)  knob="c"; defaults="1 8 16 32"  ;;
    query) knob="M"; defaults="1 2 4 8 16" ;;
    topk)  knob="M"; defaults="4 8 16"     ;;
    *) echo "pool=${pool} は掃引対象外です（conv / query / topk のみ）。" >&2; exit 1 ;;
esac
VALS=${VALS:-$defaults}
mkdir -p tt_eval_logs

# 未評価のものを評価（eval_xfmr_pool.sh がログを書く）
for v in ${VALS}; do
    case "${pool}" in
        conv)       pooltag="conv_c${v}"; kv="ratio=${v}" ;;
        query|topk) pooltag="${pool}_m${v}"; kv="slots=${v}" ;;
    esac
    out="tt_eval_logs/pool_${pooltag}_${scope}_n${N}_${ckpt%.pth}.log"
    if [ -f "$out" ] && grep -q "macro-F1" "$out"; then continue; fi
    d=$(ls -d exp/asr_*-turntaking-xfmr-pool_"${pooltag}"-"${scope}"_n"${N}"_p"${max_past}" 2>/dev/null | tail -1 || true)
    if [ -z "$d" ] || [ ! -f "$d/$ckpt" ]; then echo "[skip] ${knob}=${v}: 未学習"; continue; fi
    echo "===== eval pool=${pool} ${knob}=${v} ====="
    env "${kv%%=*}=${kv#*=}" pool="${pool}" scope="${scope}" N="${N}" max_past="${max_past}" \
        ckpt="${ckpt}" ./eval_xfmr_pool.sh >/dev/null
done

# 1 行分の集計（$1=表示名 $2=pooltag）
row() {
    local label="$1" pooltag="$2"
    local out="tt_eval_logs/pool_${pooltag}_${scope}_n${N}_${ckpt%.pth}.log"
    [ -f "$out" ] || return 0
    local ke kn ai mf bs d cer
    ke=$(sed -n 's/.*<継続>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    kn=$(sed -n 's/.*<終了>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    ai=$(sed -n 's/.*<相槌>[^0-9]*[0-9.]*  *[0-9.]*  *\([0-9.]*\).*/\1/p' "$out"|head -1)
    mf=$(sed -n 's/.*macro-F1 = \([0-9.]*\).*/\1/p' "$out"|head -1)
    bs=$(sed -n 's/.*二値分離 acc = \([0-9.]*\).*/\1/p' "$out"|head -1)
    d=$(ls -d exp/asr_*-turntaking-xfmr-pool_"${pooltag}"-"${scope}"_n"${N}"_p"${max_past}" 2>/dev/null | tail -1 || true)
    cer=$(awk -F'|' '/^\|decode/ && /\/eval\|/ {if ($4+0>w){w=$4+0;e=$9}} END{if(e!="")print e}' "$d/RESULTS.md" 2>/dev/null || true)
    echo "| ${label} | ${ke:--} | ${kn:--} | ${ai:--} | ${mf:--} | **${bs:--}** | ${cer:--} |"
}

echo
echo "================ pool=${pool} / ${scope} N=${N} / ckpt=${ckpt} ================"
echo "| ${knob} | 継続F1 | 完了F1 | 相槌F1 | macro-F1 | **二値分離** | CER |"
echo "| --- | --- | --- | --- | --- | --- | --- |"
for v in ${VALS}; do
    case "${pool}" in
        conv)       row "${v}" "conv_c${v}" ;;
        query|topk) row "${v}" "${pool}_m${v}" ;;
    esac
done
echo
echo "参考) 既存手法（同じ scope / N の評価ログがある場合のみ）"
echo "| 手法 | 継続F1 | 完了F1 | 相槌F1 | macro-F1 | **二値分離** | CER |"
echo "| --- | --- | --- | --- | --- | --- | --- |"
row "frames (c=1 相当)" "frames"
row "max (M=1)"         "max"
row "attn (M=1)"        "attn"
