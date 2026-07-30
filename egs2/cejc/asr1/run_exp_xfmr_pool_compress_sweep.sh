#!/usr/bin/env bash
# 過去発話圧縮の「つまみ」を掃引する（過去発話数 N は固定）。
# 出典がはっきりしている 3 手法だけを対象にする:
#
#   conv  固定率・soft  つまみ=圧縮率 c    Compressive Transformer      [Rae+ ICLR2020  arXiv:1911.05507]
#   query 固定長・soft  つまみ=スロット数 M Set Transformer PMA / Perceiver
#                                          [Lee+ ICML2019 arXiv:1810.00825 / Jaegle+ ICML2021 arXiv:2103.03206]
#   topk  固定長・hard  つまみ=スロット数 M Compressive Transformer "most-used" [Rae+ ICLR2020  arXiv:1911.05507]
#
#   pool=conv  VALS="1 8 16 32"  N=1 scope=same pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_compress_sweep.sh
#   pool=query VALS="1 2 4 8 16" N=1 scope=same pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_compress_sweep.sh
#   pool=topk  VALS="4 8 16"     N=1 scope=same pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_compress_sweep.sh
#
# VALS の意味は pool で変わる:
#   conv       → 圧縮率 c（過去 Tp フレーム → ceil(Tp/c) スロット。出力長は過去長に比例）
#   query/topk → スロット数 M（過去長に依らず常に M 本）
#
# 端点は既存手法と一致するので、掃引の両端がそのまま既存手法との比較になる:
#   conv c=1 ≡ frames（圧縮なし） / conv c>=Tp ≡ max / query M=1 ≡ attn（単一ヘッド）
#
# 発話単位 (pool=utt) は直接の出典がないため本掃引の対象外。
# max / attn / frames / utt は run_xfmr_pool.sh で単発実行する。
# N を振る掃引は run_exp_xfmr_pool_series.sh（つまみを固定して N=1..5）。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export pool=${pool:?pool=conv / query / topk のいずれかを指定}
export scope=${scope:-same}
export N=${N:-1}
export max_past=${max_past:-30}

case "${pool}" in
    conv)
        knob="圧縮率 c"; defaults="1 8 16 32"
        ref='Rae+ ICLR2020 (arXiv:1911.05507)  固定率・soft'
        ;;
    query)
        knob="スロット数 M"; defaults="1 2 4 8 16"
        ref='Lee+ ICML2019 (arXiv:1810.00825) / Jaegle+ ICML2021 (arXiv:2103.03206)  固定長・soft'
        ;;
    topk)
        knob="スロット数 M"; defaults="4 8 16"
        ref='Rae+ ICLR2020 "most-used" (arXiv:1911.05507)  固定長・hard'
        ;;
    *)
        echo "pool=${pool} は掃引対象外です（conv / query / topk のみ）。" >&2
        echo "max / attn / frames / utt は run_xfmr_pool.sh で単発実行してください。" >&2
        exit 1
        ;;
esac

VALS=${VALS:-$defaults}
echo "================================================================"
echo " pool=${pool}  掃引: ${knob} = ${VALS}"
echo " 出典: ${ref}"
echo " 固定: scope=${scope} N=${N} max_past=${max_past}s  Stage1=${pureasr_tag}"
echo "================================================================"

for v in ${VALS}; do
    case "${pool}" in
        conv)       export ratio="${v}" ;;
        query|topk) export slots="${v}" ;;
    esac
    echo "===================== pool=${pool} ${knob}=${v} (${scope} N=${N}) ====================="
    ./run_xfmr_pool.sh "$@"
done

echo "===== 掃引完了（pool=${pool} / ${knob} = ${VALS}）====="
echo "評価: pool=${pool} VALS=\"${VALS}\" N=${N} scope=${scope} ./eval_xfmr_pool_compress_sweep.sh"
