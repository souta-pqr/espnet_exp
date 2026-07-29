#!/usr/bin/env bash
# Adapter の「適応の強さ」を掃引する（N=0 固定・過去発話なし）。
#   補正の大きさ <= s × 入力の大きさ  となる s を振り、CER とタグ性能のトレードオフを見る。
#
#   pureasr_tag=20260713-pureasr RATIOS="0.05 0.1 0.2 0.4" ./run_exp_adapter_ratio_sweep.sh
#
# 既に測定済み: s=無制限 → CER 36.8 / macro-F1 0.732
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
BASE=myconf/train_asr_turntaking_xfmr_adapter.yaml

for s in ${RATIOS:-0.05 0.1 0.2 0.4}; do
    key=$(echo "$s" | sed 's/0\.//; s/\.//')      # 0.05 -> 05, 0.1 -> 1, 0.2 -> 2
    cfg=myconf/.gen_adapter_ratio_${key}.yaml
    # max_ratio 行を差し替え（無ければ tt_adapter_ln の後ろに挿入）
    if grep -q "tt_adapter_max_ratio" "$BASE"; then
        sed "s/^\( *\)tt_adapter_max_ratio:.*/\1tt_adapter_max_ratio: ${s}/" "$BASE" > "$cfg"
    else
        sed "s/^\( *\)tt_adapter_ln:\(.*\)/\1tt_adapter_ln:\2\n\1tt_adapter_max_ratio: ${s}/" "$BASE" > "$cfg"
    fi
    echo "===================== Adapter 適応強さ s=${s}  (config=${cfg}) ====================="
    grep -E "tt_adapter_(ln|max_ratio|bottleneck)" "$cfg" | sed 's/^/    /'
    base_cfg="$cfg" tag_suffix="-adapter_r${key}" ./run_xfmr_n0.sh "$@"
done
echo "===== 掃引完了。評価: RATIOS=\"${RATIOS:-0.05 0.1 0.2 0.4}\" ./eval_adapter_ratio_sweep.sh ====="
