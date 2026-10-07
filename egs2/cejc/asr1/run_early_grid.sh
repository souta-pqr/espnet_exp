#!/usr/bin/env bash
# 早期確定：細かい時刻グリッドで確率ダンプを取る。
#   ./run_early_grid.sh                       # 打ち切り学習モデル（既定）
#   tag_suffix= ./run_early_grid.sh           # 通常学習モデル
#
# eval  … 停止ルールの評価対象
# train_dev … TEASER の master 学習と閾値較正に使う（eval を汚さないため）
#
# 既に TSV があるグリッド点は飛ばすので、途中で止めて再開しても無駄がない。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

tag_suffix=${tag_suffix--trunc}        # -trunc（打ち切り学習）/ 空文字（通常学習）
pool=${pool:-max}; scope=${scope:-same}; N=${N:-5}
ckpt=${ckpt:-valid.loss.ave.pth}
grid=${grid:-"0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5 0.6"}
sets=${sets:-"eval train_dev"}

stem="pool_${pool}${tag_suffix}_${scope}_n${N}"
echo "===== 早期確定グリッド $(date +%H:%M:%S) ====="
echo "  モデル: ${stem}   点数: $(echo $grid|wc -w) × $(echo $sets|wc -w) セット"
n_run=0; n_skip=0; fail=0
for ds in ${sets}; do
  for t in ${grid}; do
    dsfx=""; [ "$ds" = "eval" ] || dsfx="_${ds}"
    tsfx=""; [ "$t" = "0" ] || tsfx="_t${t}"
    f="tt_preds/${stem}${dsfx}${tsfx}_${ckpt%.pth}.tsv"
    if [ -s "$f" ]; then n_skip=$((n_skip+1)); continue; fi
    t0=$(date +%s)
    printf -- "----- [%s] %-10s t=%-5s " "$(date +%H:%M:%S)" "$ds" "$t"
    if trunc=$t pool="$pool" tag_suffix="$tag_suffix" scope="$scope" N="$N" \
       dset="$ds" ckpt="$ckpt" dump=1 ./eval_xfmr_pool.sh >/dev/null 2>&1; then
      echo "OK ($((($(date +%s)-t0)/60))分)"; n_run=$((n_run+1))
    else
      echo "FAIL"; fail=$((fail+1))
    fi
  done
done
echo "===== 完了 $(date +%H:%M:%S)  実行 ${n_run} / 既存 ${n_skip} / 失敗 ${fail} ====="
echo
echo "解析:"
echo "  python local/turntaking/early_commit.py --stem ${stem} \\"
echo "      --times 0.6,0.5,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0"
echo "  python local/turntaking/teaser.py --stem ${stem} \\"
echo "      --times 0.6,0.5,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0"
exit $((fail > 0))
