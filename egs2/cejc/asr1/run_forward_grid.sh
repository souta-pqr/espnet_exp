#!/usr/bin/env bash
# 前向きグリッド：発話「開始から」E 秒だけ聞いた状態でのダンプを取る。
#
# run_early_grid.sh が「発話末の t 秒手前」で切るのに対し、こちらは経過時間で切る。
# 実行時に観測できるのは経過時間のほうなので、Economy（10 節）の将来コスト推定や
# FIRMBOUND の後ろ向き帰納を、実運用の全区間で当てはめられるようになる。
# 現状のダンプは発話末から 0.6 秒の窓しか覆っておらず、そこが唯一の弱点だった。
#
#   tag_suffix=-trunc03 pool=attn N=5 ./run_forward_grid.sh
#   sets=eval tag_suffix= pool=attn N=5 ./run_forward_grid.sh     # 打切なしモデル
#
# 既に TSV があるグリッド点は飛ばすので、途中で止めて再開しても無駄がない。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

tag_suffix=${tag_suffix--trunc}
pool=${pool:-attn}; scope=${scope:-same}; N=${N:-5}
ckpt=${ckpt:-valid.loss.ave.pth}
# 現発話の平均は 1.47 秒。0.25 秒（評価側の下限）から 3 秒までを覆う。
grid=${grid:-"0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0"}
# withsil=1 で発話後の無音を含めて切り出す（元録音から。ファイル名は _e1.0s 形式）
withsil=${withsil:-0}
sets=${sets:-"eval train_dev"}

stem="pool_${pool}${tag_suffix}_${scope}_n${N}"
echo "===== 前向きグリッド $(date +%H:%M:%S) ====="
echo "  モデル: ${stem}   点数: $(echo $grid|wc -w) × $(echo $sets|wc -w) セット"
n_run=0; n_skip=0; fail=0
for ds in ${sets}; do
  for e in ${grid}; do
    dsfx=""; [ "$ds" = "eval" ] || dsfx="_${ds}"
    ssfx=""; [ "${withsil}" = "1" ] && ssfx="s"
    f="tt_preds/${stem}${dsfx}_e${e}${ssfx}_${ckpt%.pth}.tsv"
    if [ -s "$f" ]; then n_skip=$((n_skip+1)); continue; fi
    t0=$(date +%s)
    printf -- "----- [%s] %-10s E=%-5s " "$(date +%H:%M:%S)" "$ds" "$e"
    if keep=$e withsil="$withsil" pool="$pool" tag_suffix="$tag_suffix" scope="$scope" N="$N" \
       dset="$ds" ckpt="$ckpt" dump=1 ./eval_xfmr_pool.sh >/dev/null 2>&1; then
      echo "OK ($((($(date +%s)-t0)/60))分)"; n_run=$((n_run+1))
    else
      echo "FAIL"; fail=$((fail+1))
    fi
  done
done
echo "===== 完了 $(date +%H:%M:%S)  実行 ${n_run} / 既存 ${n_skip} / 失敗 ${fail} ====="
exit $((fail > 0))
