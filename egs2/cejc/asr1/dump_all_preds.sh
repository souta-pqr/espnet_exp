#!/usr/bin/env bash
# 全手法の区間ごとの確率を tt_preds/*.tsv に貯める。
#   ./dump_all_preds.sh              # eval と train_dev の両方（既定）
#   sets=eval ./dump_all_preds.sh    # eval だけ
#
# eval    … AUC・3クラス F1 の測定対象
# train_dev … 閾値較正（analyze_preds.py --dev）に使う。eval とクラス比がずれている
#             （継続 16.9% vs 26.7%）ので、較正が転移するかは結果を見て判断する。
#
# 一度貯めれば、以後は指標を変えても GPU を使わず analyze_preds.py で計算できる。
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
ckpt=${ckpt:-valid.loss.ave.pth}
sets=${sets:-"eval train_dev"}

# "説明:eval_xfmr_pool.sh に渡す環境変数"
models=(
    "conv c=6   N=1:pool=conv  ratio=6  N=1"
    "query m8   N=1:pool=query slots=8  N=1"
    "topk  m8   N=1:pool=topk  slots=8  N=1"
    "max        N=5:pool=max            N=5"
    "attn       N=5:pool=attn           N=5"
    "conv c=26  N=5:pool=conv  ratio=26 N=5"
    "query m8   N=5:pool=query slots=8  N=5"
    "topk  m8   N=5:pool=topk  slots=8  N=5"
    "utt  k1    N=5:pool=utt   uslots=1 N=5"
    "frames     N=5:pool=frames         N=5"
    "attn+LoRA  N=5:pool=attn  tag_suffix=-lora N=5"
)

mkdir -p tt_preds
echo "===== 確率ダンプ開始 $(date +%H:%M:%S)  ckpt=${ckpt}  sets=${sets} ====="
fail=0
for m in "${models[@]}"; do
    name="${m%%:*}"; kv="${m#*:}"
    for ds in ${sets}; do
        t0=$(date +%s)
        printf -- "----- [%s] %-16s %-10s " "$(date +%H:%M:%S)" "${name}" "${ds}"
        if env ${kv} scope=same dset="${ds}" ckpt="${ckpt}" dump=1 \
               ./eval_xfmr_pool.sh > /dev/null 2>&1; then
            echo "OK ($((($(date +%s)-t0)/60))分)"
        else
            echo "FAIL"; fail=$((fail+1))
        fi
    done
done
echo "===== 完了 $(date +%H:%M:%S)  失敗 ${fail} 件 ====="
ls -la tt_preds/ | tail -n +2
echo
echo "解析:  python local/turntaking/analyze_preds.py tt_preds/*_${ckpt%.pth}.tsv"
exit $((fail > 0))
