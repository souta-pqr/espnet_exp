#!/bin/bash
# (e) を加えたプロンプトで、Qwen3-32B-AWQ 思考なしの票を全データ付け直す（2026-10-10，ユーザ了承済み）。
#  1. Qwen3-32B-AWQ を ~/.cache/huggingface/hub にダウンロード（8B の実行中に済ませる）
#  2. 8B 思考あり (e) 版（run_full_e.sh，logs_e/state）が終わるのを待つ（GPU が 1 枚のため）
#  3. 確認：本番の仕組みで gold 922 件を出し、gold 単独で出した 32B (e) 版と比べる（思考なしは決定的なので 0.98 以上）
#  4. 本番（relabel/full_e/Qwen3-32B-AWQ-nothink.add_e.tsv）。設定は本番の 32B と同じ（最大長 5120・GPU メモリ 0.85）
#  5. 32B を消す（リンク先の共通 blobs の実体と、モデルのディレクトリ）
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
R=/autofs/diamond5/share/users/kobori/relabel
D=$R/full_e
H=~/.cache/huggingface/hub
st() { echo "$1" > logs_e32/state; }
mkdir -p logs_e32

echo "=== $(date '+%F %T') ダウンロード"
$PY -c "from huggingface_hub import snapshot_download; print('DONE', snapshot_download('Qwen/Qwen3-32B-AWQ'))" || { st download_failed; exit 1; }
export HF_HUB_OFFLINE=1

echo "=== $(date '+%F %T') 8B の終了を待つ"
until [ "$(cat logs_e/state 2>/dev/null)" = done ]; do
  pgrep -f "run_full_e[.]sh" >/dev/null || { echo "8B が done なしで止まった"; st waiting_8b_failed; exit 1; }
  sleep 300
done

ARGS="--model Qwen/Qwen3-32B-AWQ --name Qwen3-32B-AWQ-nothink.add_e --max-tokens 64 --max-model-len 5120 --gpu-mem 0.85 --chunk 256 --system $R/prompt_check/system_add_e.txt"
echo "=== $(date '+%F %T') 確認：gold 922 件"
$PY run_full.py $ARGS --only-gold --outdir $D/check_gold922 || { st error; exit 1; }
$PY - <<PYEOF || { st gate_failed; exit 10; }
def rd(p):
    return {int(l.split("\t")[0]): l.rstrip("\n").split("\t")[1] for l in open(p, encoding="utf-8") if "\t" in l}
a = rd("$D/check_gold922/Qwen3-32B-AWQ-nothink.add_e.tsv"); b = rd("$R/prompt_check/out/Qwen3-32B-AWQ-nothink.add_e.tsv")
k = sorted(set(a) & set(b)); agree = sum(a[x] == b[x] for x in k) / len(k)
print(f"gold 単独の 32B (e) 版との一致: {len(k)} 件 / 一致率 {agree:.3f}")
raise SystemExit(0 if len(k) == 922 and agree >= 0.98 else 1)
PYEOF

echo "=== $(date '+%F %T') 本番"
$PY run_full.py $ARGS --outdir $D || { st error; exit 1; }

echo "=== $(date '+%F %T') 32B を消す"
for l in $H/models--Qwen--Qwen3-32B-AWQ/blobs/*; do
  [ -L "$l" ] && rm -v "$(readlink -f "$l")"
done
rm -rf $H/models--Qwen--Qwen3-32B-AWQ $H/.locks/models--Qwen--Qwen3-32B-AWQ
du -sh $H/blobs
st done
echo "=== $(date '+%F %T') 終了"
