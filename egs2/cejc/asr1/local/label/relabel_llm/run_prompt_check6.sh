#!/bin/bash
# 別系統の LLM を (e) 版プロンプトで試す（2026-10-12）。人手正解 922 件だけ。
# モデルは /data/huggingface/hub の共有キャッシュを読むだけ（ダウンロードしない）。
# run_gold_prompts.py に --family（gemma / harmony 形式の思考の取り出し）・--trust-remote-code・--llm-kwargs を足した（元は .bak_v）。
# 出力：/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_v/
set -uo pipefail
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/relabel_llm
PY=~/anaconda3/envs/llm/bin/python
P=/autofs/diamond5/share/users/kobori/relabel/prompt_check/gold1000_prompts.add_e.jsonl
O=/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_v
H=/data/huggingface/hub
snap() { ls -d $H/models--$1/snapshots/*/ | head -1; }
NO="--max-tokens 64 --max-model-len 5120 --chunk 256"
TH="--max-tokens 4096 --max-model-len 10240 --chunk 512"
MM='{"limit_mm_per_prompt": {"image": 0, "video": 0}, "max_num_seqs": 64}'
run() {  # 名前 モデル 追加引数...
  local n=$1 m=$2; shift 2
  [ -s $O/$n.tsv ] && [ "$(wc -l < $O/$n.tsv)" -ge 922 ] && { echo "済み: $n"; return; }
  echo "=== $(date "+%F %T") $n"
  $PY run_gold_prompts.py --prompts $P --model "$(snap $m)" --name $n --outdir $O "$@" || echo "失敗: $n"
}
run gemma-4-12B-it.add_e            google--gemma-4-12B-it-qat-w4a16-ct $NO --gpu-mem 0.85
run gemma-4-12B-it-think.add_e      google--gemma-4-12B-it-qat-w4a16-ct $TH --gpu-mem 0.85 --thinking
run llm-jp-4-8b-instruct.add_e      llm-jp--llm-jp-4-8b-instruct $TH --gpu-mem 0.85 --trust-remote-code --reasoning-effort low
run llm-jp-4-8b-thinking.add_e      llm-jp--llm-jp-4-8b-thinking $TH --gpu-mem 0.85 --trust-remote-code
run GPT-OSS-Swallow-20B.add_e       tokyotech-llm--GPT-OSS-Swallow-20B-RL-v0.1-MXFP4 $TH --gpu-mem 0.85
run Qwen3-Swallow-8B-think.add_e    tokyotech-llm--Qwen3-Swallow-8B-RL-v0.2 $TH --gpu-mem 0.85 --thinking
run Qwen3.8-27B-INT4-nothink.add_e  RedHatAI--Qwen3.8-27B-INT4 $NO --gpu-mem 0.92 --llm-kwargs "$MM"
run gemma-4-31B-it-think.add_e      google--gemma-4-31B-it-qat-w4a16-ct $TH --gpu-mem 0.92 --thinking
run Qwen3.8-27B-INT4-think.add_e    RedHatAI--Qwen3.8-27B-INT4 $TH --gpu-mem 0.92 --thinking --llm-kwargs "$MM"
run Qwen3-Swallow-32B-AWQ-think.add_e tokyotech-llm--Qwen3-Swallow-32B-RL-v0.2-AWQ-INT4 $TH --gpu-mem 0.92 --thinking
echo "=== $(date "+%F %T") 終了"
echo done > $O/DONE6
