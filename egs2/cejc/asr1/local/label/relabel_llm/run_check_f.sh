#!/bin/bash
# (f) 版プロンプトの確認と，規則で「完了」にしていた区間（rule_nextdiff）の確認（2026-10-16 準備）。
# gemma-4-31B-it 思考あり．本番と同じ run_full.py で，(e) 版と (f) 版を同じ条件で流す（思考ありは条件が違うと判定が 1 割ほど入れ替わるため）。
#   1. 人手正解 1000 件（LLM の 922 件 ＋ 規則で完了の 78 件）
#   2. 意味で確かめた 270 件（Claude の判定と比べる）
# 1000 件で約 35 分，270 件で約 10 分の見込み（全データ 137,812 件が約 76 時間だったことから）。
# 比べ方：egs2/cejc/asr1 で python local/turntaking/compare_route_check.py
# 出力：/autofs/diamond5/share/users/kobori/relabel/prompt_check/out_route/
set -uo pipefail
export PATH=$HOME/anaconda3/envs/llm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/relabel_llm
R=/autofs/diamond5/share/users/kobori/relabel
O=$R/prompt_check/out_route
mkdir -p $O
M=$(ls -d /data/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/ | head -1)
run() {  # 名前 プロンプト 追加引数...
  local n=$1 v=$2; shift 2
  echo "=== $(date '+%F %T') $n"
  python run_full.py --model $M --family gemma --thinking --max-tokens 2560 --max-model-len 7680 --gpu-mem 0.92 --chunk 256 \
      --routes "llm rule_nextdiff" --system $R/prompt_check/system_$v.txt --name $n --outdir $O "$@" || echo "失敗: $n"
}
run gemma-4-31B-it-think.add_e.gold1000 add_e --only-gold
run gemma-4-31B-it-think.add_f.gold1000 add_f --only-gold
run gemma-4-31B-it-think.add_e.check270 add_e --lines $R/prompt_check/lines_check270.txt
run gemma-4-31B-it-think.add_f.check270 add_f --lines $R/prompt_check/lines_check270.txt
echo "=== $(date '+%F %T') 終了"
echo done > $O/DONE
