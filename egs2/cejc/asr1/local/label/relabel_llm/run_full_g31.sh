#!/bin/bash
# gemma-4-31B-it 思考あり・(e) 版プロンプトで全データを判定する（2026-10-12〜、約 76 時間）。
# 人手正解 922 件の確認で単体 0.868（今の多数決 0.836）。ただし同じ 922 件で選んだので、採用は別の人手正解で確かめてから。
# モデルは /data/huggingface/hub の共有キャッシュを読むだけ。run_full.py に --family・--llm-kwargs を足した（元は .bak_g）。
# 最大長 7680（31B の思考ありは 10240 だと KV が足りない）・出力 2560（gold で最大 1712）。止めても続きから再開できる。
set -uo pipefail
export PATH=$HOME/anaconda3/envs/llm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/relabel_llm
R=/autofs/diamond5/share/users/kobori/relabel
mkdir -p logs_g31
M=$(ls -d /data/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots/*/ | head -1)
echo running > logs_g31/state
echo "=== $(date '+%F %T') 本番"
python run_full.py --model $M --name gemma-4-31B-it-think.add_e --family gemma --thinking \
  --max-tokens 2560 --max-model-len 7680 --gpu-mem 0.92 --chunk 256 \
  --system $R/prompt_check/system_add_e.txt --outdir $R/full_e || { echo error > logs_g31/state; exit 1; }
echo done > logs_g31/state
echo "=== $(date '+%F %T') 終了"
