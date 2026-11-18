#!/bin/bash
# Easy Turn（書き起こしを使わない版）の評価データのダンプが終わったら，silver12 へ写して評価の 3 段目だけ流し直す（2026-11-04）
cd ~/2025/espnet/egs2/cejc/asr1/tt_preds
until [ -e easyturn_notext_dump.done ]; do sleep 300; done
scp -q easyturn_notext_eval_g_post5.0_noend.tsv silver12:2026/espnet_tt/egs2/cejc/asr1/tt_preds/ || exit 1
ssh silver12 'cd ~/2026/espnet_tt/egs2/cejc/asr1 && bash tt_analysis/longlat_20261024/run_eval_new.sh easyturnnotext easyturnne gt > tt_batch_logs/eval_easyturnnotext.log 2>&1; touch tt_batch_logs/eval_easyturnnotext.done'
