#!/bin/bash
# 11/22：chain26 に閾値 0.70 を飛ばさせるための印を，chain26 が終わったら消す（後で流すため）
D=~/2026/work/rt_eval_20261102
until [ -e $D/chain26.done ]; do sleep 120; done
grep -q 飛ばす印 $D/eval_g_et_th0.70.done && rm $D/eval_g_et_th0.70.done
