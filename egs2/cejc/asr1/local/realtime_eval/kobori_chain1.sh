#!/bin/bash
# 書き出しを順に流す（2026-11-12）：開発 espm（起動済み）→ 評価 espm → 開発・評価 esp（つないだ音声なし）
D=~/2026/work/rt_eval_20261102
cd $D
until [ -e dump_dev_g_espm.done ]; do sleep 60; done
bash kobori_rt_dump.sh eval_g 6 espm weights/encac_espm.pt
bash kobori_rt_dump.sh dev_g 6 esp weights/encac_esp.pt
bash kobori_rt_dump.sh eval_g 6 esp weights/encac_esp.pt
echo 終わり $(date) > chain1.done
