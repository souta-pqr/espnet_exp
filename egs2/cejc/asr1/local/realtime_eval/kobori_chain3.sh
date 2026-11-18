#!/bin/bash
# 選び直した決め方（無音 0.4 秒・0.84／無音 1.5 秒・0.6／無音 3 秒で P(相槌) 0.7 未満）で評価データを実際に流して確かめる（2026-11-13．chain2 の後）
D=~/2026/work/rt_eval_20261102
until [ -e $D/chain2.done ]; do sleep 60; done
cd $D
echo "=== $(date) 確認の実走"
bash kobori_rt_real.sh eval_g 6 real_espm_p weights/encac_espm.pt --g1 0.4 --th1 0.84 --g2 1.5 --th2 0.6 --S 3 --phi 0.7
echo 終わり $(date) > $D/chain3.done
