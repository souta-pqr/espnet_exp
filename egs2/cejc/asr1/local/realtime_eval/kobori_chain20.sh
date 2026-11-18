#!/bin/bash
# 2026-11-18：文字を当てる課題を足した版（espmk1）を，選んだ決め方（無音 0.4 秒・0.84／無音 2 秒・0.4／無音 3 秒で相槌 0.1 未満）で評価データに実際に流して確かめる
cd ~/2026/work/rt_eval_20261102
bash kobori_rt_real.sh eval_g 6 real_espmk1_p weights/encac_espmk1.pt --g1 0.4 --th1 0.84 --g2 2.0 --th2 0.4 --S 3 --phi 0.1
echo 終わり $(date) > chain20.done
