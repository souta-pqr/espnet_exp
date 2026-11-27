#!/bin/bash
cd ~/2026/work/rt_eval_20261102
for d in dev_g eval_g; do echo "=== $(date) $d"; bash kobori_rt_dump10.sh $d 6 espmk1gs weights/encac_espmk1gs.pt; done
