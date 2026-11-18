#!/bin/bash
# 2026-11-16：文字を当てる課題を足した版の名前を espmk に付け直して，CEJC の実時間のスコアを書き出す．
#   chain14 で espmt と名付けたが，11/13 の別の版（最後の 30 フレームの平均も読む版）と同じ名前で，dump_{dev,eval}_g_espmt が 11/13 のまま残っていて書き出しが飛ばされたため．
#   weights/encac_espmt.pt は 11/16 の版で上書きされている（11/13 の版は exp/encac_espmt_20261113/best.pt から作り直せる）
D=~/2026/work/rt_eval_20261102
until [ -e $D/chain15.done ] || ! pgrep -f '[k]obori_chain15[.]sh' > /dev/null; do sleep 60; done
cd ~/2026/work/exp-realtime-epd
cp -p weights/encac_espmt.pt weights/encac_espmk.pt; cp -p weights/encac_espmt_ref.npz weights/encac_espmk_ref.npz
cd $D
for v in 1911 1712 1902; do [ -d dump_hazumi${v}_espmt ] && mv dump_hazumi${v}_espmt dump_hazumi${v}_espmk; done
for d in dev_g eval_g; do echo "=== $(date) 実時間のスコア espmk $d"; bash kobori_rt_dump.sh $d 6 espmk weights/encac_espmk.pt; done
echo 終わり $(date) > $D/chain16.done
