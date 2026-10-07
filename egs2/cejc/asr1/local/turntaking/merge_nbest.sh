#!/usr/bin/env bash
# N-best のジョブ別出力を連結する。
#
# asr.sh は 1best_recog しか集約しない（asr.sh:2024-2026）ので、
# 2〜N 番目は logdir/output.*/ に残ったままになる。
#
#   ./local/turntaking/merge_nbest.sh <decode_dir> <dset> [N]
set -eu
D=$1; S=$2; N=${3:-5}
for n in $(seq 1 "$N"); do
    out="$D/$S/${n}best_recog"
    mkdir -p "$out"
    for f in text token token_int score; do
        src_any=0
        : > "$out/$f.tmp"
        for i in $(ls "$D/$S/logdir" | grep -oE '^output\.[0-9]+$' | sort -t. -k2 -n); do
            src="$D/$S/logdir/$i/${n}best_recog/$f"
            [ -f "$src" ] || continue
            cat "$src" >> "$out/$f.tmp"; src_any=1
        done
        if [ "$src_any" = 1 ]; then
            LC_ALL=C sort -u "$out/$f.tmp" > "$out/$f"
        fi
        rm -f "$out/$f.tmp"
    done
    echo "  ${n}best: $(wc -l < "$out/text") 行"
done
