#!/usr/bin/env python
"""CEJC を「話者が重ならない」学習・開発・評価に分け直し、ESPnet の data/<set>/ を作る（2026-10-05）。

旧分割（section-detection/exp19/rebuild_splits.py）は会話グループごとに 1〜2 会話を評価へ移しており、
評価の発話の 48.5% が学習にもいる話者のものだった。開発は 1 グループだけだった。
CEJC の 51 グループどうしは話者を 1 人も共有しない（メタ情報の話者・会話対応表で確認）ので、
**グループ単位**で振り分ければ話者が重ならない。

- 発話: 時系列テキスト（section-detection/exp17/chronological）のうち、LLM ラベルがあるもの全部
  （旧データと同じ規則。各会話の先頭 10・末尾 10 はラベルが無いので入らない）
- テキスト: format_text_kanji.format_line（旧データの text を 234,621 発話すべて再現できることを確認済み）
- ラベル: Qwen3-8B full_context/oneshot（exp27 の PRIMARY）を**時刻で突き合わせた**もの
  （tt_analysis/qwen3_fc_os_uttid.tsv、local/turntaking/gold1000_map.py）。0=継続 / 1=完了 / 2=相槌
- 音声: CEJC_safia（旧データと同じ）
- 振り分け: 種類（C/K/S/T/W）ごとに乱数で選ぶ。評価・開発とも T 2・K 1・W 1・C か S 1 の 5 グループ

  python local/cejc_groupsplit.py --seed 0
"""
import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "section-detection/exp19")
from format_text_kanji import format_line  # noqa: E402
from make_text_kanji import make_utt_id, parse_chronological  # noqa: E402

SAFIA = "/autofs/diamond2/share/corpus/CEJC_safia/data"
UR = re.compile(r"^開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)(?:\s+<\w+>)?\s*$")
# 種類ごとに評価・開発へ回すグループ数（残りは学習）
PER_TYPE = {"T": 2, "K": 1, "W": 1, "CS": 1}


def wav_path(reco):
    conv, mic = reco.rsplit("_", 1)
    sess = re.sub(r"[a-z]$", "", conv)
    return f"{SAFIA}/{conv[:4]}/{sess}/{reco}.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chrono", default="section-detection/exp17/chronological")
    ap.add_argument("--labels", default="tt_analysis/qwen3_fc_os_uttid.tsv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--suffix", default="_g", help="train<suffix> / dev<suffix> / eval<suffix>")
    args = ap.parse_args()

    lab = {}
    for line in list(open(args.labels, encoding="utf-8"))[1:]:
        _, u, r = line.rstrip("\n").split("\t")
        lab[u] = r

    utts = {}  # uid -> (reco, start, end, text, tag)
    for cid, lines in parse_chronological(args.chrono):
        for x in lines:
            s, e, spk, t = UR.match(x).groups()
            uid = make_utt_id(cid, spk, float(s), float(e))
            if uid not in lab:
                continue
            out = format_line(f"{uid} <yes> {t}")
            text = out.split(" ", 2)[2] if out.count(" ") >= 2 else ""
            if not text.strip():
                continue
            reco = "_".join(uid.split("_")[:3])
            utts[uid] = (reco, float(s), float(e), text, lab[uid])

    hours = collections.Counter()
    for uid, (reco, s, e, *_r) in utts.items():
        hours[uid[:4]] += (e - s) / 3600
    groups = sorted(hours)
    rng = np.random.default_rng(args.seed)
    split = {g: "train" for g in groups}
    for key, n in PER_TYPE.items():
        cand = [g for g in groups if g[0] in key]
        pick = rng.permutation(cand)[: 2 * n]
        for g in pick[:n]:
            split[g] = "eval"
        for g in pick[n:]:
            split[g] = "dev"

    by_set = collections.defaultdict(list)
    for uid in sorted(utts):
        by_set[split[uid[:4]]].append(uid)
    missing = sorted({utts[u][0] for u in utts if not Path(wav_path(utts[u][0])).exists()})
    if missing:
        print("音声が無い録音:", missing[:10], len(missing))
        missing = set(missing)

    for name in ("train", "dev", "eval"):
        d = Path(args.out) / f"{name}{args.suffix}"
        d.mkdir(parents=True, exist_ok=True)
        ids = [u for u in by_set[name] if utts[u][0] not in missing]
        recos = sorted({utts[u][0] for u in ids})
        with open(d / "text", "w") as ft, open(d / "tag", "w") as fg, \
                open(d / "segments", "w") as fs, open(d / "utt2spk", "w") as fu:
            for u in ids:
                reco, s, e, text, tag = utts[u]
                ft.write(f"{u} {text}\n")
                fg.write(f"{u} {tag}\n")
                fs.write(f"{u} {reco} {s:.3f} {e:.3f}\n")
                fu.write(f"{u} {'_'.join(u.split('_')[:2])}\n")
        with open(d / "wav.scp", "w") as fw:
            for r in recos:
                fw.write(f"{r} {wav_path(r)}\n")
        s2u = collections.defaultdict(list)
        for u in ids:
            s2u["_".join(u.split("_")[:2])].append(u)
        with open(d / "spk2utt", "w") as f:
            for spk in sorted(s2u):
                f.write(f"{spk} {' '.join(s2u[spk])}\n")
        gs = sorted(g for g in groups if split[g] == name)
        h = sum((utts[u][2] - utts[u][1]) for u in ids) / 3600
        tags = collections.Counter(utts[u][4] for u in ids)
        print(f"{name:5s} グループ {len(gs):2d}  発話 {len(ids):7d}  {h:6.1f} 時間  "
              f"継続/完了/相槌 {tags['0']}/{tags['1']}/{tags['2']}"
              + (f"  {' '.join(gs)}" if name != "train" else ""))


if __name__ == "__main__":
    main()
