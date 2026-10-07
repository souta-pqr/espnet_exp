#!/usr/bin/env python
"""cejc_groupsplit.py で漏れた 8 会話を、Qwen3 の結果ファイルから直接取り出して追加分の data を作る（2026-10-07）。

漏れた理由：結果ファイルは会話を番号でしか持たず、手元の時系列テキスト（exp17/chronological, 569 会話）で
ID に対応づけたため、そこに無い 8 会話が落ちた。結果ファイルの会話番号は CEJC_safia の 575 会話を ID 順に並べた
順番と一致する（対応済みの 567 会話で全一致を確認）ので、その順で ID を決める。
8 会話のグループ（C002・K003・K004・T004・T013）はすべて学習側なので、data/train_g_extra に書く。
LINE → 発話 ID の対応も tt_analysis/qwen3_fc_os_uttid.tsv に追記する。

  python local/cejc_groupsplit_extra.py
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, "section-detection/exp19")
from format_text_kanji import format_line  # noqa: E402
from make_text_kanji import make_utt_id  # noqa: E402

RES = "/autofs/diamond5/share/users/kobori/result_oneshot.txt"
SAFIA = "/autofs/diamond2/share/corpus/CEJC_safia/data"
MAP = "tt_analysis/qwen3_fc_os_uttid.tsv"
OUT = Path("data/train_g_extra")


def wav_path(reco):
    conv, mic = reco.rsplit("_", 1)
    sess = re.sub(r"[a-z]$", "", conv)
    return f"{SAFIA}/{conv[:4]}/{sess}/{reco}.wav"


def main():
    convs = sorted({p.name.rsplit("_", 1)[0] for p in Path(SAFIA).glob("*/*/*.wav")})
    assert len(convs) == 575, len(convs)
    have = {int(l.split("\t")[0]) for l in list(open(MAP))[1:]}
    LR = re.compile(r"^\[LINE (\d+)\] 会話=(\d+)")
    TR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")
    rows = []; ln = c = None; sec = None; tgt = None
    for raw in open(RES, encoding="utf-8"):
        if raw.startswith("[LINE"):
            m = LR.match(raw); ln, c = int(m[1]), int(m[2]); sec = None; tgt = None; continue
        if ln in have:
            continue
        if raw.startswith("[TARGET]"): sec = "t"; continue
        if raw.startswith("[POST_HISTORY]") or raw.startswith("[HISTORY]"): sec = None; continue
        if sec == "t" and raw.startswith("- "):
            tgt = TR.match(raw.rstrip("\n")).groups(); sec = None; continue
        if raw.startswith("[RESULT]") and tgt:
            s, e, spk, t = tgt
            cid = convs[c - 1]
            rows.append((ln, make_utt_id(cid, spk, float(s), float(e)), int(raw.split()[1]), cid, s, e, t))
    print(f"追加する区間 {len(rows)}（会話 {sorted({r[3] for r in rows})}）")
    with open(MAP, "a") as f:
        for ln, u, r, *_ in rows:
            f.write(f"{ln}\t{u}\t{r}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    utts = {}
    for ln, u, r, cid, s, e, t in rows:
        out = format_line(f"{u} <yes> {t}")
        text = out.split(" ", 2)[2] if out.count(" ") >= 2 else ""
        if text.strip():
            utts[u] = ("_".join(u.split("_")[:3]), float(s), float(e), text, r)
    ids = sorted(utts)
    with open(OUT / "text", "w") as ft, open(OUT / "tag", "w") as fg, \
            open(OUT / "segments", "w") as fs, open(OUT / "utt2spk", "w") as fu:
        for u in ids:
            reco, s, e, text, tag = utts[u]
            ft.write(f"{u} {text}\n"); fg.write(f"{u} {tag}\n")
            fs.write(f"{u} {reco} {s:.3f} {e:.3f}\n"); fu.write(f"{u} {'_'.join(u.split('_')[:2])}\n")
    with open(OUT / "wav.scp", "w") as fw:
        for r in sorted({utts[u][0] for u in ids}):
            fw.write(f"{r} {wav_path(r)}\n")
    s2u = collections.defaultdict(list)
    for u in ids:
        s2u["_".join(u.split("_")[:2])].append(u)
    with open(OUT / "spk2utt", "w") as f:
        for spk in sorted(s2u):
            f.write(f"{spk} {' '.join(s2u[spk])}\n")
    h = sum(utts[u][2] - utts[u][1] for u in ids) / 3600
    tags = collections.Counter(utts[u][4] for u in ids)
    print(f"{OUT}: 発話 {len(ids)} / {h:.2f} 時間 / 継続・完了・相槌 {tags[0]}/{tags[1]}/{tags[2]}"
          f"（空テキストで除外 {len(rows) - len(ids)}）")


if __name__ == "__main__":
    main()
