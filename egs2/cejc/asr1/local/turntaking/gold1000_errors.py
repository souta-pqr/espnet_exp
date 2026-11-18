#!/usr/bin/env python
"""ラベル付けの LLM（3 モデルの多数決）が人手正解とどこで食い違うかを調べる（2026-10-09）。

対象は人手正解のうち LLM が判定する 922 件。票は正本（tt_analysis/relabel_votes_final/）。
プロンプトの本文から、判定対象発話の文末の形・長さ・同じ話者が再び話すまでの時間を取り出し、
その特徴ごとに人手と多数決の「完了」の割合と食い違いの向きを数える。

  python local/turntaking/gold1000_errors.py [--dump tt_analysis/gold1000_errors.tsv]
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote_tags import read_votes                                   # noqa: E402

PROMPTS = "tt_analysis/gold1000_prompts.jsonl"
V = "tt_analysis/relabel_votes_final/"
TAG = re.compile(r"</?[^<>\n]{1,20}?と重複>")
TGT = re.compile(r"【判定対象発話】\n\[(.+?) \| ([\d.]+)秒\] (.*)\n同話者継続間隔: (.+)")
GAP = re.compile(r"^([\d.]+)秒（その間に(他話者の発話なし|他話者(\d+)発話あり)")

# 文末の形（上から順に当てはめる）
ENDINGS = [
    ("フィラー・言い直しで終わる", r"\((F|D) [^()]*\)$"),
    ("接続の形（て・で・けど・から・し・が 等）", r"(て|で|けど|けれど|けども|から|ので|し|が|のに|たり|とか|って|ても|でも|ながら)$"),
    ("問いかけ（か・の・疑問詞 等）", r"(か|かな|かね|の|っけ|なに|何|なんで|どう|どこ|いつ|だれ|誰|ない\?|\?)$"),
    ("同意を求める形（よね・じゃん・でしょ 等）", r"(よね|だよね|でしょ|でしょう|じゃん|じゃない|もんね|ね|よな|な)$"),
    ("言い切り（だ・です・ます・た・よ 等）", r"(だ|です|ます|た|よ|わ|ぞ|ません|ました|でした|だった|ちゃった|る|い)$"),
]


def ending(t):
    t = TAG.sub("", t).strip()
    for name, pat in ENDINGS:
        if re.search(pat, t):
            return name
    return "その他（名詞・感動詞 等で終わる）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    v32 = {}
    for f in ("Qwen3-32B-AWQ-nothink.tsv", "Qwen3-32B-AWQ-nothink.silver13.tsv"):
        v32.update(read_votes(V + f))
    v8t = read_votes(V + "Qwen3-8B-think.tsv")
    rows = []
    for l in open(PROMPTS, encoding="utf-8"):
        r = json.loads(l)
        if r["route"] != "llm":
            continue
        body = r["user"].rsplit("---", 1)[-1]          # 例を除いた本体
        m = TGT.search(body)
        spk, dur, text, gapinfo = m.group(1), float(m.group(2)), m.group(3), m.group(4)
        g = GAP.match(gapinfo)
        gap = float(g.group(1)) if g else None
        others = int(g.group(3)) if g and g.group(3) else 0
        ln = r["line_no"]
        vote = int(r["qwen"] + v32[ln] + v8t[ln] >= 2)
        rows.append(dict(line_no=ln, gold=r["gold"], qwen=r["qwen"], vote=vote, text=TAG.sub("", text),
                         dur=dur, gap=gap, others=others, end=ending(text)))

    def table(key, title, order=None):
        c = collections.defaultdict(list)
        for x in rows:
            c[key(x)].append(x)
        print(f"\n■ {title}")
        print(f"{'':42s}{'件数':>5s}{'人手 完了率':>11s}{'多数決 完了率':>12s}{'一致率':>8s}{'人手完了→継続':>14s}{'人手継続→完了':>14s}")
        for k in (order or sorted(c)):
            xs = c.get(k, [])
            if not xs:
                continue
            n = len(xs)
            fn = sum(x["gold"] == 1 and x["vote"] == 0 for x in xs)
            fp = sum(x["gold"] == 0 and x["vote"] == 1 for x in xs)
            print(f"{k:42s}{n:5d}{sum(x['gold'] for x in xs)/n:11.2f}{sum(x['vote'] for x in xs)/n:12.2f}"
                  f"{sum(x['gold'] == x['vote'] for x in xs)/n:8.2f}{fn:14d}{fp:14d}")

    table(lambda x: x["end"], "文末の形", [n for n, _ in ENDINGS] + ["その他（名詞・感動詞 等で終わる）"])

    def gapbin(x):
        if x["gap"] is None:
            return "5 同じ話者が再び話さない"
        if x["others"] == 0:
            return "1 間に他の話者なし（" + ("1 秒未満）" if x["gap"] < 1 else "1 秒以上）")
        return "3 間に他の話者あり（" + ("1 秒未満）" if x["gap"] < 1 else "1 秒以上）")
    table(gapbin, "判定対象の後，同じ話者が再び話すまで")
    table(lambda x: "1 0.5 秒未満" if x["dur"] < 0.5 else ("2 0.5〜1.5 秒" if x["dur"] < 1.5 else "3 1.5 秒以上"),
          "判定対象発話の長さ")
    n = len(rows)
    print(f"\n全体 {n} 件：一致率 {sum(x['gold'] == x['vote'] for x in rows)/n:.3f}，"
          f"人手完了→継続 {sum(x['gold'] == 1 and x['vote'] == 0 for x in rows)}，"
          f"人手継続→完了 {sum(x['gold'] == 0 and x['vote'] == 1 for x in rows)}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            f.write("line_no\tgold\tvote\tqwen\tend\tdur\tgap\tothers\ttext\n")
            for x in rows:
                f.write(f"{x['line_no']}\t{x['gold']}\t{x['vote']}\t{x['qwen']}\t{x['end']}\t{x['dur']}\t"
                        f"{x['gap']}\t{x['others']}\t{x['text']}\n")


if __name__ == "__main__":
    main()
