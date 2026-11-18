#!/usr/bin/env python
"""相槌の規則を「相槌の一覧の語を並べただけの発話」に広げたとき，何件が LLM 判定から相槌に移るかを数える（2026-10-16）。

今の規則（detect_section.text.py の is_aizuchi）は，正規化した発話が AIZUCHI_SET の語に完全一致するときだけ相槌にする。
そのため「ふーんうんうん」「あっあーああーあ」のように一覧の語を並べたものは LLM の判定に回り，はい／いいえの 2 択で付けられる。
ここでは，正規化した発話を AIZUCHI_SET の語（後ろに伸ばし「ー」が付いてもよい）の並びに分けられるときに相槌とする。

  python local/turntaking/aizuchi_concat_check.py --out tt_analysis/aizuchi_concat_20261016
出力：<out>/moved.tsv（line_no，人手正解，gemma，多数決，発話），標準出力に集計
"""
import argparse
import collections
import importlib.util
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote_tags import read_votes                                     # noqa: E402

LAB = "/home/kobori/work/cejc-label"
spec = importlib.util.spec_from_file_location("det27", f"{LAB}/detect_section.text.py")
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)
RES = "/autofs/diamond5/share/users/kobori/result_oneshot.txt"
R = "/autofs/diamond5/share/users/kobori/relabel/full_e/"
LR = re.compile(r"^\[LINE (\d+)\]")
UR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")
UNITS = sorted(det.AIZUCHI_SET, key=len, reverse=True)


def is_aizuchi_concat(text):
    s = re.sub(r"[\s、。，．・!?！？]", "", det._normalize_for_aizuchi(text))

    @lru_cache(None)
    def ok(i):
        if i == len(s):
            return True
        for u in UNITS:
            if s.startswith(u, i):
                j = i + len(u)
                while j < len(s) and s[j] == "ー":
                    j += 1
                if ok(j):
                    return True
        return False
    return s != "" and ok(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = {int(l.split("\t")[1]): int(l.split("\t")[3]) for l in list(open("eval-1000/gold.txt"))[1:]}
    vg = read_votes(R + "gemma-4-31B-it-think.add_e.tsv")
    v32 = read_votes(R + "Qwen3-32B-AWQ-nothink.add_e.tsv")
    v8 = read_votes(R + "Qwen3-8B-think.add_e.tsv")
    moved = []; n_llm = 0; cur = None; blk = None; sec = None
    for raw in open(RES, encoding="utf-8"):
        if raw.startswith("[LINE"):
            cur = int(LR.match(raw)[1]); blk = {"t": None, "p": []}; sec = None; continue
        if cur is None:
            continue
        if raw.startswith("[HISTORY]"): sec = "h"; continue
        if raw.startswith("[TARGET]"): sec = "t"; continue
        if raw.startswith("[POST_HISTORY]"): sec = "p"; continue
        if raw.startswith("[RESULT]"):
            t = blk["t"]
            if t is not None and not det.is_aizuchi(t) and not det.is_turn_end_next_different_speaker(t, blk["p"]):
                n_llm += 1
                if is_aizuchi_concat(t.text):
                    q = int(raw.split()[1])
                    vote = int(q + v32.get(cur, 0) + v8.get(cur, 0) >= 2)
                    moved.append((cur, gold.get(cur, ""), vg.get(cur, ""), vote, t.text))
            cur = None; continue
        m = UR.match(raw.rstrip("\n"))
        if m and sec in ("t", "p"):
            u = det.EvalUtterance(*m.groups())
            if sec == "t": blk["t"] = u
            else: blk["p"].append(u)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "moved.tsv", "w", encoding="utf-8") as f:
        f.write("line_no\tgold\tgemma\tvote\ttext\n")
        for r in moved:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"LLM 判定の区間 {n_llm}，相槌に移る区間 {len(moved)}（{len(moved) / n_llm:.2%}）")
    g = [r for r in moved if r[1] != ""]
    print(f"うち人手正解 922 件に入るもの {len(g)}：人手 完了 {sum(r[1] for r in g)} / 継続 {sum(1 - r[1] for r in g)}")
    gm = [r for r in moved if r[2] != ""]
    print(f"gemma が完了とした割合 {sum(r[2] for r in gm) / max(len(gm), 1):.3f}，多数決 {sum(r[3] for r in moved) / max(len(moved), 1):.3f}")
    c = collections.Counter(det._normalize_for_aizuchi(r[4]) for r in moved)
    print("多いもの：", "，".join(f"{k}({v})" for k, v in c.most_common(30)))


if __name__ == "__main__":
    main()
