#!/usr/bin/env python
"""gemma-4-31B（思考あり，(e) 版）の判定と，広げた相槌の規則で tag を作り直す（2026-10-21）。

  相槌：今の規則（一覧の語と完全一致）＋一覧の語の並び（tt_analysis/aizuchi_concat_20261016/moved.tsv）
  それ以外：gemma の判定（LLM が判定していた 137,812 区間と，直後に別の話者で「完了」にしていた 148,961 区間）
  gemma が読めなかった区間（.unparsed.tsv の 10 件）は今のラベルのまま。
人手正解 1000 件との一致率：(e) 版の多数決＋規則 0.832 → gemma 0.874。

  python local/turntaking/gemma_tags.py [--write]
--write なしでは集計だけ出して何も書かない。--write で data/<set>/tag を data/<set>/tag.vote_e に退避してから書き換える。
"""
import argparse
import collections
import shutil
from pathlib import Path

from vote_tags import MAP, SETS, read_votes

R = "/autofs/diamond5/share/users/kobori/relabel/"
GEMMA = [R + "full_e/gemma-4-31B-it-think.add_e", R + "full_route/gemma-4-31B-it-think.add_e.rule_nextdiff"]
MOVED = "tt_analysis/aizuchi_concat_20261016/moved.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    g = {}; bad = set()
    for f in GEMMA:
        g.update(read_votes(f + ".tsv"))
        bad |= {int(l.split("\t")[0]) for l in open(f + ".unparsed.tsv") if l.strip()}
    moved = {int(l.split("\t")[0]) for l in list(open(MOVED))[1:]}
    line2utt = {}
    for l in list(open(MAP))[1:]:
        ln, u, r = l.rstrip("\n").split("\t")
        line2utt[int(ln)] = (u, int(r))
    new = {}
    for ln, (u, r) in line2utt.items():
        if r == 2 or ln in moved:
            new[u] = 2
        elif ln in g and ln not in bad:
            new[u] = g[ln]
    print(f"gemma {len(g)} 件（読めず {len(bad)} 件は今のまま）/ 相槌に移す {len(moved)} 件")
    for s in SETS:
        p = Path("data") / s / "tag"
        rows = [l.rstrip("\n").split(" ") for l in open(p)]
        out = [(u, str(new.get(u, int(t)))) for u, t in rows]
        c0 = collections.Counter(t for _, t in rows); c1 = collections.Counter(t for _, t in out)
        ch = collections.Counter((x, y) for (_, x), (_, y) in zip(rows, out) if x != y)
        n = len(rows)
        print(f"{s:13s} {n:7d} 発話  継続/完了/相槌  今 {c0['0']/n:.3f}/{c0['1']/n:.3f}/{c0['2']/n:.3f}"
              f"  → {c1['0']/n:.3f}/{c1['1']/n:.3f}/{c1['2']/n:.3f}  変更 {sum(ch.values())}"
              f"（完了→継続 {ch[('1', '0')]}・継続→完了 {ch[('0', '1')]}・→相槌 {ch[('0', '2')] + ch[('1', '2')]}）")
        if s == "train_g":
            c = c1
            print(f"  クラス重み（完了÷各クラス）[継続, 完了, 相槌] = [{c['1']/c['0']:.2f}, 1.0, {c['1']/c['2']:.2f}]")
        if args.write:
            bak = p.with_name("tag.vote_e")
            if not bak.exists():
                shutil.copy(p, bak)
            with open(p, "w") as f:
                for u, t in out:
                    f.write(f"{u} {t}\n")
    print("書き込み済み（元は tag.vote_e）" if args.write else "（--write なし：何も書いていない）")


if __name__ == "__main__":
    main()
