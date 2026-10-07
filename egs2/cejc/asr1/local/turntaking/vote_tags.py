#!/usr/bin/env python
"""3 つの LLM 判定の多数決で、新しい分割（train_g / dev_g / eval_g）の tag を作り直す（2026-10-07）。

  票 1: 今の Qwen3-8B 思考なし（Qwen3 の結果ファイル。今の data/*_g/tag と同じ）
  票 2: Qwen3-32B-AWQ 思考なし（silver9）
  票 3: Qwen3-8B 思考あり（silver13）
人手正解 922 件で、多数決は一致率 0.762 → 0.810、継続 F1 0.717 → 0.805（docs/label_vote_20261007.md）。

ルールで決まる区間（相槌＝2、直後に別話者＝1）は変えない。票 2・3 が無い LINE（ルールの区間）は今のまま。
票 2・3 は LINE 番号で書かれている（relabel/full/<名前>.tsv: LINE<TAB>はい/いいえ）。

  python local/turntaking/vote_tags.py --v32b <tsv> --v8think <tsv> [--write]
--write なしでは集計だけ出して何も書かない。--write で data/<set>/tag を data/<set>/tag.qwen に退避してから書き換える。
"""
import argparse
import collections
import shutil
from pathlib import Path

MAP = "tt_analysis/qwen3_fc_os_uttid.tsv"     # LINE → 発話 ID, Qwen3 の結果
SETS = ("train_g", "train_g_extra", "dev_g", "eval_g")   # train_g_extra は漏れていた 8 会話（local/cejc_groupsplit_extra.py）


def read_votes(path):
    v = {}
    for l in open(path, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].isdigit() and p[1] in ("はい", "いいえ"):
            v[int(p[0])] = 1 if p[1] == "はい" else 0
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v32b", required=True)
    ap.add_argument("--v8think", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    a = read_votes(args.v32b); b = read_votes(args.v8think)
    line2utt = {}; qwen = {}
    for l in list(open(MAP))[1:]:
        ln, u, r = l.rstrip("\n").split("\t")
        line2utt[int(ln)] = u; qwen[u] = int(r)
    print(f"票 2（32B）{len(a)} 件 / 票 3（8B 思考あり）{len(b)} 件 / 両方ある {len(a.keys() & b.keys())} 件")
    new = {}; change = collections.Counter()
    for ln, u in line2utt.items():
        q = qwen[u]
        if q == 2 or ln not in a or ln not in b:
            continue                                   # 相槌・ルールの区間・票が欠けた区間は今のまま
        v = int(q + a[ln] + b[ln] >= 2)
        new[u] = v; change[(q, v)] += 1
    print(f"多数決を取った区間 {len(new)} / 変わる区間 {change[(0, 1)] + change[(1, 0)]}"
          f"（継続→完了 {change[(0, 1)]}・完了→継続 {change[(1, 0)]}）")
    for s in SETS:
        p = Path("data") / s / "tag"
        rows = [l.rstrip("\n").split(" ") for l in open(p)]
        out = [(u, str(new.get(u, int(t)))) for u, t in rows]
        c0 = collections.Counter(t for _, t in rows); c1 = collections.Counter(t for _, t in out)
        n = len(rows)
        print(f"{s:8s} {n:7d} 発話  継続/完了/相槌  今 {c0['0']/n:.3f}/{c0['1']/n:.3f}/{c0['2']/n:.3f}"
              f"  → 多数決 {c1['0']/n:.3f}/{c1['1']/n:.3f}/{c1['2']/n:.3f}"
              f"  変更 {sum(x != y for (_, x), (_, y) in zip(rows, out))}")
        if s == "train_g":
            c = c1
            print(f"  クラス重み（完了÷各クラス）[継続, 完了, 相槌] = "
                  f"[{c['1']/c['0']:.2f}, 1.0, {c['1']/c['2']:.2f}]")
        if args.write:
            bak = p.with_name("tag.qwen")
            if not bak.exists():
                shutil.copy(p, bak)
            with open(p, "w") as f:
                for u, t in out:
                    f.write(f"{u} {t}\n")
    print("書き込み済み（元は tag.qwen）" if args.write else "（--write なし：何も書いていない）")


if __name__ == "__main__":
    main()
