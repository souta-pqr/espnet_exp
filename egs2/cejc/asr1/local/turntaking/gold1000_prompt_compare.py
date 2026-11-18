#!/usr/bin/env python
"""簡単にしたプロンプトで人手正解 922 件を付け直した結果を比べる（2026-10-09）。

元のプロンプトの結果は silver13 で同じ vLLM・同じ Qwen3-8B（思考なし）で出した Qwen3-8B-nothink-vllm.tsv。
変種は local/turntaking/gold1000_prompt_variants.py で作り、silver13 の run_prompt_check*.sh で判定した。
各変種を 3 モデルの多数決の 1 票目（元の Qwen3-8B）と入れ替えた場合も出す（残り 2 票は gold 922 件の判定結果）。

  python local/turntaking/gold1000_prompt_compare.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold1000_errors import ending, ENDINGS                          # noqa: E402
from vote_tags import read_votes                                     # noqa: E402

R = "/autofs/diamond5/share/users/kobori/relabel/"
OUT = R + "prompt_check/out/"
NAMES = [("元のプロンプト", R + "Qwen3-8B-nothink-vllm.tsv"),
         ("基準を定義に置き換え", OUT + "Qwen3-8B-nothink.def.tsv"),
         ("重なりタグを外す", OUT + "Qwen3-8B-nothink.noovl.tsv"),
         ("定義＋重なりタグを外す", OUT + "Qwen3-8B-nothink.def_noovl.tsv"),
         ("基準に (e) 言い切りを追加", OUT + "Qwen3-8B-nothink.add_e.tsv")]


def f1(y, p, c):
    tp = np.sum((y == c) & (p == c))
    return 2 * tp / max(np.sum(y == c) + np.sum(p == c), 1)


def main():
    rows = [json.loads(l) for l in open("tt_analysis/gold1000_prompts.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["route"] == "llm"]
    ln = [r["line_no"] for r in rows]
    y = np.array([r["gold"] for r in rows])
    ends = [ending(r["user"].rsplit("---", 1)[-1].split("【判定対象発話】\n", 1)[1].split("\n", 1)[0].split("] ", 1)[1])
            for r in rows]
    v32 = read_votes(R + "Qwen3-32B-AWQ-nothink.tsv")
    v8t = read_votes(R + "Qwen3-8B-think.tsv")
    o32 = np.array([v32[l] for l in ln]); o8t = np.array([v8t[l] for l in ln])
    preds = {}
    for nm, p in NAMES:
        if Path(p).exists():
            v = read_votes(p)
            if all(l in v for l in ln):
                preds[nm] = np.array([v[l] for l in ln])
    base = preds["元のプロンプト"]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(2000)]

    def ci(p, q):
        d = [np.mean(p[s] == y[s]) - np.mean(q[s] == y[s]) for s in idx]
        return np.mean(p == y) - np.mean(q == y), np.percentile(d, 2.5), np.percentile(d, 97.5)

    print(f"人手正解 {len(y)} 件（人手の完了率 {y.mean():.2f}）／Qwen3-8B 思考なし")
    print(f"{'':28s}{'一致率':>7s}{'継続F1':>8s}{'完了F1':>8s}{'完了率':>7s}   元との差 [95%]")
    for nm, p in preds.items():
        d, lo, hi = ci(p, base)
        print(f"{nm:28s}{np.mean(p == y):7.3f}{f1(y, p, 0):8.3f}{f1(y, p, 1):8.3f}{p.mean():7.2f}   "
              f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n文末の形ごとの一致率")
    order = [n for n, _ in ENDINGS] + ["その他（名詞・感動詞 等で終わる）"]
    print(f"{'':42s}{'件数':>5s}" + "".join(f"{nm[:8]:>10s}" for nm in preds))
    for e in order:
        m = np.array([x == e for x in ends])
        print(f"{e:42s}{m.sum():5d}" + "".join(f"{np.mean(p[m] == y[m]):10.2f}" for p in preds.values()))

    print("\n多数決の 1 票目（Qwen3-8B 思考なし）を入れ替えた場合（2・3 票目は Qwen3-32B 思考なし・Qwen3-8B 思考あり）")
    vb = ((base + o32 + o8t) >= 2).astype(int)
    for nm, p in preds.items():
        v = ((p + o32 + o8t) >= 2).astype(int)
        d, lo, hi = ci(v, vb)
        print(f"{nm:28s}一致率 {np.mean(v == y):.3f}  継続F1 {f1(y, v, 0):.3f}  完了F1 {f1(y, v, 1):.3f}  "
              f"完了率 {v.mean():.2f}  元の多数決との差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # 思考ありの 8B（3 票目）に (e) を加えたプロンプトを使った場合
    th = OUT + "Qwen3-8B-think.add_e.tsv"
    if Path(th).exists() and all(l in read_votes(th) for l in ln):
        vt = read_votes(th); te = np.array([vt[l] for l in ln])
        # 正本（全件付け直しの本番）の 8B 思考ありの票
        fin = read_votes("tt_analysis/relabel_votes_final/Qwen3-8B-think.tsv")
        tf = np.array([fin[l] for l in ln])
        q = np.array([r["qwen"] for r in rows])          # 元のラベル（多数決の 1 票目の正本）
        a32 = {}
        for f in ("Qwen3-32B-AWQ-nothink.tsv", "Qwen3-32B-AWQ-nothink.silver13.tsv"):
            a32.update(read_votes("tt_analysis/relabel_votes_final/" + f))
        f32 = np.array([a32[l] for l in ln])
        print("\nQwen3-8B 思考あり：元のプロンプト（正本の票）と (e) を加えたプロンプト")
        for nm, p in (("元のプロンプト", tf), ("(e) を加えたプロンプト", te)):
            print(f"{nm:28s}一致率 {np.mean(p == y):.3f}  継続F1 {f1(y, p, 0):.3f}  完了F1 {f1(y, p, 1):.3f}  完了率 {p.mean():.2f}")
        print(f"{'文末の形':40s}{'件数':>5s}{'元':>8s}{'(e)':>8s}")
        for e in order:
            m = np.array([x == e for x in ends])
            print(f"{e:40s}{m.sum():5d}{np.mean(tf[m] == y[m]):8.2f}{np.mean(te[m] == y[m]):8.2f}")
        vf = ((q + f32 + tf) >= 2).astype(int)
        ve = ((q + f32 + te) >= 2).astype(int)
        d, lo, hi = ci(ve, vf)
        print("\n多数決（正本の 1・2 票目＋3 票目）")
        for nm, v in (("正本の多数決", vf), ("3 票目を (e) 版に", ve)):
            print(f"{nm:28s}一致率 {np.mean(v == y):.3f}  継続F1 {f1(y, v, 0):.3f}  完了F1 {f1(y, v, 1):.3f}  完了率 {v.mean():.2f}")
        print(f"差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for e in order:
            m = np.array([x == e for x in ends])
            print(f"  {e:40s}{m.sum():5d}{np.mean(vf[m] == y[m]):8.2f}{np.mean(ve[m] == y[m]):8.2f}")

        # 32B 思考なし（2 票目）に (e) を加えたプロンプトを使った場合（silver13 で一時的にダウンロードして判定）
        t32 = OUT + "Qwen3-32B-AWQ-nothink.add_e.tsv"
        if Path(t32).exists() and all(l in read_votes(t32) for l in ln):
            v32e = read_votes(t32); e32 = np.array([v32e[l] for l in ln])
            print("\nQwen3-32B-AWQ 思考なし：元のプロンプト（正本の票）と (e) を加えたプロンプト")
            for nm, p in (("元のプロンプト", f32), ("(e) を加えたプロンプト", e32)):
                print(f"{nm:28s}一致率 {np.mean(p == y):.3f}  継続F1 {f1(y, p, 0):.3f}  完了F1 {f1(y, p, 1):.3f}  完了率 {p.mean():.2f}")
            d, lo, hi = ci(e32, f32)
            print(f"差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"{'文末の形':40s}{'件数':>5s}{'元':>8s}{'(e)':>8s}")
            for e in order:
                m = np.array([x == e for x in ends])
                print(f"{e:40s}{m.sum():5d}{np.mean(f32[m] == y[m]):8.2f}{np.mean(e32[m] == y[m]):8.2f}")
            print("\n多数決：2 票目・3 票目の入れ替え")
            combos = (("正本の多数決", (q, f32, tf)),
                      ("2 票目（32B）だけ (e) 版に", (q, e32, tf)),
                      ("3 票目（8B 思考あり）だけ (e) 版に", (q, f32, te)),
                      ("2・3 票目とも (e) 版に", (q, e32, te)))
            for nm, vs in combos:
                v = (sum(vs) >= 2).astype(int)
                d, lo, hi = ci(v, vf)
                print(f"{nm:34s}一致率 {np.mean(v == y):.3f}  継続F1 {f1(y, v, 0):.3f}  完了F1 {f1(y, v, 1):.3f}  "
                      f"完了率 {v.mean():.2f}  正本との差 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
