#!/usr/bin/env python
"""継続/完了の 0.65 の壁が「モデルの限界」か「ラベルの限界」かを切り分ける分析。

evaluate_multitask.py --dump_preds が出した TSV（utt/true/pred/確率）に
data/eval/text と segments を突き合わせ、次の4点を出す。

  A. 同一表層テキスト群のラベル不一致 → テキストのみで到達可能な上限（オラクル）
  B. 文末形式ごとの真ラベル分布・多数派ベースライン・モデル精度
  C. 確信度と正誤の関係（selective accuracy: 自信のある区間だけ残したら伸びるか）
  D. 発話長との関係

いずれも人手アノテーション不要。学習も不要。
"""
import argparse
from collections import Counter, defaultdict

NAMES = {0: "継続", 1: "完了", 2: "相槌"}


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            r = dict(zip(header, p))
            r["true"], r["pred"] = int(r["true"]), int(r["pred"])
            for k in ("p_cont", "p_end", "p_bc"):
                r[k] = float(r[k])
            r["n_samp"] = int(r["n_samp"])
            rows.append(r)
    return rows


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1]
    return m


def sec_A(rows):
    """同一表層テキスト群のラベル不一致 = テキストのみのオラクル上限。"""
    print("=" * 70)
    print("A. 同一表層テキストのラベル不一致（テキストのみのオラクル上限）")
    print("=" * 70)
    groups = defaultdict(list)
    for r in rows:
        groups[r["txt"]].append(r)
    multi = {t: g for t, g in groups.items() if len(g) >= 2}
    n_multi = sum(len(g) for g in multi.values())
    print(f"継続/完了 区間 {len(rows)} 件 / 異なり表層 {len(groups)} 種")
    print(f"  うち 2件以上出現する表層に属する区間: {n_multi} 件 "
          f"({n_multi/len(rows):.1%})")

    # 多数派ラベルを当てられるのが「テキストだけを見た完璧なモデル」の上限
    oracle = sum(max(Counter(x["true"] for x in g).values()) for g in multi.values())
    model = sum(sum(x["true"] == x["pred"] for x in g) for g in multi.values())
    print(f"  この部分集合での オラクル上限 acc = {oracle/n_multi:.3f}")
    print(f"  この部分集合での モデル      acc = {model/n_multi:.3f}")
    print(f"  → 同一テキストなのにラベルが割れる割合 = {1-oracle/n_multi:.1%}")

    print("\n  ラベルが割れている高頻度の表層（上位20）:")
    split = []
    for t, g in multi.items():
        c = Counter(x["true"] for x in g)
        if len(c) >= 2:
            split.append((len(g), t, c))
    split.sort(reverse=True, key=lambda x: x[0])
    for n, t, c in split[:20]:
        acc = sum(x["true"] == x["pred"] for x in groups[t]) / n
        print(f"    {n:5d}件  継続{c.get(0,0):5d}/完了{c.get(1,0):5d}"
              f"  model_acc={acc:.2f}  「{t[:40]}」")
    n_split = sum(n for n, _, _ in split)
    print(f"  ラベルが割れる表層に属する区間: {n_split} 件 "
          f"({n_split/len(rows):.1%} of 継続/完了)")


def sec_B(rows, topk=25):
    """文末形式ごとの分布とモデル精度。"""
    print()
    print("=" * 70)
    print("B. 文末形式（末尾2文字）ごとの真ラベル分布とモデル精度")
    print("=" * 70)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["txt"][-2:]][:0] = [r]
    items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:topk]
    print(f"{'末尾':>6} {'件数':>6} {'継続率':>7} {'多数派':>7} {'model':>7} {'差':>6}")
    tot_or = tot_md = tot_n = 0
    for suf, g in items:
        n = len(g)
        c = Counter(x["true"] for x in g)
        cont = c.get(0, 0) / n
        orc = max(c.values()) / n
        md = sum(x["true"] == x["pred"] for x in g) / n
        tot_or += max(c.values()); tot_md += sum(x["true"] == x["pred"] for x in g)
        tot_n += n
        print(f"{suf:>6} {n:6d} {cont:7.2f} {orc:7.3f} {md:7.3f} {md-orc:+6.3f}")
    print(f"{'計':>6} {tot_n:6d} {'':>7} {tot_or/tot_n:7.3f} {tot_md/tot_n:7.3f}"
          f" {(tot_md-tot_or)/tot_n:+6.3f}")
    print("  多数派 = 末尾2文字だけ見て多数派ラベルを答える上限。"
          "model がこれを上回れば、音響・文脈から上積みできている。")


def sec_C(rows):
    """確信度と正誤（selective accuracy）。"""
    print()
    print("=" * 70)
    print("C. 確信度と正誤（継続/完了の2値マージン |p_cont - p_end|）")
    print("=" * 70)
    for r in rows:
        r["margin"] = abs(r["p_cont"] - r["p_end"])
        r["ok"] = int(r["true"] == r["pred"])
    ok = [r["margin"] for r in rows if r["ok"]]
    ng = [r["margin"] for r in rows if not r["ok"]]
    print(f"  正答 {len(ok)} 件 平均マージン = {sum(ok)/len(ok):.3f}")
    print(f"  誤答 {len(ng)} 件 平均マージン = {sum(ng)/len(ng):.3f}")
    srt = sorted(rows, key=lambda r: -r["margin"])
    print(f"\n  {'確信上位':>8} {'件数':>6} {'acc':>7}")
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        k = int(len(srt) * frac)
        print(f"  {frac:8.0%} {k:6d} {sum(r['ok'] for r in srt[:k])/k:7.3f}")
    print("  → 高確信部分でも acc が頭打ちなら「自信を持って間違えている」"
          "＝ラベル/タスク定義側の疑い。")


def sec_D(rows):
    """発話長との関係。"""
    print()
    print("=" * 70)
    print("D. 発話長ごとの精度（16kHz サンプル数 → 秒）")
    print("=" * 70)
    edges = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
    print(f"{'長さ(s)':>12} {'件数':>6} {'継続率':>7} {'多数派':>7} {'model':>7}")
    for lo, hi in edges:
        g = [r for r in rows if lo <= r["n_samp"] / 16000 < hi]
        if not g:
            continue
        c = Counter(x["true"] for x in g)
        lbl = f"{lo}-{hi}" if hi < 1e9 else f"{lo}+"
        print(f"{lbl:>12} {len(g):6d} {c.get(0,0)/len(g):7.2f}"
              f" {max(c.values())/len(g):7.3f}"
              f" {sum(x['true']==x['pred'] for x in g)/len(g):7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="--dump_preds の TSV")
    ap.add_argument("--text", default="data/eval/text")
    ap.add_argument("--out_errors", default="", help="誤答例を書き出す TSV")
    args = ap.parse_args()

    rows = read_tsv(args.preds)
    text = {k: v.replace(" ", "") for k, v in read_map(args.text).items()}
    for r in rows:
        r["txt"] = text.get(r["utt"], "")
    n_notxt = sum(1 for r in rows if not r["txt"])
    if n_notxt:
        print(f"[warn] text が引けない区間 {n_notxt} 件")

    print(f"全 {len(rows)} 区間 / 3クラス acc = "
          f"{sum(r['true']==r['pred'] for r in rows)/len(rows):.3f}")
    noyes = [r for r in rows if r["true"] in (0, 1) and r["txt"]]
    print(f"継続/完了 のみ {len(noyes)} 区間 / acc = "
          f"{sum(r['true']==r['pred'] for r in noyes)/len(noyes):.3f}\n")

    sec_A(noyes)
    sec_B(noyes)
    sec_C(noyes)
    sec_D(noyes)

    if args.out_errors:
        errs = sorted((r for r in noyes if r["true"] != r["pred"]),
                      key=lambda r: -abs(r["p_cont"] - r["p_end"]))
        with open(args.out_errors, "w", encoding="utf-8") as f:
            f.write("utt\ttrue\tpred\tmargin\ttext\n")
            for r in errs:
                f.write(f"{r['utt']}\t{NAMES[r['true']]}\t{NAMES[r['pred']]}"
                        f"\t{abs(r['p_cont']-r['p_end']):.3f}\t{r['txt']}\n")
        print(f"\n誤答 {len(errs)} 件を確信度降順で {args.out_errors} に出力")


if __name__ == "__main__":
    main()
