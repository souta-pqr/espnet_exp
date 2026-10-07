#!/usr/bin/env python
"""テキストのみのベースラインを train_nodup で学習し eval で評価する。

狙い: 音声モデル（Stage2）の 継続/完了 0.656 が「テキストだけで届く水準」なのか
「テキスト＋音響で上積みできている水準」なのかを、学習/評価を分けた公平な条件で測る。

  (1) 末尾N文字の多数派ルール（train で推定 → eval に適用）
  (2) 文字 n-gram ロジスティック回帰（末尾重み付き）

正解ラベルは data/<set>/tag（0=継続 1=完了 2=相槌）。継続/完了 の2値のみを対象。
"""
import argparse
from collections import Counter, defaultdict


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1]
    return m


def load(d):
    text = {k: v.replace(" ", "") for k, v in read_map(f"{d}/text").items()}
    tag = {k: int(v) for k, v in read_map(f"{d}/tag").items()}
    X, y = [], []
    for u, t in text.items():
        if u in tag and tag[u] in (0, 1) and t:
            X.append(t)
            y.append(tag[u])
    return X, y


def macro_f1(conf):
    """conf[真][予測] の 2x2 から macro-F1。"""
    f1 = []
    for c in (0, 1):
        tp = conf[c][c]
        prec = tp / max(conf[0][c] + conf[1][c], 1)
        rec = tp / max(sum(conf[c]), 1)
        f1.append(2 * prec * rec / max(prec + rec, 1e-8))
    return sum(f1) / 2


def suffix_rule(Xtr, ytr, Xte, yte, n):
    """末尾n文字 → train の多数派ラベル。未知の末尾は全体多数派。"""
    cnt = defaultdict(Counter)
    for x, t in zip(Xtr, ytr):
        cnt[x[-n:]][t] += 1
    back = Counter(ytr).most_common(1)[0][0]
    rule = {k: c.most_common(1)[0][0] for k, c in cnt.items() if sum(c.values()) >= 3}
    ok = cov = 0
    for x, t in zip(Xte, yte):
        p = rule.get(x[-n:])
        cov += p is not None
        ok += (p if p is not None else back) == t
    return ok / len(yte), cov / len(yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_nodup")
    ap.add_argument("--eval", default="data/eval")
    args = ap.parse_args()

    Xtr, ytr = load(args.train)
    Xte, yte = load(args.eval)
    print(f"train 継続/完了 {len(ytr)} 件  (継続率 {ytr.count(0)/len(ytr):.3f})")
    print(f"eval  継続/完了 {len(yte)} 件  (継続率 {yte.count(0)/len(yte):.3f})")
    print(f"多数派クラスのみ答える acc = "
          f"{max(yte.count(0), yte.count(1))/len(yte):.3f}\n")

    print("(1) 末尾N文字の多数派ルール（train 推定 → eval 適用、出現3件以上の末尾のみ）")
    print(f"{'N':>3} {'eval acc':>9} {'被覆率':>8}")
    for n in (1, 2, 3, 4, 6):
        acc, cov = suffix_rule(Xtr, ytr, Xte, yte, n)
        print(f"{n:3d} {acc:9.3f} {cov:8.1%}")

    print("\n(2) 文字 n-gram ロジスティック回帰")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    # 発話全体の文字 n-gram ＋ 末尾6文字の文字 n-gram（文末形式を強調）
    tail = lambda xs: [x[-6:] for x in xs]  # noqa: E731
    from sklearn.preprocessing import FunctionTransformer
    feats = make_union(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3),
        make_pipeline(
            FunctionTransformer(tail, validate=False),
            TfidfVectorizer(analyzer="char", ngram_range=(1, 6), min_df=3),
        ),
    )
    # 音声モデルは継続/完了の recall がほぼ均衡しているので、比較のため
    # class_weight あり/なしの両方を出す（動作点を揃えないと acc は比べられない）。
    for cw in (None, "balanced"):
        clf = make_pipeline(
            feats, LogisticRegression(max_iter=2000, C=2.0, class_weight=cw))
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        conf = [[0, 0], [0, 0]]
        for p, t in zip(pred, yte):
            conf[t][p] += 1
        print(f"  class_weight={cw}")
        print(f"    eval 継続/完了 acc = "
              f"{sum(p == t for p, t in zip(pred, yte))/len(yte):.3f}"
              f"   2値 macro-F1 = {macro_f1(conf):.3f}")
        print(f"    混同行列 [真→予測]: 継続{conf[0]}  完了{conf[1]}")
        print(f"    recall 継続={conf[0][0]/max(sum(conf[0]),1):.3f}"
              f" 完了={conf[1][1]/max(sum(conf[1]),1):.3f}")


if __name__ == "__main__":
    main()
