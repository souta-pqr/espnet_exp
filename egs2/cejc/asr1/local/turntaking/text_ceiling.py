#!/usr/bin/env python
"""ラベルの予測可能性の上限を測る（正解書き起こしを使う強いテキストモデル）。

正解ラベルは言語モデルが**書き起こしとタイミングだけ**から付けている。
つまりラベルはテキストの関数なので、テキストから当てられる精度が
**音声モデルが到達しうる上限**になる。

資料はこの比較を文字 n-gram の TF-IDF ＋ ロジスティック回帰で行い AUC 0.718 を得た
（音声モデルは 0.715）。しかしそれは弱いモデルの値で、
**ラベルの本当の予測可能性は測れていない**。ここでは日本語 BERT を微調整して測り直す。

  上限が高い  → ラベルは予測可能。音声モデル側が律速しており、改良を続ける根拠になる
  上限が 0.72 前後 → ラベル自体が限界。モデルをいじる作業は打ち切るべき

**因果性**：与える情報は判定時点で得られるものだけに限る
（現発話の書き起こし・直前の文脈・現発話長・直前の間）。
「次に誰が話し始めるか」などの未来情報は与えない。
これを入れると、タイミング規則で決まるラベルを自明に当ててしまい上限が過大になる。

  python local/turntaking/text_ceiling.py --epochs 2
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

NAMES = ["継続", "完了", "相槌"]


def read_map(path, sep=None):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            if len(p) == 2:
                m[p[0]] = p[1]
    return m


def session_of(reco):
    return re.sub(r"_IC\d+$", "", reco)


def build(dset, n_ctx=3, text_scp=None):
    """各区間について (入力テキスト, ラベル) を作る。未来情報は使わない。

    text_scp を渡すと書き起こしをそれで置き換える（ASR 仮説での評価用）。
    現発話も文脈も同じ源に揃える —— 実運用では正解書き起こしは存在しないため。
    """
    d = Path("data") / dset
    text = read_map(Path(text_scp) if text_scp else d / "text")
    tag = {k: int(v) for k, v in read_map(Path("dump/raw") / dset / "tag").items()}
    seg = {}
    with open(d / "segments", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                seg[p[0]] = (p[1], float(p[2]), float(p[3]))

    # セッション単位で時系列に並べる（話者をまたぐ文脈も与える＝上限の測定なので広めに）
    by_sess = {}
    for u in text:
        if u not in seg:
            continue
        by_sess.setdefault(session_of(seg[u][0]), []).append(u)
    for k in by_sess:
        by_sess[k].sort(key=lambda x: seg[x][1])

    for u in list(by_sess and seg):
        text.setdefault(u, "")          # ASR 仮説に無い区間は空文字（件数をそろえる）
    rows = []
    for k, utts in by_sess.items():
        for i, u in enumerate(utts):
            if u not in tag:
                continue
            reco, b, e = seg[u]
            ctx = []
            for j in range(max(0, i - n_ctx), i):
                v = utts[j]
                vr, vb, ve = seg[v]
                if ve > b:                      # 未来にはみ出す発話は入れない
                    continue
                who = "自分" if vr == reco else "相手"
                ctx.append(f"{who}:{text[v].replace(' ', '')}")
            # 直前の間（負なら重複）。判定時点で分かる量だけ。
            gap = b - max([seg[v][2] for v in utts[max(0, i - n_ctx):i]] or [b])
            cur = text[u].replace(" ", "")
            head = f"[長さ{e-b:.1f}秒][間{gap:.1f}秒]"
            rows.append({
                "utt": u,
                "text": " / ".join(ctx) + " ‖ " + head + "自分:" + cur,
                "label": tag[u],
            })
    return rows


def auc_cont_end(prob, lab):
    """継続 vs 完了 の AUC（相槌を除く）。スコアは p(完了) − p(継続)。"""
    m = (lab == 0) | (lab == 1)
    s = prob[m, 1] - prob[m, 0]
    y = (lab[m] == 1).astype(int)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 同点は平均順位
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(u))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(prob, lab, name):
    pred = prob.argmax(1)
    conf = np.zeros((3, 3), int)
    for y, p in zip(lab, pred):
        conf[y, p] += 1
    rec = [conf[c, c] / max(conf[c].sum(), 1) for c in range(3)]
    pre = [conf[c, c] / max(conf[:, c].sum(), 1) for c in range(3)]
    f1 = [2 * pre[c] * rec[c] / max(pre[c] + rec[c], 1e-9) for c in range(3)]
    a = auc_cont_end(prob, lab)
    print(f"\n===== {name} =====")
    print(f"macro-F1 = {np.mean(f1):.4f}   継続vs完了 AUC = {a:.4f}   "
          f"acc = {np.trace(conf)/conf.sum():.4f}")
    for c in range(3):
        print(f"  {NAMES[c]:>4}  P {pre[c]:.3f}  R {rec[c]:.3f}  F1 {f1[c]:.3f}  "
              f"（{conf[c].sum()} 件）")
    return np.mean(f1), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tohoku-nlp/bert-base-japanese-v3")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--out", default="tt_analysis/text_ceiling.json")
    ap.add_argument("--save", default="", help="学習した重みの保存先")
    ap.add_argument("--train_text", default="",
                    help="学習も この書き起こしで行う（ASR 仮説で学習＝条件をそろえる）")
    ap.add_argument("--train_text2", default="",
                    help="2 本目の学習用書き起こし。渡すと 1 本目に連結して学習データを"
                         "倍にする（誤りの質が違う仮説を混ぜる＝ノイズ増強）")
    ap.add_argument("--dev_text", default="", help="dev 側の書き起こし")
    ap.add_argument("--eval_text", default="",
                    help="eval をこの書き起こしでも評価する（ASR 仮説を渡す）")
    ap.add_argument("--seed", type=int, default=None,
                    help="乱数シード。既定（指定なし）は従来どおり固定しない")
    ap.add_argument("--train_set", default="train_nodup")
    ap.add_argument("--dev_set", default="train_dev")
    ap.add_argument("--eval_set", default="eval")
    ap.add_argument("--class_weight", default="1.92 1.0 1.63",
                    help="交差エントロピーのクラス重み [継続 完了 相槌]。既定は旧データの割合"
                         "（完了÷各クラス）。新しい分割 train_g では 2.88 1.0 1.89")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.seed is not None:
        import random
        from transformers import set_seed
        random.seed(args.seed); np.random.seed(args.seed); set_seed(args.seed)

    tr = build(args.train_set, text_scp=args.train_text or None)
    if args.train_text2:
        tr2 = build(args.train_set, text_scp=args.train_text2)
        print(f"学習データを連結: {len(tr)} + {len(tr2)} = {len(tr) + len(tr2)}")
        tr = tr + tr2
    dv = build(args.dev_set, text_scp=args.dev_text or None)
    ev = build(args.eval_set)
    if args.max_train:
        tr = tr[: args.max_train]
    print(f"train {len(tr)} / dev {len(dv)} / eval {len(ev)}")
    print("入力例:", tr[5]["text"][:160])

    tk = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3).cuda()

    class DS(Dataset):
        def __init__(self, rows): self.r = rows
        def __len__(self): return len(self.r)
        def __getitem__(self, i): return self.r[i]

    def collate(batch):
        enc = tk([b["text"] for b in batch], truncation=True,
                 max_length=args.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc

    dl = DataLoader(DS(tr), batch_size=args.bs, shuffle=True, collate_fn=collate,
                    num_workers=4, drop_last=True)
    steps = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    # 音声モデルと同じクラス重み
    w = torch.tensor([float(x) for x in args.class_weight.split()]).cuda()
    print(f"クラス重み [継続 完了 相槌] = {w.tolist()}")
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    done = 0
    while done < steps:
        for batch in dl:
            if done >= steps:
                break
            batch = {k: v.cuda() for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(out.float(), y, weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
            done += 1
            if done % 500 == 0:
                print(f"  step {done}/{steps}  loss {loss.item():.4f}", flush=True)

    @torch.no_grad()
    def infer(rows):
        model.eval()
        P = []
        dl2 = DataLoader(DS(rows), batch_size=256, collate_fn=collate, num_workers=4)
        for batch in dl2:
            batch = {k: v.cuda() for k, v in batch.items()}
            batch.pop("labels")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                P.append(model(**batch).logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(P), np.array([r["label"] for r in rows])

    sets = [("dev（正解書き起こし）", dv), ("eval（正解書き起こし）", ev)]
    if args.eval_text:
        sets.append(("eval（ASR 認識結果）", build(args.eval_set, text_scp=args.eval_text)))
    res = {}
    for nm, rows in sets:
        prob, lab = infer(rows)
        f1, a = report(prob, lab, f"{args.model} / {nm}")
        res[nm] = {"macro_f1": float(f1), "auc_cont_end": float(a)}
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save); tk.save_pretrained(args.save)
        print(f"重みを保存: {args.save}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
