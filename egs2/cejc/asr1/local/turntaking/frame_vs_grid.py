#!/usr/bin/env python
"""フレーム単位ダンプと、波形を切る前向きグリッドを突き合わせる。

同じ経過時刻での判定が一致するかを見る。両者の違いは符号化の仕方だけである。

  フレーム単位 … 長い音声を 1 回符号化し、位置 p を読む。
                 実際のストリーミングが時刻 p に持っている表現と一致する
                 （後続 6 フレームで表現が確定することを実測済み）。
  グリッド     … 時刻ごとに波形を切って符号化し直し、最終フレームを読む。
                 切断端が**境界として処理される**ので、
                 実運用では生じない「ここで音声が途切れた」信号が入る。

差が大きければ、既存の早期確定の数値は境界処理の影響を受けていることになる。

  python local/turntaking/frame_vs_grid.py --stem pool_attn-trunc-new_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anticipate_commit import cap_index, utt_lengths                 # noqa: E402
from early_commit import prf                                         # noqa: E402

GRID = ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0", "2.5", "3.0"]


def load_frame(path):
    """utt → (経過秒の配列, 確率 (M,3))。"""
    import collections
    el = collections.defaultdict(list)
    pr = collections.defaultdict(list)
    lab = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            u = p[0]
            lab[u] = int(p[1])
            el[u].append(float(p[3]))
            pr[u].append([float(p[4]), float(p[5]), float(p[6])])
    return ({u: np.array(v) for u, v in el.items()},
            {u: np.array(v) for u, v in pr.items()}, lab)


def load_grid(stem, ckpt, dset):
    """格子ごとの確率を utt 順にそろえて (S, N, 3) で返す。"""
    from anticipate_commit import load_tsv_h
    sfx = "" if dset == "eval" else f"_{dset}"
    P, base, lab = [], None, None
    for e in GRID:
        u, y, p, h, _ = load_tsv_h(f"tt_preds/{stem}{sfx}_e{e}s_{ckpt}.tsv")
        if base is None:
            base, lab = u, y
            order = {x: i for i, x in enumerate(u)}
        else:
            idx = np.array([order[x] for x in u])
            q = np.zeros_like(p); q[idx] = p; p = q
        P.append(p)
    return np.stack(P), lab, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-trunc-new_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--dset", default="eval")
    ap.add_argument("--max_wait", type=float, default=0.25)
    args = ap.parse_args()

    fpath = (f"tt_preds/frame_attn{args.stem.split('attn')[1].split('_same')[0]}"
             f"_same_n5_{args.dset}_{args.ckpt}.tsv")
    if not Path(fpath).exists():
        sys.exit(f"フレーム単位ダンプが無い: {fpath}")
    EL, PR, labf = load_frame(fpath)
    Pg, labg, utts = load_grid(args.stem, args.ckpt, args.dset)
    ev = np.array([float(e) for e in GRID])

    # 共通の区間だけを使う
    common = [u for u in utts if u in EL]
    gi = {u: i for i, u in enumerate(utts)}
    print(f"区間 {len(common)} / グリッド {len(utts)}　フレーム単位 {len(EL)}")

    # 1) 同じ経過時刻での確率の一致
    print(f"\n{'経過秒':>8}{'p_end の平均差':>14}{'|Δp| 平均':>11}"
          f"{'予測一致率':>11}{'相関':>8}")
    Pf = np.zeros((len(ev), len(common), 3))
    for si, t in enumerate(ev):
        rows = []
        for u in common:
            e = EL[u]
            j = int(np.argmin(np.abs(e - t)))      # 最も近いフレーム
            rows.append(PR[u][j])
        Pf[si] = np.array(rows)
        g = Pg[si, [gi[u] for u in common]]
        d = Pf[si] - g
        agree = (Pf[si].argmax(-1) == g.argmax(-1)).mean()
        r = np.corrcoef(Pf[si][:, 1], g[:, 1])[0, 1]
        print(f"{t:8.2f}{d[:,1].mean():14.3f}{np.abs(d).mean():11.3f}"
              f"{agree:11.3f}{r:8.3f}")

    # 2) 停止ルールを両方に載せて比べる
    lab = np.array([labf[u] for u in common])
    L = utt_lengths(common, args.dset)
    cap = cap_index(ev, L, args.max_wait)
    print(f"\n同じ停止ルールを両方に載せた場合（無音込み・区間検出 {args.max_wait} 秒）")
    print(f"{'符号化の仕方':>16}{'規則':>14}{'完了の遅延':>11}{'応答ミス':>9}"
          f"{'誤割り込み':>10}{'macroF1':>9}")
    for nm, P in (("フレーム単位", Pf),
                  ("波形を切る", Pg[:, [gi[u] for u in common]])):
        for rule, tau in (("区間検出まで待つ", None), ("完了τ=0.70", 0.70)):
            S = len(ev)
            if tau is None:
                first = np.full(len(lab), S - 1)
                pr = P.argmax(-1)[np.minimum(first, cap), np.arange(len(lab))]
            else:
                ok = P[..., 1] >= tau
                fired = ok.any(0)
                first = np.where(fired, ok.argmax(0), S - 1)
                idx = np.minimum(first, cap)
                pr = P.argmax(-1)[idx, np.arange(len(lab))]
                use = fired & (first <= cap)
                pr[use] = 1
            lat = ev[np.minimum(first, cap)] - L
            f1, mac, uar, acc = prf(lab, pr)
            m1 = lab == 1
            print(f"{nm:>16}{rule:>14}{lat[m1].mean():11.3f}"
                  f"{(pr[m1]!=1).mean():9.3f}"
                  f"{((pr==1)&~m1).sum()/max((~m1).sum(),1):10.3f}{mac:9.4f}")


if __name__ == "__main__":
    main()
