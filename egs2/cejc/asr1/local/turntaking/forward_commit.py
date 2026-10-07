#!/usr/bin/env python
"""前向きグリッド（発話開始からの経過時間）での早期確定を評価する。

これまでの評価は「発話末の t 秒手前」を横軸に取っていたが、その索引は実行時には
分からない（6 節の較正はそれで因果性を失っていた）。前向きグリッドは
**発話開始からの経過時間**で切るので、実運用のストリーミングと同じ状態になる。

指標も endpointing の慣行に寄せる：
  応答遅延 = 確定した経過時間 − 発話長
    正 … 発話が終わってから確定した（応答が遅れた）
    負 … 発話の途中で確定した（早切り）
従来の「平均短縮」は −（応答遅延）に相当し、符号が逆なだけで同じもの。

  python local/turntaking/forward_commit.py --stem pool_attn-trunc03_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from early_commit import load_tsv, prf                            # noqa: E402

GRID = ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0", "2.5", "3.0"]
ELAPSED_EDGES = np.array([0, .4, .6, .85, 1.1, 1.4, 1.8, 2.4, 1e9])
MARGIN_EDGES = np.array([0.0, .1, .2, .35, .5, .7, .85, 1.01])
MIN_GROUP = 20


def load_forward(stem, ckpt, dev=False, sil=False):
    """(S,N,3) の確率と (N,) のラベル・発話長。utt id で整列する。

    sil=True で「発話後の無音まで含めて切り出した」ダンプ（_e1.0s 形式）を読む。
    """
    sfx = "_train_dev" if dev else ""
    ss = "s" if sil else ""
    P, base, lab = [], None, None
    for e in GRID:
        u, y, p, ns = load_tsv(f"tt_preds/{stem}{sfx}_e{e}{ss}_{ckpt}.tsv")
        if base is None:
            base, lab = u, y
            order = {x: i for i, x in enumerate(u)}
        else:
            assert len(u) == len(base), f"区間数が不一致: {e}"
            idx = np.array([order[x] for x in u])
            q = np.zeros_like(p); q[idx] = p; p = q
        P.append(p)
    return np.stack(P), lab, base


def utt_lengths(utts, dset):
    seg = {}
    for line in open(f"data/{dset}/segments"):
        u, s, b, e = line.split(); seg[u] = float(e) - float(b)
    return np.array([seg[u] for u in utts])


ENERGY_EDGES = np.array([0, .003, .006, .012, 1e9])   # 直近 0.25 秒の RMS


def load_energy(utts, dset):
    """各格子点における直近 0.25 秒のエネルギー (S,N)。無音かどうかの手がかり。

    実運用では音声区間検出が同じ情報を持つので、これを状態に使うのは因果的に正当。
    発話末の位置は使っていない。
    """
    z = np.load(f"tt_preds/energy_{dset}.npz", allow_pickle=True)
    m = {u: v for u, v in zip(z["utt"], z["val"])}
    return np.array([m[u] for u in utts]).T


def states(P, En=None):
    """(S,N) の状態 id。予測クラス × margin ビン × 経過時間ビン（すべて因果的に既知）。

    En を渡すと直近エネルギーのビンも状態に加える（無音の手がかり）。
    """
    pred = P.argmax(-1)
    srt = np.sort(P, -1)
    mb = np.clip(np.digitize(srt[..., 2] - srt[..., 1], MARGIN_EDGES) - 1,
                 0, len(MARGIN_EDGES) - 2)
    eb = np.clip(np.digitize(np.array([float(e) for e in GRID]), ELAPSED_EDGES) - 1,
                 0, len(ELAPSED_EDGES) - 2)[:, None] * np.ones(P.shape[1], dtype=int)
    nm, ne = len(MARGIN_EDGES) - 1, len(ELAPSED_EDGES) - 1
    s = (pred * nm + mb) * ne + eb
    G = 3 * nm * ne
    if En is not None:
        nb = len(ENERGY_EDGES) - 1
        gb = np.clip(np.digitize(En, ENERGY_EDGES) - 1, 0, nb - 1)
        s = s * nb + gb
        G = G * nb
    return s, G


def eco_tables(Pd, labd, End=None):
    """状態(i,g)にいた系列が時刻 j で示す (真ラベル, 予測) 同時分布。コスト非依存。"""
    sd, G = states(Pd, End)
    S = Pd.shape[0]; predj = Pd.argmax(-1)
    M = np.zeros((S, G, S, 3, 3)); n = np.zeros((S, G))
    for i in range(S):
        for g in np.unique(sd[i]):
            m = sd[i] == g; cnt = int(m.sum()); n[i, g] = cnt
            y = labd[m]
            for j in range(i, S):
                np.add.at(M[i, g, j], (y, predj[j, m]), 1.0)
            M[i, g, i:] /= cnt
    return M, n, G


def eco_stop(Pe, M, n, G, C, alpha, Ene=None):
    S = Pe.shape[0]; se, _ = states(Pe, Ene)
    ev = np.array([float(e) for e in GRID])
    ecost = np.einsum("igjab,ab->igj", M, C) + alpha * ev[None, None, :]
    ok = np.zeros((S, G), dtype=bool)
    for i in range(S):
        for g in range(G):
            if n[i, g] < MIN_GROUP:
                continue
            fut = [ecost[i, g, j] for j in range(i + 1, S)]
            ok[i, g] = ecost[i, g, i] <= (min(fut) if fut else np.inf)
    trig = ok[np.arange(S)[:, None], se]
    return np.where(trig.any(0), trig.argmax(0), S - 1)


def cap_by_segmenter(first, length, max_wait):
    """音声区間検出が「発話末 + max_wait 秒」で発火する、という現実的な制約をかける。

    実運用では無音を無制限に聞き続けることはなく、区間検出がそこで区切って判定を促す。
    max_wait を超える時点で止まる予定だった区間は、その手前の格子点で強制的に確定させる。
    """
    ev = np.array([float(e) for e in GRID])
    limit = length + max_wait
    idx = np.searchsorted(ev, limit, side="right") - 1      # limit 以下の最後の格子点
    idx = np.clip(idx, 0, len(ev) - 1)
    return np.minimum(first, idx)


def report(name, first, Pe, lab, length, rows, max_wait=0.0):
    if max_wait > 0:
        first = cap_by_segmenter(first, length, max_wait)
    ev = np.array([float(e) for e in GRID])
    j = np.arange(len(lab))
    pr = Pe.argmax(-1)[first, j]
    lat = ev[first] - length                      # 応答遅延（秒）。負は早切り
    f1, mac, uar, acc = prf(lab, pr)
    # 応答遅延として意味があるのは「真に完了だった区間」での遅延。
    # あわせて完了の取りこぼし（応答が始まらない）と誤検出（割り込み）を出す。
    m1 = lab == 1
    lat_end = lat[m1].mean()
    miss = (pr[m1] != 1).mean()                   # 完了を取りこぼす＝応答が遅れる
    false = ((pr == 1) & (lab != 1)).sum() / max((lab != 1).sum(), 1)   # 誤って割り込む
    rows.append((name, lat.mean(), lat_end, miss, false, mac, f1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-trunc03_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--max_wait", type=float, default=0.0,
                    help="区間検出が発話末+この秒数で発火するとして、判定を強制する（0 で無制限）")
    ap.add_argument("--sil", action="store_true",
                    help="発話後の無音を含めて切り出したダンプを使う（実運用条件）")
    args = ap.parse_args()

    Pe, labe, utte = load_forward(args.stem, args.ckpt, sil=args.sil)
    Pd, labd, uttd = load_forward(args.stem, args.ckpt, dev=True, sil=args.sil)
    Le = utt_lengths(utte, "eval")
    M, n, G = eco_tables(Pd, labd)
    C = 1.0 - np.eye(3)

    rows = []
    S = Pe.shape[0]
    mx = Pe.max(-1)
    for tau in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]:
        ok = mx >= tau
        first = np.where(ok.any(0), ok.argmax(0), S - 1)
        report(f"単一τ={tau:.2f}" if tau <= 1 else "常に最後まで待つ",
               first, Pe, labe, Le, rows, args.max_wait)
    # 対照：単一τ＋経過時間ゲート。経過時間を使う最も安価な方法で、
    # Economy の優位が「経過時間を使ったから」だけなのかを切り分ける。
    ev = np.array([float(e) for e in GRID])
    for gate in [0.5, 0.75, 1.0, 1.25]:
        for tau in [0.80, 0.90, 0.95]:
            ok = (mx >= tau) & (ev[:, None] >= gate)
            first = np.where(ok.any(0), ok.argmax(0), S - 1)
            report(f"τ={tau:.2f}+ゲート{gate}", first, Pe, labe, Le, rows, args.max_wait)
    for a in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        report(f"Economy α={a:.2f}", eco_stop(Pe, M, n, G, C, a), Pe, labe, Le, rows, args.max_wait)
    # 非対称コスト：真=完了 を取り逃す誤り（応答が始まらない）を w 倍に重くする。
    # macro-F1 ではなく応答遅延を目的にするなら、こちらが正しいコストの形。
    for w in [2.0, 4.0]:
        Cw = 1.0 - np.eye(3); Cw[1, :] *= w; Cw[1, 1] = 0.0
        report(f"取りこぼし{w:g}倍 α=0", eco_stop(Pe, M, n, G, Cw, 0.0), Pe, labe, Le, rows, args.max_wait)
    # 逆向き：誤って「完了」と判定して割り込む誤りを重くする。
    # 対話システムでは割り込みのほうが体感コストが高いので、こちらが実用的な向き。
    for w in [2.0, 4.0, 8.0]:
        Cw = 1.0 - np.eye(3); Cw[:, 1] *= w; Cw[1, 1] = 0.0
        report(f"誤割り込み{w:g}倍 α=0", eco_stop(Pe, M, n, G, Cw, 0.0), Pe, labe, Le, rows, args.max_wait)

    print(f"モデル {args.stem} / eval {len(labe)} 区間 / 発話長 平均 {Le.mean():.2f} 秒"
          + ("  【発話後の無音を含む】" if args.sil else "  【無音なし】"))
    print("応答遅延 = 確定時刻 − 発話末。正なら待ちすぎ、負なら早切り。\n")
    print(f"{'停止ルール':>18}{'遅延平均':>10}{'完了の遅延':>11}{'取りこぼし':>11}"
          f"{'誤割り込み':>11}{'macroF1':>9}{'完了F1':>8}")
    for nm, m_, le, ms, fa, mac, f1 in rows:
        print(f"{nm:>18}{m_:10.3f}{le:11.3f}{ms:11.3f}{fa:11.3f}{mac:9.4f}{f1[1]:8.3f}")


if __name__ == "__main__":
    main()
