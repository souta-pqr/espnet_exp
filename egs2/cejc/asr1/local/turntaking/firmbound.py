#!/usr/bin/env python
"""FIRMBOUND 型の停止規則（有限ホライズンの後ろ向き帰納）を実装して比べる。

資料 9 節の注記と 14 節の残件。FIRMBOUND（Ebihara et al., ICLR 2025,
"Learning the Optimal Stopping for Early Classification within Finite Horizons
via Sequential Probability Ratio Test"）は、有限ホライズンのベイズリスクを
後ろ向き帰納で最小化し、**時変の最適境界**を導く。掃引した固定閾値とは別物。

資料はこれを「発話末＝ホライズンが実行時に未知」「ダンプが発話末から 0.6 秒の窓しか
覆っていない」という 2 点で保留していた。**前向きグリッド（発話開始からの経過時間）の
ダンプが揃ったことで、どちらも解消している** —— ホライズンは経過時間で定義でき、
実行時に既知だからである。

  価値関数（コスト最小化）
    V_S(s)   = 今止めたときの期待誤分類コスト（最終格子点では止めるしかない）
    V_i(s)   = min( 今止めるコスト,  E[V_{i+1} | s] + α·Δt )
    停止規則 = 今止めるコスト ≤ E[V_{i+1} | s] となる最初の i

肝は E[V_{i+1} | s] の推定。FIRMBOUND はこれを **凸関数学習**で推定する。
コスト最小化では価値関数は信念（事後確率）について**凹**なので、
ここでは min-affine（K 本の線形関数の最小値）で当てはめる。
K を振れば当てはめの自由度を制御でき、資料 10 節の
「自由度には最適点があり、かなり低い側にある」という知見を検証できる。

推定は train_dev のみで行い、eval には一切触れない。

  python local/turntaking/firmbound.py --stem pool_attn-trunc-new_same_n5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anticipate_commit import cap_index, load_forward, utt_lengths   # noqa: E402
from early_commit import prf                                        # noqa: E402

GRID9 = "0.25 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0"
GRID17 = ("0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25 1.375 1.5 "
          "1.75 2.0 2.25 2.5 2.75 3.0")
EPS = 1e-8


def belief_features(P):
    """事後確率 (N,3) → 回帰の説明変数 (N,3)。

    対数尤度比 2 本（FIRMBOUND の十分統計量に対応）＋ 定数項のための 1。
    """
    lp = np.log(np.clip(P, EPS, 1.0))
    return np.stack([lp[:, 0] - lp[:, 1], lp[:, 2] - lp[:, 1], np.ones(len(P))], 1)


def fit_min_affine(X, y, K, iters=25, seed=0):
    """y ≈ min_j (a_j·x) を当てはめる（凹関数の当てはめ）。

    交互法：各点を最小を与える枝に割り当て → 枝ごとに最小二乗。
    K=1 なら通常の線形回帰と一致する。
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 4 * K or K <= 1:
        A = np.linalg.lstsq(X, y, rcond=None)[0][None, :]
        return A
    # 初期化：y の分位で K 群に分ける
    order = np.argsort(y)
    A = np.zeros((K, X.shape[1]))
    for j, idx in enumerate(np.array_split(order, K)):
        A[j] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
    for _ in range(iters):
        pred = X @ A.T                       # (n, K)
        who = pred.argmin(1)
        newA = A.copy()
        moved = False
        for j in range(K):
            m = who == j
            if m.sum() >= X.shape[1] + 2:
                newA[j] = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
                moved = True
        if not moved:
            break
        if np.allclose(newA, A, atol=1e-9):
            A = newA
            break
        A = newA
    return A


def calibrate(Pd, labd, Pe):
    """時刻ごとに dev で多項ロジスティック較正し、dev/eval の事後確率を作り直す。

    モデルの softmax をそのまま信念に使うと、自信過剰がそのまま
    「今止めるコスト」の過小評価になり、停止が早くなりすぎる。
    FIRMBOUND が密度比推定で事後確率を作り直すのに対応する工程。
    当てはめは dev のみで、eval には触れない。
    """
    from sklearn.linear_model import LogisticRegression
    S = Pd.shape[0]
    Qd = np.empty_like(Pd)
    Qe = np.empty_like(Pe)
    for i in range(S):
        Xd = belief_features(Pd[i])[:, :2]
        Xe = belief_features(Pe[i])[:, :2]
        lr = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        lr.fit(Xd, labd)
        pd_ = np.zeros((len(Xd), 3)); pe_ = np.zeros((len(Xe), 3))
        pd_[:, lr.classes_] = lr.predict_proba(Xd)
        pe_[:, lr.classes_] = lr.predict_proba(Xe)
        Qd[i], Qe[i] = pd_, pe_
    return Qd, Qe


def stop_cost(P, C):
    """今止めたときの期待誤分類コスト (S,N) と、そのときの予測クラス (S,N)。"""
    # cost[k] = Σ_y C[y,k] P[y]
    c = np.einsum("sny,yk->snk", P, C)
    return c.min(-1), c.argmin(-1)


def backward_induction(Pd, C, alpha, dt, K, seed=0):
    """train_dev で後ろ向き帰納し、各時刻の継続価値の当てはめ A_i を返す。"""
    S, N, _ = Pd.shape
    sc, _ = stop_cost(Pd, C)
    V = sc[S - 1].copy()                      # 最終格子点では止めるしかない
    As = [None] * S
    for i in range(S - 2, -1, -1):
        X = belief_features(Pd[i])
        A = fit_min_affine(X, V, K, seed=seed)
        cont = (X @ A.T).min(1) + alpha * dt[i]
        As[i] = A
        V = np.minimum(sc[i], cont)
    return As


def firmbound_stop(Be, As, C, alpha, dt):
    """eval で前向きに走らせ、停止した格子点 (N,) を返す。

    信念 Be は停止の判断にだけ使う。**予測クラスは他の規則と同じ生の argmax** を
    使うので、表に出る差は「いつ止めたか」だけに由来する。
    """
    S, N, _ = Be.shape
    sc, _ = stop_cost(Be, C)
    first = np.full(N, S - 1)
    done = np.zeros(N, dtype=bool)
    for i in range(S - 1):
        X = belief_features(Be[i])
        cont = (X @ As[i].T).min(1) + alpha * dt[i]
        take = (~done) & (sc[i] <= cont)
        first[take] = i
        done |= take
        if done.all():
            break
    return first


def evaluate(name, first, pred, lab, ev, cap, L, rows):
    idx = np.minimum(first, cap)
    pr = pred[idx, np.arange(len(lab))]
    lat = ev[idx] - L
    f1, mac, uar, acc = prf(lab, pr)
    m1 = lab == 1
    rows.append((name, lat.mean(), lat[m1].mean(), (pr[m1] != 1).mean(),
                 ((pr == 1) & ~m1).sum() / max((~m1).sum(), 1), mac, f1[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pool_attn-trunc-new_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    ap.add_argument("--max_wait", type=float, default=0.25)
    ap.add_argument("--pieces", default="1 2 4 8 16",
                    help="min-affine の枝数 K（当てはめの自由度）")
    ap.add_argument("--alphas", default="0.0 0.05 0.1 0.2 0.4",
                    help="遅延コスト α（秒あたり）")
    ap.add_argument("--calib", choices=["none", "multinomial"], default="multinomial",
                    help="信念に使う事後確率を dev で較正するか")
    args = ap.parse_args()

    grid = (GRID17 if args.stem.endswith("-tail_same_n5") else GRID9).split()
    grid = [e for e in grid
            if Path(f"tt_preds/{args.stem}_e{e}s_{args.ckpt}.tsv").exists()]
    ev = np.array([float(e) for e in grid])
    dt = np.diff(ev, append=ev[-1])

    Pe, He, labe, utte, _ = load_forward(args.stem, args.ckpt, grid, sil=True)
    Pd, Hd, labd, uttd, _ = load_forward(args.stem, args.ckpt, grid, dev=True, sil=True)
    L = utt_lengths(utte, "eval")
    cap = cap_index(ev, L, args.max_wait)
    C = 1.0 - np.eye(3)

    # 停止規則が使う信念。較正すると「今止めるコスト」が実際の誤り率に近づく。
    if args.calib == "multinomial":
        Bd, Be = calibrate(Pd, labd, Pe)
    else:
        Bd, Be = Pd, Pe

    rows = []
    # 基準：区間検出まで待つ
    sc_e, pred_e = stop_cost(Pe, C)   # 予測クラスは較正前後で変わらない
    evaluate("区間検出まで待つ", np.full(len(labe), len(ev) - 1), pred_e,
             labe, ev, cap, L, rows)
    # 対照：完了確率 τ（現時点の推奨）
    for tau in (0.70, 0.80):
        ok = Pe[..., 1] >= tau
        fired = ok.any(0)
        first = np.where(fired, ok.argmax(0), len(ev) - 1)
        pr = pred_e.copy()
        # 発火し、かつ区間検出より前に発火した区間だけ「完了」で確定させる
        j = np.arange(len(labe))
        use = fired & (first <= cap)
        pr[first[use], j[use]] = 1
        evaluate(f"完了τ={tau:.2f}（現推奨）", first, pr, labe, ev, cap, L, rows)

    for K in [int(k) for k in args.pieces.split()]:
        for a in [float(x) for x in args.alphas.split()]:
            As = backward_induction(Bd, C, a, dt, K)
            first = firmbound_stop(Be, As, C, a, dt)
            evaluate(f"FIRMBOUND K={K} α={a:g}", first, pred_e, labe, ev, cap, L, rows)

    print(f"モデル {args.stem} / eval {len(labe)} 区間 / 発話長 平均 {L.mean():.2f} 秒 / "
          f"区間検出 {args.max_wait} 秒【無音込み】")
    print(f"応答遅延 = 確定時刻 − 発話末（負なら発話中に確定）。較正: {args.calib}。"
          "推定は train_dev のみ、eval には触れていない。\n")
    print(f"{'停止ルール':>24}{'遅延平均':>9}{'完了の遅延':>10}{'応答ミス':>9}"
          f"{'誤割り込み':>10}{'macroF1':>9}{'完了F1':>8}")
    for nm, m_, le, ms, fa, mac, f1 in rows:
        print(f"{nm:>24}{m_:9.3f}{le:10.3f}{ms:9.3f}{fa:10.3f}{mac:9.4f}{f1:8.3f}")


if __name__ == "__main__":
    main()
