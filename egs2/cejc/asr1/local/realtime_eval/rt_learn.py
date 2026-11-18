#!/usr/bin/env python
"""応答の決め方を学習で作る試み（2026-11-15）．rt_rule.py と同じ場所で動かす．
無音の中の各時刻について「今応答してよいか」（直前の区間が完了か）を，その無音の中のスコアの動き・無音の長さ・相槌の確率・
直前の声の長さなどから当てる小さな判定器（勾配ブースティング）を開発データで学習し，その出力を新しいスコアとして，
「無音が g1 秒以上・新しいスコアが th1 以上で応答（＋無音 S 秒・相槌の確率 phi 未満で応答）」の決め方を選ぶ．
開発データの中は録音で 2 つに分けて互いに予測する（選ぶときに学習に使った録音の予測を使わない）．評価データは開発の全部で学習した判定器で予測する．
  python rt_learn.py <開発 data> <silero> <書き出し> <評価 data> <silero> <書き出し>"""
import itertools, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from rt_rule import Data, fmt, OFF

def feats(D):
    n = len(D.t); first = D.first
    start = np.zeros(n, int); start[first] = first; start = np.maximum.accumulate(start)      # その無音の最初のフレーム
    sc = np.maximum(D.sc, 0.0)
    cs = np.cumsum(sc); k = np.arange(n) - start + 1
    mean = (cs - np.where(start > 0, cs[start - 1], 0.0)) / k
    mx = sc.copy()
    for i in range(1, 200):                       # 無音の中のこれまでの最大（無音は長くて 5 秒＝約 160 フレーム）
        j = np.arange(n) - i; ok = j >= start
        if not ok.any(): break
        mx[ok] = np.maximum(mx[ok], sc[j[ok]])
    prev = np.where(np.arange(n) > start, sc[np.maximum(np.arange(n) - 3, start)], sc)       # 約 0.1 秒前のスコア
    # 直前の声の長さ・その前の無音の長さ（録音が変わる所は大きな値になるので切る）
    last = np.r_[first[1:] - 1, n - 1]                     # 各無音の最後のフレーム
    se = D.se_all[D.rid[first]]
    voice = np.r_[30.0, np.clip(se[1:] - D.t[last[:-1]], 0, 30)]
    psil = np.r_[5.0, D.run[last[:-1]]]
    r = np.searchsorted(first, np.arange(n), "right") - 1
    return np.c_[sc, D.pb, D.run, mx, mean, sc - prev, voice[r], psil[r], D.sc >= 0], se[r]

def labels(D, se):
    k = np.clip(np.searchsorted(D.b, se, "right") - 1, 0, len(D.b) - 1)
    y = np.full(len(se), -1)                               # -1＝学習に使わない
    inside = (se >= D.b[k]) & (se < D.e[k] - 0.3)
    end = (se >= D.e[k] - 0.3) & (se <= D.e[k] + 0.5) & D.heard[k]
    y[inside] = 0
    y[end] = (D.lab[k][end] == 1).astype(int)
    return y

def fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    u = (y >= 0) & (X[:, 2] >= 0.25) & (X[:, 2] <= 3.3)
    idx = np.flatnonzero(u)[::2]
    return m.fit(X[idx], y[idx])

D = Data(*sys.argv[1:4]); E = Data(*sys.argv[4:7])
XD, seD = feats(D); XE, seE = feats(E); yD = labels(D, seD)
print("開発のフレーム", len(yD), "学習に使う", int((yD >= 0).sum()), "うち応答してよい", round(float((yD == 1).mean()), 3))
rec = (D.t // OFF).astype(int); half = rec % 2 == 0
pD = np.zeros(len(yD))
for a in (half, ~half):
    pD[~a] = fit(XD[a], yD[a]).predict_proba(XD[~a])[:, 1]
pE = fit(XD, yD).predict_proba(XE)[:, 1]
D.sc = pD; E.sc = pE
rows = []
for g1, th1, (S, phi) in itertools.product((0.3, 0.4, 0.5, 0.6), np.round(np.arange(0.20, 0.961, 0.02), 2), [(None, None)] + [(S, p) for S in (2.5, 3.0) for p in (0.1, 0.7, 2.0)]):
    r = (g1, float(th1), None, None, S, phi)
    rows.append((r, D.score(D.events(r))))
print("候補", len(rows))
for cut, miss, bc in ((0.08, 0.10, 0.10), (0.07, 0.12, 0.10), (0.06, 0.12, 0.20), (0.06, 0.15, 0.10), (0.08, 0.08, 0.20), (0.07, 0.10, 0.20)):
    ok = [x for x in rows if x[1]["cut"] <= cut and x[1]["miss"] <= miss and x[1]["bc"] <= bc]
    if not ok: print(f"== cut-in ≤ {cut:.0%}・応答しなかった ≤ {miss:.0%}・相槌の後 ≤ {bc:.0%}：なし"); continue
    r, m = min(ok, key=lambda x: (x[1]["lat"], x[1]["miss"]))
    print(f"== cut-in ≤ {cut:.0%}・応答しなかった ≤ {miss:.0%}・相槌の後 ≤ {bc:.0%}：{r}")
    print("   開発：", fmt(m)); print("   評価：", fmt(E.score(E.events(r))))
