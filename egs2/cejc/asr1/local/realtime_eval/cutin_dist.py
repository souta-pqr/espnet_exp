#!/usr/bin/env python
"""cut-in が起きた時刻の分布（2026-11-27．教授の指摘）．
cut-in rate に数えた区間ごとに，その応答の（無音の長さ，無音が始まってから応答するまでの時間）を書き出す．
無音の長さは silero-vad（提案手法の書き出しの中の確率．どの方式でも同じ）で，声が戻るまで（最大 5 秒）．
応答は無音の中でしか出ないので，点は y = x の線より下に来る．rt_rule.py と同じ場所で動かす．
  python cutin_dist.py <data> <silero> <提案手法の書き出し> <出力 npz> 方式名=rt_eval の出力|rule:g1,th1,g2,th2,S,phi ...
"""
import sys
import numpy as np
from rt_rule import Data, real_events, fmt

D = Data(*sys.argv[1:4])
out = sys.argv[4]
# 無音（rid）ごとの長さ
first = D.first
mx = np.maximum.reduceat(D.run, first)
sil_len = dict(zip(D.rid[first].tolist(), mx.tolist()))


def cutins(ev):
    t, se, cond = ev
    o = np.argsort(t); t, se = t[o], se[o]
    b, e, hor = D.b, D.e, D.hor
    i1 = np.searchsorted(t, e, "left") - 1
    i1c = np.maximum(i1, 0)
    mid = (i1 >= 0) & (t[i1c] >= b) & (se[i1c] >= b)
    i2 = np.maximum(np.searchsorted(t, e, "left"), np.searchsorted(se, b, "left"))
    ok = i2 < len(t); i2c = np.minimum(i2, len(t) - 1)
    ok &= t[i2c] <= hor
    s = D.heard
    end, cont = s & (D.lab == 1), s & (D.lab == 0)
    took = mid | ok
    cut = (end & mid) | (cont & took)
    j = np.where(mid, i1c, i2c)[cut]          # 区間の途中の応答を優先
    tt, ss = t[j], se[j]
    k = np.clip(np.searchsorted(D.t, tt - 1e-6), 0, len(D.t) - 1)   # 応答した時刻のフレーム
    rid = D.rid[k]
    same = np.abs(D.se_all[rid] - ss) < 0.05
    x = np.array([sil_len[r] for r in rid])
    return x, tt - ss, same, cut.sum() / (end.sum() + cont.sum()), D.lab[cut]


res = {}
for a in sys.argv[5:]:
    name, src = a.split("=", 1)
    if src.startswith("rule:"):
        r = tuple(None if v in ("", "-") else float(v) for v in src[5:].split(","))
        ev = D.events(r)
    else:
        ev = real_events(D, src)
    x, y, same, rate, lab = cutins(ev)
    print(f"{name}\t{fmt(D.score(ev))}")
    print(f"   cut-in {len(x)} 件（この数え方での率 {rate:.1%}）・無音の対応がつかない {np.sum(~same)} 件・y > x {np.sum(y > x + 0.05)} 件")
    print(f"   応答までの時間の中央値 {np.median(y):.2f} 秒・無音の長さの中央値 {np.median(x):.2f} 秒・無音が 5 秒を超えた {np.mean(x >= 4.99):.1%}")
    res[name] = np.stack([x, y, lab])
np.savez(out, **res)
