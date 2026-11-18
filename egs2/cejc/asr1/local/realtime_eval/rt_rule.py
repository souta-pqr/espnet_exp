#!/usr/bin/env python
"""rt_dump.py の書き出し（実時間のスコア）に，応答の決め方を後から当てはめて採点する（2026-11-12）．数え方は rt_score.py と同じ．
無音（silero の確率が 0.5 未満の続き）ごとに，決め方を満たした最初の時刻で応答する（満たさず 5 秒続いたら発話が終わる）．

  python rt_rule.py <data> <silero の出力> <書き出し> --rule g1,th1,g2,th2,S,phi [--check <rt_eval の出力>]   1 つの決め方を採点（--check：実際に流した結果と比べる）
  python rt_rule.py <data> <silero の出力> <書き出し> --grid [--cut 0.08] [--eval <評価データ> <silero> <書き出し>]   候補を全部試して選ぶ
"""
import argparse, csv, glob, itertools, os
from collections import defaultdict
import numpy as np

HOP, W, FREE, MAXSIL = 512 / 16000, 5.0, 2.0, 5.0
OFF = 1e5      # 録音ごとに時刻をずらして 1 本の列にする
STALE = os.environ.get("NOSTALE") != "1"   # NOSTALE=1：話している間もスコアが出続ける方式（encac 以外）の書き出し用（11/13）


class Data:
    def __init__(self, data, sdir, ddir):
        recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ddir}/*.npz"))
        self.recs = recs
        seg = defaultdict(list)
        for l in open(f"{data}/segments"):
            u, rec, b, e = l.split(); seg[rec].append((float(b), float(e), u))
        tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
        T, RUN, SC, PB, RID, SE = [], [], [], [], [], []
        B, E, HOR, LAB, G, HEARD, PG = [], [], [], [], [], [], []
        rid0 = 0
        for k, rec in enumerate(recs):
            d = np.load(f"{ddir}/{rec}.npz")
            p, run = d["p"], d["run"]
            n = len(p)
            t = (np.arange(n) + 1) * HOP
            # 各フレームでの最新のスコア（発話の頭では None．声が戻ると 0 が出るので，無音の中のスコアはその無音で出たもの）
            idx = np.searchsorted(d["st"], t + 1e-6) - 1
            sc = np.where(idx >= 0, d["ss"][np.maximum(idx, 0)], -1.0)
            pb = np.where(idx >= 0, d["sb"][np.maximum(idx, 0)], 0.0)
            sil = p < 0.5
            st = sil & ~np.r_[False, sil[:-1]]                 # 無音の始まり
            rid = np.cumsum(st) - 1
            # 無音の前に声があったものだけ（録音の頭の無音は発話の外）
            ok = sil & (rid >= (0 if not sil[0] else 1)) & (run <= MAXSIL + HOP)
            se = (np.flatnonzero(st)) * HOP                     # 最後に声があった時刻
            # 声が戻った直後の 0 より前のスコアを使わない：無音の始まりより前に出たスコアは 0 とみなす
            first = np.flatnonzero(st)
            stale = idx < np.searchsorted(d["st"], t[first][rid] - HOP + 1e-6)
            if STALE: sc = np.where(stale & (sc > 0), 0.0, sc); pb = np.where(stale, 0.0, pb)
            T.append(t[ok] + k * OFF); RUN.append(run[ok]); SC.append(sc[ok]); PB.append(pb[ok]); RID.append(rid[ok] + rid0); SE.append(se + k * OFF)
            rid0 += len(se)
            lst = sorted(seg[rec]); x = np.load(f"{sdir}/{rec}.npy")
            for j, (b, e, u) in enumerate(lst):
                if u not in tag: continue
                g = max(0.0, lst[j + 1][0] - e) if j + 1 < len(lst) else np.inf
                pu = lst[j - 1] if j else None
                B.append(b + k * OFF); E.append(e + k * OFF); HOR.append(e + min(g, W) + k * OFF); LAB.append(tag[u]); G.append(g)
                HEARD.append(x[int(b / HOP):int(e / HOP) + 1].max(initial=0) >= 0.5)
                PG.append(b - pu[1] if pu is not None and tag.get(pu[2]) == 0 else -1.0)   # 直前が継続のとき，その間
        self.t, self.run, self.sc, self.pb, self.rid = map(np.concatenate, (T, RUN, SC, PB, RID))
        self.se_all = np.concatenate(SE)
        self.first = np.flatnonzero(np.r_[True, self.rid[1:] != self.rid[:-1]])
        self.b, self.e, self.hor, self.lab, self.g, self.heard, self.pg = map(np.array, (B, E, HOR, LAB, G, HEARD, PG))

    def events(self, r):
        """決め方 r＝(g1, th1, g2, th2, S, phi) → 応答の (時刻, 最後に声があった時刻, 条件)"""
        g1, th1, g2, th2, S, phi = r
        c1 = (self.run >= g1) & (self.sc >= th1)
        c2 = (self.run >= g2) & (self.sc >= th2) if g2 is not None else np.zeros(len(c1), bool)
        c3 = (self.run >= S) & (self.pb < phi) if S is not None else np.zeros(len(c1), bool)
        m = c1 | c2 | c3
        pos = np.arange(len(m)); big = len(m)
        f = np.minimum.reduceat(np.where(m, pos, big), self.first)
        hit = f < big
        f = f[hit]
        cond = np.where(c1[f], 1, np.where(c2[f], 2, 3))
        return self.t[f], self.se_all[self.rid[f]], cond

    def score(self, ev, heard_only=True):
        t, se, cond = ev
        o = np.argsort(t); t, se, cond = t[o], se[o], cond[o]
        b, e, hor = self.b, self.e, self.hor
        # 途中で応答：b <= t < e で se >= b のものがあるか（se は t の順に増える → その範囲の最後のものを見る）
        i1 = np.searchsorted(t, e, "left") - 1
        mid = (i1 >= 0) & (t[np.maximum(i1, 0)] >= b) & (se[np.maximum(i1, 0)] >= b)
        i2 = np.maximum(np.searchsorted(t, e, "left"), np.searchsorted(se, b, "left"))
        ok = (i2 < len(t)); i2c = np.minimum(i2, len(t) - 1)
        ok &= t[i2c] <= hor
        lt = np.where(ok, t[i2c] - e, np.inf)
        cd = np.where(ok, cond[i2c], 0)
        s = self.heard if heard_only else np.ones(len(b), bool)
        lab, g = self.lab, self.g
        took = mid | np.isfinite(lt)
        end, cont, bc = s & (lab == 1), s & (lab == 0), s & (lab == 2)
        st = end & (g >= FREE)
        la = lt[st & ~mid]
        pg = self.pg
        hes = st & ~mid & (pg >= 0.4) & (pg <= 2.0)
        lp = cont & (g >= 2) & (g < 3)
        return dict(cut=((end & mid).sum() + (cont & took).sum()) / (end.sum() + cont.sum()), lat=float(np.median(la)), miss=(st & ~mid & ~np.isfinite(lt)).sum() / st.sum(),
                    lat_ok=float(np.median(la[np.isfinite(la)])), bc=took[bc].mean(), hes=float(np.median(lt[hes])), oth=float(np.median(lt[st & ~mid & ~hes])),
                    wait=1 - took[lp].mean(), p90=float(np.percentile(la[np.isfinite(la)], 90)), mean_ok=float(la[np.isfinite(la)].mean()), le1=float(np.mean(la <= 1.0)), c2=float(np.mean(cd[st & ~mid] == 2)), n=int(st.sum()))


def fmt(m):
    return (f"cut-in rate {m['cut']:.1%}  latency {m['lat']:.2f}  応答しなかった {m['miss']:.1%}  応答した分の latency {m['lat_ok']:.2f}  1 秒以内 {m['le1']:.1%}  "
            f"相槌の後に応答 {m['bc']:.1%}  言いよどんだ後 {m['hes']:.2f}／それ以外 {m['oth']:.2f}  間 2〜3 秒で待てた {m['wait']:.1%}")


def real_events(D, rdir):
    T, SE, C = [], [], []
    for k, rec in enumerate(D.recs):
        for r in csv.DictReader(open(f"{rdir}/{rec}.tsv"), delimiter="\t"):
            if r["cond"] != "0":
                T.append(float(r["t"]) + k * OFF); SE.append(float(r["speech_end"]) + k * OFF); C.append(int(r["cond"]))
    return np.array(T), np.array(SE), np.array(C)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("sdir"); ap.add_argument("ddir")
    ap.add_argument("--rule", default=None); ap.add_argument("--check", default=None)
    ap.add_argument("--grid", action="store_true"); ap.add_argument("--cut", type=float, default=0.08)
    ap.add_argument("--eval", nargs=3, default=None)
    a = ap.parse_args()
    D = Data(a.data, a.sdir, a.ddir)
    print(f"録音 {len(D.recs)}・無音 {len(D.se_all)}")
    if a.rule:
        r = tuple(None if v in ("", "-") else float(v) for v in a.rule.split(","))
        ev = D.events(r)
        print("後から当てはめた：", len(ev[0]), "回", fmt(D.score(ev)))
        if a.check:
            rv = real_events(D, a.check)
            print("実際に流した  ：", len(rv[0]), "回", fmt(D.score(rv)))
            a_, b_ = np.sort(ev[0]), np.sort(rv[0])
            j = np.minimum(np.searchsorted(a_, b_), len(a_) - 1)
            d = np.minimum(np.abs(a_[j] - b_), np.abs(a_[np.maximum(j - 1, 0)] - b_))
            print(f"実際の応答のうち，後から当てはめた応答が 0.05 秒以内にある {np.mean(d <= 0.05):.1%}・0.2 秒以内 {np.mean(d <= 0.2):.1%}")
    if a.grid:
        G1 = (0.3, 0.4, 0.5, 0.6, 0.8); TH1 = np.round(np.arange(0.50, 0.981, 0.02), 2)
        G2 = (None, 1.0, 1.5, 2.0, 2.5); TH2 = (0.3, 0.4, 0.5, 0.6)
        S3 = ((3.0, 0.1), (3.0, 0.3), (3.0, 2.0), (2.5, 0.1), (2.5, 2.0))
        rows = []
        for g1, th1, g2, (S, phi) in itertools.product(G1, TH1, G2, S3):
            for th2 in (TH2 if g2 is not None else (None,)):
                if g2 is not None and (g2 <= g1 or th2 >= th1): continue
                r = (g1, float(th1), g2, th2, S, phi)
                rows.append((r, D.score(D.events(r))))
        print(f"候補 {len(rows)}")
        E = Data(*a.eval) if a.eval else None
        for name, key in (("latency が最小", lambda m: (m["lat"], m["miss"])), ("latency が最小（応答しなかった ≤ 10%）", lambda m: (m["miss"] > 0.10, m["lat"], m["miss"]))):
            for cut in (a.cut, 0.07, 0.06):
                ok = [x for x in rows if x[1]["cut"] <= cut]
                r, m = min(ok, key=lambda x: key(x[1]))
                print(f"== {name}・cut-in rate ≤ {cut:.0%}：g1 {r[0]} θ1 {r[1]} g2 {r[2]} θ2 {r[3]} S {r[4]} φ {r[5]}")
                print("   開発：", fmt(m))
                if E: print("   評価：", fmt(E.score(E.events(r))))
