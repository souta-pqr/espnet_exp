#!/usr/bin/env python
"""応答の評価：遮り率と遅れの曲線（2026-10-16）。先行研究（Shannon+ 2017，Raux+ 2008，Udupa+ 2025）に合わせる。

正解は書き起こしの区間とラベル（tag：0=継続，1=完了，2=相槌）。
判定器を区間の先頭から「区間末 + 次の発話の開始まで（最大 --window 秒）」まで流し，
「完了」の確率が閾値 θ を --sustain 秒続けて超えた最初の時刻で応答する。
判定の時刻にはエンコーダの先読み（--lookahead 秒）を足す。

  完了区間：区間末より前に応答 → 遮り．区間末の後に応答 → 遅れ = 応答時刻 − 区間末．範囲内で応答しない → 応答なし
  継続区間：範囲内（区間の途中と区間末の後）で応答 → 遮り
  相槌区間：別に数える（範囲内で応答した割合）

  遮り率 = 遮った区間 / （完了区間 + 継続区間）
  遅れ   = 完了区間のうち遮らなかったもので数える．応答なしは +∞ として中央値と 90% 点に含める（Shannon+ 2017 の EP50/EP90 と同じく全区間で数える）

ベースライン：silero-vad の確率で無音の長さだけを見て応答する（silero の既定：0.5 以上で音声，0.35 未満から無音を数え始める）。

  python local/turntaking/eot_eval.py --data data/eval_g \\
      --dump tt_preds/frame_..._eval_g_valid.loss.ave_post2.0.tsv --name ge \\
      --silero tt_analysis/vad_silero/eval_g --out tt_analysis/eot_eval_eval_g

--pick_from <開発データの出力> を付けると，開発データで遮り率が --pick_cutin 以下になる点のうち
「次の発話まで window 秒以上」の完了区間での遅れの中央値が最小の動作点を選び，その点での値を
会話単位のブートストラップ（--n_boot 回）による 95% 区間つきで pick.txt に書く。--pair a b で a − b の差も出す。
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

THETAS = np.round(np.arange(0.05, 0.99, 0.01), 3)
SIL_HOP = 512 / 16000          # silero の 1 フレーム


def load_segments(data, tag_file=None):
    seg = {}
    by_rec = defaultdict(list)
    for l in open(f"{data}/segments"):
        u, rec, b, e = l.split()
        seg[u] = (rec, float(b), float(e))
        by_rec[rec].append((float(b), u))
    gap = {}                      # 同じチャネルの次の区間の開始まで（重なりは 0）
    for rec, lst in by_rec.items():
        lst.sort()
        for (b, u), nxt in zip(lst, lst[1:] + [None]):
            e = seg[u][2]
            gap[u] = np.inf if nxt is None else max(0.0, nxt[0] - e)
    tag = {l.split()[0]: int(l.split()[1]) for l in open(tag_file or f"{data}/tag")}
    return seg, gap, tag


def load_dump(path):
    el, pe, pb = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(path) as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for row in r:
            el[row[0]].append(float(row[3])); pe[row[0]].append(float(row[5])); pb[row[0]].append(float(row[6]))
    return {u: (np.array(el[u]), np.array(pe[u]), np.array(pb[u])) for u in el}


def first_cross(score, thetas):
    """各 θ について score ≥ θ となる最初の添字（なければ len）。"""
    pm = np.maximum.accumulate(score) if len(score) else score
    return np.searchsorted(pm, thetas, side="left")


def sliding_min(x, k):
    if k <= 1 or len(x) < k:
        return x if k <= 1 else np.full(0, 0.0)
    return np.lib.stride_tricks.sliding_window_view(x, k).min(axis=1)


def summarize(t_resp, utts, seg, gap, tag, window, free_sec):
    """t_resp: (区間数, 動作点数) 区間の先頭からの応答時刻（応答なしは inf）。"""
    dur = np.array([seg[u][2] - seg[u][1] for u in utts])[:, None]
    hor = dur + np.minimum(np.array([gap[u] for u in utts]), window)[:, None]
    lab = np.array([tag[u] for u in utts])
    resp = t_resp <= hor
    end, cont, bc = lab == 1, lab == 0, lab == 2
    cut_end = resp & (t_resp < dur)
    n_cut = cut_end[end].sum(0) + resp[cont].sum(0)
    cutin = n_cut / (end.sum() + cont.sum())
    lat = np.where(resp & (t_resp >= dur), t_resp - dur, np.inf)[end]
    ok = ~cut_end[end]                                   # 遮らなかった完了区間
    # 完了区間を「同じ話者が free_sec 秒以内に次の発話を始める」かどうかで分けても数える
    resume = (np.array([gap[u] for u in utts]) < free_sec)[end]
    groups = {"": np.ones_like(resume), "_free": ~resume, "_resume": resume}
    out = dict(cutin=cutin, cut_end=cut_end[end].mean(0), cut_cont=resp[cont].mean(0),
               bc_resp=resp[bc].mean(0), n_free=int((~resume).sum()), n_resume=int(resume.sum()))
    for g, gm in groups.items():
        p50, p90, noresp = [], [], []
        for j in range(t_resp.shape[1]):
            x = lat[ok[:, j] & gm, j]
            p50.append(np.quantile(x, 0.5) if len(x) else np.nan)
            p90.append(np.quantile(x, 0.9) if len(x) else np.nan)
            noresp.append(np.isinf(x).mean() if len(x) else np.nan)
        out["p50" + g], out["p90" + g], out["noresp" + g] = map(np.array, (p50, p90, noresp))
        out["cut_end" + g] = cut_end[end][gm].mean(0)
    return out


def model_points(dump, utts, sustains, lookahead):
    cols, names = [], []
    for s in sustains:
        T = np.full((len(utts), len(THETAS)), np.inf)
        for i, u in enumerate(utts):
            el, pe = dump[u][:2]
            k = max(1, int(round(s / (el[1] - el[0])))) if len(el) > 1 else 1
            m = sliding_min(pe, k)
            idx = first_cross(m, THETAS)
            hit = idx < len(m)
            T[i, hit] = el[idx[hit] + k - 1] + lookahead
        cols.append(T)
        names += [f"θ={t:.2f} 継続{s:g}s" for t in THETAS]
    return np.concatenate(cols, 1), names


def silence_runs(sdir, utts, seg, window):
    """区間ごとに，silero の各フレームの終わりの時刻（区間の先頭から）と，そこまで無音が続いた長さ。"""
    cache, out = {}, {}
    for u in utts:
        rec, b, e = seg[u]
        if rec not in cache:
            cache[rec] = np.load(f"{sdir}/{rec}.npy").astype(np.float32)
        p = cache[rec]
        i0 = int(b / SIL_HOP); i1 = min(len(p), int((e + window) / SIL_HOP) + 1)
        x = p[i0:i1]
        # 無音が続いた長さ（0.35 未満で数え始め，0.5 以上で戻す）
        run = np.zeros(len(x)); start = -1
        for j, v in enumerate(x):
            if v >= 0.5:
                start = -1
            elif start < 0 and v < 0.35:
                start = j
            run[j] = 0 if start < 0 else (j - start + 1) * SIL_HOP
        out[u] = ((i0 + np.arange(len(x)) + 1) * SIL_HOP - b, run)
    return out


def silero_points(runs, utts, sil_max=2.0):
    sil = np.round(np.arange(0.05, sil_max + 1e-4, 0.025), 3)
    T = np.full((len(utts), len(sil)), np.inf)
    for i, u in enumerate(utts):
        t, run = runs[u]
        idx = first_cross(run, sil)
        hit = idx < len(run)
        T[i, hit] = t[idx[hit]]
    return T, [f"無音{s:g}s" for s in sil]


def gated_points(dump, runs, utts, lookahead, gates):
    """無音が gate 秒続き，かつその時点で使える最新の「完了」の確率が θ 以上になった最初の時刻で応答する。"""
    cols, names = [], []
    for g in gates:
        T = np.full((len(utts), len(THETAS)), np.inf)
        for i, u in enumerate(utts):
            t, run = runs[u]
            el, pe = dump[u][:2]
            k = np.searchsorted(el + lookahead, t, side="right") - 1   # 時刻 t で使える最新の判定
            p = np.where(k >= 0, pe[np.clip(k, 0, len(pe) - 1)], -1.0)
            score = np.where(run >= g, p, -1.0)
            idx = first_cross(score, THETAS)
            hit = idx < len(score)
            T[i, hit] = t[idx[hit]]
        cols.append(T)
        names += [f"無音{g:g}s θ={th:.2f}" for th in THETAS]
    return np.concatenate(cols, 1), names


def fallback_points(dump, runs, utts, lookahead, S, phi):
    """無音が S 秒続き，かつその時点で使える最新の「相槌」の確率が phi 未満になった最初の時刻。"""
    T = np.full(len(utts), np.inf)
    for i, u in enumerate(utts):
        t, run = runs[u]
        el, _, pb = dump[u]
        k = np.searchsorted(el + lookahead, t, side="right") - 1
        b = np.where(k >= 0, pb[np.clip(k, 0, len(pb) - 1)], 1.0)
        hit = np.where((run >= S) & (b < phi))[0]
        if len(hit):
            T[i] = t[hit[0]]
    return T


def best_at(res, names, key, targets, minimize):
    """key ≤ target を満たす点のうち minimize が最小のもの。"""
    out = []
    for t in targets:
        ok = np.where(res[key] <= t + 1e-12)[0]
        if len(ok) == 0:
            out.append((t, None)); continue
        v = np.where(np.isfinite(res[minimize][ok]), res[minimize][ok], np.inf)
        j = ok[np.lexsort((res["noresp"][ok], v))[0]]      # 同じなら応答なしが少ない点
        out.append((t, j))
    return out


def fmt(x):
    return "> 窓" if not np.isfinite(x) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dump", nargs="*", default=[])
    ap.add_argument("--name", nargs="*", default=[])
    ap.add_argument("--silero", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="", help="正解ラベル（既定は <data>/tag）")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--lookahead", type=float, default=0.2,
                    help="エンコーダの先読み（block 42・hop 3・look_ahead 3 で最大 6 フレーム ≒ 0.2 秒）")
    ap.add_argument("--lookaheads", nargs="*", type=float, default=None,
                    help="--dump ごとの先読み（秒）．省略時はすべて --lookahead（VAP・Smart Turn など先読みの無いものは 0）")
    ap.add_argument("--sustain", default="0 0.1 0.2 0.3 0.5")
    ap.add_argument("--gates", default="0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.8 1.0",
                    help="判定器と silero を組み合わせるときの無音の長さ（--silero が要る）")
    ap.add_argument("--cutin_targets", default="0.02 0.04 0.05 0.06 0.10")
    ap.add_argument("--latency_targets", default="0.2 0.3 0.5")
    ap.add_argument("--fallback", default="", help="無音がこの秒数続いたら判定器に関係なく応答する（空白区切りで複数）")
    ap.add_argument("--fb_bc", default="1.01", help="長い無音で応答するときの相槌の確率の上限（空白区切り．1.01 は条件なし）")
    ap.add_argument("--fb_gates", default="0.6 0.8 1.0", help="長い無音の規則と組み合わせる判定器側の無音の長さ")
    ap.add_argument("--free_sec", type=float, default=None,
                    help="遅れを数える完了区間：次の発話までこの秒数以上空くもの（既定は --window）")
    ap.add_argument("--pick_from", default="", help="動作点を選ぶ開発データの出力（curve_*.tsv がある所）")
    ap.add_argument("--pick_cutin", type=float, default=0.05)
    ap.add_argument("--sil_max", type=float, default=2.0, help="silero-vad のみの無音の長さの候補の上限（秒）")
    ap.add_argument("--pick_miss", type=float, default=1.0, help="動作点を選ぶときの応答なし（次の発話まで 2 秒以上の完了区間）の上限")
    ap.add_argument("--pair", nargs=2, default=None, help="差を出す 2 つの方式名")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    if a.free_sec is None:
        a.free_sec = a.window
    seg, gap, tag = load_segments(a.data, a.tag or None)
    sustains = [float(x) for x in a.sustain.split()]
    systems = []
    dumps = [load_dump(p) for p in a.dump]
    utts = sorted(set(tag) & set(seg))
    for d in dumps:
        utts = [u for u in utts if u in d]
    print(f"区間 {len(utts)}（継続 {sum(tag[u]==0 for u in utts)}，完了 {sum(tag[u]==1 for u in utts)}，"
          f"相槌 {sum(tag[u]==2 for u in utts)}）")
    runs = silence_runs(a.silero, utts, seg, a.window) if a.silero else None
    gates = [float(x) for x in a.gates.split()]
    fb_gates = [float(x) for x in a.fb_gates.split()]
    las = a.lookaheads if a.lookaheads else [a.lookahead] * len(dumps)
    assert len(las) == len(dumps), "--lookaheads の数が --dump と合わない"
    for nm, d, la in zip(a.name, dumps, las):
        T, names = model_points(d, utts, sustains, la)
        systems.append((nm, T, names))
        if runs is not None:
            T, names = gated_points(d, runs, utts, la, gates)
            systems.append((f"{nm}+silero", T, names))
            if a.fallback:
                # 判定器が「完了」と言わなくても，無音が S 秒続き，かつ相槌の確率が φ 未満なら応答する
                sel = [j for j, n in enumerate(names) if float(n.split("s")[0][2:]) in fb_gates]
                cols, fn = [], []
                for S in [float(x) for x in a.fallback.split()]:
                    for phi in [float(x) for x in a.fb_bc.split()]:
                        Tf = fallback_points(d, runs, utts, la, S, phi)
                        cols.append(np.minimum(T[:, sel], Tf[:, None]))
                        fn += [f"{names[j]} 無音{S:g}s・相槌<{phi:g}で応答" for j in sel]
                systems.append((f"{nm}+silero+fb", np.concatenate(cols, 1), fn))
    if runs is not None:
        T, names = silero_points(runs, utts, a.sil_max)
        systems.append(("silero", T, names))

    Path(a.out).mkdir(parents=True, exist_ok=True)
    ct = [float(x) for x in a.cutin_targets.split()]
    lt = [float(x) for x in a.latency_targets.split()]
    lines = []
    for nm, T, names in systems:
        r = summarize(T, utts, seg, gap, tag, a.window, a.free_sec)
        with open(f"{a.out}/curve_{nm}.tsv", "w") as f:
            f.write("point\tcutin\tcut_end\tcut_cont\tlat_p50\tlat_p90\tnoresp\tbc_resp"
                    "\tlat_p50_free\tlat_p90_free\tnoresp_free\n")
            for j, n in enumerate(names):
                f.write(f"{n}\t{r['cutin'][j]:.4f}\t{r['cut_end'][j]:.4f}\t{r['cut_cont'][j]:.4f}"
                        f"\t{r['p50'][j]:.4f}\t{r['p90'][j]:.4f}\t{r['noresp'][j]:.4f}\t{r['bc_resp'][j]:.4f}"
                        f"\t{r['p50_free'][j]:.4f}\t{r['p90_free'][j]:.4f}\t{r['noresp_free'][j]:.4f}\n")
        lines.append(f"## {nm}")
        lines.append("遮り率の上限\t遅れ中央値\t遅れ90%点\t応答なし\t相槌に応答\t動作点")
        for t, j in best_at(r, names, "cutin", ct, "p50"):
            if j is None:
                lines.append(f"{t:.0%}\t（届かない）"); continue
            lines.append(f"{t:.0%}\t{fmt(r['p50'][j])}\t{fmt(r['p90'][j])}\t{r['noresp'][j]:.3f}"
                         f"\t{r['bc_resp'][j]:.3f}\t{names[j]}（遮り {r['cutin'][j]:.3f}）")
            for g, gn in (("_free", f"次の発話まで {a.free_sec:g} 秒以上（{r['n_free']} 区間）"),
                          ("_resume", f"{a.free_sec:g} 秒以内に同じ話者が続ける（{r['n_resume']} 区間）")):
                lines.append(f"  └ {gn}\t{fmt(r['p50' + g][j])}\t{fmt(r['p90' + g][j])}"
                             f"\t{r['noresp' + g][j]:.3f}\t完了区間を遮った {r['cut_end' + g][j]:.3f}")
        lines.append("遅れ中央値の上限\t遮り率\t遅れ90%点\t応答なし\t相槌に応答\t動作点")
        for t, j in best_at(r, names, "p50", lt, "cutin"):
            if j is None:
                lines.append(f"{t:g}s\t（届かない）"); continue
            lines.append(f"{t:g}s\t{r['cutin'][j]:.3f}\t{fmt(r['p90'][j])}\t{r['noresp'][j]:.3f}"
                         f"\t{r['bc_resp'][j]:.3f}\t{names[j]}（中央値 {fmt(r['p50'][j])}）")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    open(f"{a.out}/summary.txt", "w").write(txt + "\n")
    if a.pick_from:
        pick(a, systems, utts, seg, gap, tag)


def pick_point(path, target, miss=1.0):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    ok = [r for r in rows if float(r["cutin"]) <= target + 1e-12 and float(r["noresp_free"]) <= miss + 1e-12]
    if not ok:
        return None
    inf_nan = lambda x: np.inf if np.isnan(float(x)) else float(x)     # 窓の外の分位点は nan になることがある
    return min(ok, key=lambda r: (inf_nan(r["lat_p50_free"]), float(r["noresp_free"])))


def point_stats(t, idx, dur, hor, lab, free, window):
    """1 つの動作点（t：区間ごとの応答時刻）の値。idx は使う区間の添字（ブートストラップで重複あり）。"""
    t, d, h, l, fr = t[idx], dur[idx], hor[idx], lab[idx], free[idx]
    resp = t <= h
    end, cont, bc = l == 1, l == 0, l == 2
    cut_end = resp & (t < d)
    cutin = (cut_end[end].sum() + resp[cont].sum()) / (end.sum() + cont.sum())
    m = end & fr & ~cut_end
    lat = np.where(resp & (t >= d), t - d, np.inf)[m]
    q = lambda x: np.inf if np.isnan(x) else x
    return dict(cutin=cutin, p50_free=q(np.quantile(lat, 0.5)), p90_free=q(np.quantile(lat, 0.9)),
                noresp_free=np.isinf(lat).mean(), bc_resp=resp[bc].mean())


def pick(a, systems, utts, seg, gap, tag):
    dur = np.array([seg[u][2] - seg[u][1] for u in utts])
    g = np.array([gap[u] for u in utts])
    hor = dur + np.minimum(g, a.window)
    lab = np.array([tag[u] for u in utts])
    free = g >= a.free_sec
    # 会話の単位は末尾の a・b を除いたもの（K008_001a と K008_001b は同じ会話．local/cejc_groupsplit.py と同じ）
    conv = np.array([re.sub(r"[a-z]$", "", seg[u][0].rsplit("_", 1)[0]) for u in utts])
    convs = sorted(set(conv))
    members = [np.where(conv == c)[0] for c in convs]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[k] for k in rng.integers(0, len(convs), len(convs))]) for _ in range(a.n_boot)]
    allidx = np.arange(len(utts))
    keys = ("cutin", "p50_free", "p90_free", "noresp_free", "bc_resp")
    cols, lines = {}, [f"開発データ（{a.pick_from}）で遮り率 {a.pick_cutin:.0%} 以下・応答なし {a.pick_miss:.0%} 以下の点を選び，この評価データで測る。"
                       f"95% 区間は会話単位のブートストラップ（{len(convs)} 会話，{a.n_boot} 回）。",
                       "方式\t動作点\t遮り率\t遅れ中央値\t遅れ90%点\t応答なし\t相槌に応答"]
    for nm, T, names in systems:
        r = pick_point(f"{a.pick_from}/curve_{nm}.tsv", a.pick_cutin, a.pick_miss)
        if r is None:
            lines.append(f"{nm}\t（届かない）"); continue
        t = T[:, names.index(r["point"])]
        cols[nm] = t
        est = point_stats(t, allidx, dur, hor, lab, free, a.window)
        bs = [point_stats(t, b, dur, hor, lab, free, a.window) for b in boots]
        cell = []
        for k in keys:
            v = np.array([x[k] for x in bs]); lo, hi = np.quantile(v, [0.025, 0.975])
            cell.append(f"{fmt(est[k])} [{fmt(lo)}, {fmt(hi)}]")
        lines.append(f"{nm}\t{r['point']}（開発 遮り {float(r['cutin']):.3f}）\t" + "\t".join(cell))
    if a.pair:
        x, y = a.pair
        cell = []
        for k in keys:
            d = [point_stats(cols[x], b, dur, hor, lab, free, a.window)[k]
                 - point_stats(cols[y], b, dur, hor, lab, free, a.window)[k] for b in boots]
            e = point_stats(cols[x], allidx, dur, hor, lab, free, a.window)[k] \
                - point_stats(cols[y], allidx, dur, hor, lab, free, a.window)[k]
            lo, hi = np.quantile(d, [0.025, 0.975])
            cell.append(f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"差 {x} − {y}\t\t" + "\t".join(cell))
    txt = "\n".join(lines)
    print(txt)
    open(f"{a.out}/pick.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
