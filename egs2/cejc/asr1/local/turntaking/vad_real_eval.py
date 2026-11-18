#!/usr/bin/env python
"""発話区間末を実際の VAD（silero-vad）で決めたときの評価（2026-10-09）。

これまでは「VAD が発火する時刻」を区間末 +0.25 秒で模擬していた。ここでは
vad_silero_probs.py で出した各チャネルの音声確率から、オンラインの VAD を再現して発火時刻を決める。

オンライン VAD（silero の get_speech_timestamps と同じヒステリシス）：
  - 音声確率 p ≥ th で発話中になる
  - 発話中に p < th − 0.15 が続き、その長さが min_sil に達した時点で「発話が終わった」と発火する
    （発火時刻 = 無音の始まり + min_sil。始まりの時刻を「VAD の発話末」と呼ぶ）

区間との対応付け（区間 [b, e]、時刻は録音内の秒）：
  - 区間末の発火：VAD の発話末が e − 0.25 以降にある最初の発火。
    書き起こしの区間末と VAD の発話末のずれは 0.25 秒まで同じ末尾とみなす。
  - 区間内の発火：VAD の発話末が (b, e − 0.25) にある発火。区間の途中のポーズで VAD が終わりと判断したもの。
    実際のシステムではそこで判定が走るので、そこで「完了」と判定すれば発話の途中での割り込みになる。
  - 区間末の発火が e + 10 秒までに無い（他の人の声を拾い続ける等）ときは、e + 10 秒で打ち切る。

判定器の確率は区間開始から 3 秒ぶんしか出していない（evaluate_multitask.py の frame_max_sec）。
発火が 3 秒より後のときは 3 秒時点の確率で判定する（模擬の評価と同じ扱い）。時間は実際の発火時刻で測る。

  python local/turntaking/vad_real_eval.py > tt_analysis/vad_real_eval.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commit_rules import persist, ASR42, TOL                         # noqa: E402
from frame_fusion import prepare, fuse                              # noqa: E402
from frame_rules import utt_lengths                                 # noqa: E402
from keizoku_focus import new_labels, scores                        # noqa: E402

HOP = 512 / 16000          # silero の 1 フレーム = 32 ms
MAXWAIT = 10.0
ALIGN = 0.25


def vad_fires(p, th, min_sil):
    """音声確率 → [(VAD の発話末, 発火時刻)]（秒）。"""
    neg = th - 0.15
    out = []
    speech = False; sil_start = None
    need = int(round(min_sil / HOP))
    for k, v in enumerate(p):
        if not speech:
            if v >= th:
                speech = True; sil_start = None
            continue
        if v >= th:
            sil_start = None
        elif v < neg:
            if sil_start is None:
                sil_start = k
            if k - sil_start + 1 >= need:
                out.append((sil_start * HOP, (k + 1) * HOP))
                speech = False; sil_start = None
        # neg ≤ v < th は状態を変えない（silero と同じ）
    return np.array(out).reshape(-1, 2)


def vad_times(utts, dset, th, min_sil):
    """区間ごとに (区間末の発火時刻, 区間内の発火時刻の一覧)。時刻は区間開始からの秒。"""
    seg = {}
    for l in open(f"data/{dset}/segments"):
        u, rec, b, e = l.split(); seg[u] = (rec, float(b), float(e))
    cache = {}
    tv = np.zeros(len(utts)); inner = []; nofire = np.zeros(len(utts), bool)
    for i, u in enumerate(utts):
        rec, b, e = seg[u]
        if rec not in cache:
            p = np.load(f"tt_analysis/vad_silero/{dset}/{rec}.npy").astype(np.float32)
            cache = {rec: vad_fires(p, th, min_sil)}
        f = cache[rec]
        ends, fires = f[:, 0], f[:, 1]
        j = np.searchsorted(ends, e - ALIGN, side="left")
        if j < len(f) and fires[j] <= e + MAXWAIT:
            tv[i] = fires[j] - b
        else:
            tv[i] = e + MAXWAIT - b; nofire[i] = True
        m = (ends > b) & (ends < e - ALIGN)
        inner.append(fires[m] - b)
    return tv, inner, nofire


class Ev:
    """発火時刻 tv（区間開始から）での評価。早期確定は tv より前の格子時刻だけ。"""
    def __init__(self, P, lab, lens, grid, tv):
        self.P, self.lab, self.lens, self.grid, self.tv = P, lab, lens, grid, tv
        self.idx = np.clip(np.searchsorted(grid, tv, side="right") - 1, 0, len(grid) - 1)
        self.ix = np.arange(len(lab))
        self.am = P.argmax(-1)

    def point(self, ok):
        S = len(self.grid)
        first = np.where(ok.any(0), ok.argmax(0), S)
        fired = (first < S) & (self.grid[np.minimum(first, S - 1)] < self.tv)
        t = np.where(fired, self.grid[np.minimum(first, S - 1)], self.tv)
        pr = np.where(fired, 1, self.am[self.idx, self.ix])
        m1 = self.lab == 1
        return dict(pr=pr, t=t, miss=(pr[m1] != 1).mean(),
                    fa=((pr == 1) & ~m1).sum() / max((~m1).sum(), 1),
                    lat=(t - self.lens)[m1].mean())


def inner_interrupt(P, grid, inner):
    """区間内の発火で「完了」と判定した区間の数（＝発話の途中での割り込み）。
    判定器の確率がある区間開始から 3 秒までの発火だけを見る。"""
    am = P.argmax(-1); n = 0; nany = 0
    for i, ts in enumerate(inner):
        if len(ts) == 0:
            continue
        nany += 1
        hit = False
        for t in ts[ts <= grid[-1]]:
            k = max(np.searchsorted(grid, t, side="right") - 1, 0)
            if am[k, i] == 1:
                hit = True; break
        n += hit
    return n, nany


def head():
    print(f"{'':34s}{'継続再現':>8s}{'継続適合':>8s}{'完了再現':>8s}{'完了適合':>8s}{'相槌F1':>8s}"
          f"{'macro':>8s}{'完了の応答時間':>10s}{'早くなる':>7s}")


def row(lbl, lab, r, base_t=None):
    s = scores(lab, r["pr"])
    sp = f"{np.mean((base_t - r['t'])[lab == 1]):7.2f}" if base_t is not None else f"{'—':>7s}"
    print(f"{lbl:34s}{s['継続'][0]:8.3f}{s['継続'][1]:8.3f}{s['完了'][0]:8.3f}{s['完了'][1]:8.3f}"
          f"{s['相槌'][2]:8.3f}{s['macro']:8.3f}{r['lat']:10.2f}{sp}")


def select_delta(ev, marg):
    """開発データで、応答ミス・誤割り込みが基準 +0.005 以内で最も早い δ（従来の選び方）。"""
    S = len(ev.grid)
    b = ev.point(np.zeros((S, len(ev.lab)), bool))
    best = None
    for dl in np.round(np.arange(0.0, 1.0, 0.02), 2):
        r = ev.point(persist(marg >= dl, 3))
        if r["miss"] <= b["miss"] + TOL and r["fa"] <= b["fa"] + TOL:
            if best is None or r["lat"] < best[1]:
                best = (dl, r["lat"])
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="frame_attn-frame2-blk42xh2_same_n5")
    ap.add_argument("--ckpt", default="valid.loss.ave")
    a = ap.parse_args()
    grid = np.arange(0.05, 3.0 + 1e-9, 0.05)
    Pd, Ptd, labd, uttsd = prepare(a.stem, a.ckpt, "train_dev", "exp/text_bert_asr",
                                   f"{ASR42}/train_dev_dec/text", grid)
    Pe, Pte, labe, uttse = prepare(a.stem, a.ckpt, "eval", "exp/text_bert_asr",
                                   f"{ASR42}/eval/text", grid)
    Ld = utt_lengths(uttsd, "train_dev"); Le = utt_lengths(uttse, "eval")
    labe_new, _ = new_labels(uttse, labe)
    print(f"評価 {len(labe)} 区間（継続 {np.sum(labe == 0)}・完了 {np.sum(labe == 1)}・相槌 {np.sum(labe == 2)}）")

    for th, msil, main_cfg in ((0.5, 0.25, True), (0.5, 0.5, False), (0.7, 0.25, False)):
        tvd, innd, nfd = vad_times(uttsd, "train_dev", th, msil)
        tve, inne, nfe = vad_times(uttse, "eval", th, msil)
        d = tve - Le
        print(f"\n{'#' * 96}\n# VAD：閾値 {th}，発火までの無音 {msil} 秒")
        print(f"#   区間末から発火までの時間（評価データ）：平均 {d.mean():.2f} 秒，中央値 {np.median(d):.2f} 秒，"
              f"四分位 [{np.percentile(d, 25):.2f}, {np.percentile(d, 75):.2f}]")
        for c, nm in ((0, "継続"), (1, "完了"), (2, "相槌")):
            m = labe == c
            print(f"#     {nm}：平均 {d[m].mean():.2f} 秒，中央値 {np.median(d[m]):.2f} 秒，"
                  f"1 秒超 {np.mean(d[m] > 1):.1%}，10 秒以内に発火なし {np.mean(nfe[m]):.1%}")
        print(f"#   模擬（+0.25 秒）より早い発火 {np.mean(d < 0.25):.1%}／遅い発火 {np.mean(d > 0.25):.1%}／"
              f"3 秒の確率が無い（発火が区間開始から 3 秒より後）{np.mean(tve > 3.0):.1%}")
        nin = sum(len(x) > 0 for x in inne)
        print(f"#   区間の途中で VAD が発火した区間 {nin}（{nin / len(inne):.1%}）")

        for cond, w in (("音声のみ（実際の推論）", 1.0), ("参考：BERT も混ぜる", 0.6)):
            if not main_cfg and w != 1.0:
                continue
            Fd = fuse(Pd, Ptd, grid, Ld, w); Fe = fuse(Pe, Pte, grid, Le, w)
            print(f"\n■ {cond}")
            sim_d = Ev(Fd, labd, Ld, grid, Ld + 0.25); sim_e = Ev(Fe, labe, Le, grid, Le + 0.25)
            vad_d = Ev(Fd, labd, Ld, grid, tvd); vad_e = Ev(Fe, labe, Le, grid, tve)
            S = len(grid)
            no = np.zeros((S, len(labe)), bool)
            margd = Fd[..., 1] - Fd[..., 0]; marge = Fe[..., 1] - Fe[..., 0]
            dl_sim = select_delta(sim_d, margd); dl_vad = select_delta(vad_d, margd)
            head()
            bs = sim_e.point(no); bv = vad_e.point(no)
            row("模擬 VAD：VAD まで待つ", labe, bs)
            row(f"模擬 VAD：余裕 δ={dl_sim:.2f}", labe, sim_e.point(persist(marge >= dl_sim, 3)), bs["t"])
            row("実 VAD：VAD まで待つ", labe, bv)
            row(f"実 VAD：余裕 δ={dl_sim:.2f}（模擬で選んだ規則）", labe, vad_e.point(persist(marge >= dl_sim, 3)), bv["t"])
            if dl_vad is not None and dl_vad != dl_sim:
                row(f"実 VAD：余裕 δ={dl_vad:.2f}（実 VAD で選び直し）", labe, vad_e.point(persist(marge >= dl_vad, 3)), bv["t"])
            elif dl_vad is not None:
                print(f"  （実 VAD の開発データで選び直しても δ={dl_vad:.2f} で同じ）")
            row("実 VAD：P(完了)≥0.70", labe, vad_e.point(Fe[..., 1] >= 0.70), bv["t"])
            if main_cfg:
                print("  新しいラベルで評価")
                row("  実 VAD：VAD まで待つ", labe_new, Ev(Fe, labe_new, Le, grid, tve).point(no))
                row(f"  実 VAD：余裕 δ={dl_sim:.2f}", labe_new,
                    Ev(Fe, labe_new, Le, grid, tve).point(persist(marge >= dl_sim, 3)), bv["t"])
            if main_cfg and w == 1.0:
                # 成績が落ちる所：区間末から発火までの時間で区間を分ける（VAD まで待つ判定）
                print("  区間末から発火までの時間ごと（実 VAD・VAD まで待つ）：件数／継続・完了・相槌の再現率／同じ区間を模擬 VAD で判定したときの再現率")
                dd = tve - Le
                for lo, hi, nm in ((-1e9, 0.25, "0.25 秒未満（模擬より早い）"), (0.25, 0.5, "0.25〜0.5 秒"),
                                   (0.5, 1.0, "0.5〜1 秒"), (1.0, 1e9, "1 秒以上")):
                    m = (dd >= lo) & (dd < hi)
                    rv = [np.mean(bv["pr"][m & (labe == c)] == c) for c in range(3)]
                    rs = [np.mean(bs["pr"][m & (labe == c)] == c) for c in range(3)]
                    print(f"    {nm:22s}{m.sum():6d}  実 " + " ".join(f"{x:.3f}" for x in rv)
                          + "  模擬 " + " ".join(f"{x:.3f}" for x in rs))
            n, nany = inner_interrupt(Fe, grid, inne)
            lab_in = np.array([labe[i] for i, x in enumerate(inne) if len(x)])
            print(f"  区間の途中の発火で「完了」と判定（発話途中の割り込み）：{n} 区間"
                  f"（途中で発火した {nany} 区間の {n / max(nany, 1):.1%}，全区間の {n / len(labe):.2%}）"
                  f"／途中で発火した区間の正解：継続 {np.sum(lab_in == 0)}・完了 {np.sum(lab_in == 1)}・相槌 {np.sum(lab_in == 2)}")


if __name__ == "__main__":
    main()
