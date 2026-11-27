"""無音の始まりが区間末より 1 秒以上遅れる完了の区間（応答しなかった割合 約 35%）の中身を調べる（2026-11-23）．
区間末 e から，その後で最初に 0.4 秒続く無音の頭 s0 までに何があるかを数える：
  - 同じ録音の別のチャネル（相手のマイク）の書き起こしの区間が重なる → 相手の声の漏れ込みの見込み
  - 自分のチャネルの音の大きさ（区間の中の音声の大きさとの比）
  - その区間の後の応答（ev2tsv の出力）・その無音で出た P(完了) の最大
  python late_sil.py <data> <silero> <書き出し> <応答の tsv> <出力の tsv>
"""
import csv, glob, os, sys, wave
from collections import defaultdict, Counter
import numpy as np

data, sdir, ddir, rdir, out = sys.argv[1:6]
H = 512 / 16000
seg = defaultdict(list)
for l in open(f"{data}/segments"):
    u, r, b, e = l.split(); seg[r].append((float(b), float(e), u))
tag = {l.split()[0]: int(l.split()[1]) for l in open(f"{data}/tag")}
txt = {l.split(" ", 1)[0]: l.rstrip("\n").split(" ", 1)[1].replace(" ", "") if " " in l else "" for l in open(f"{data}/text")}
wav = dict(l.split() for l in open(f"{data}/wav.scp"))
sess = defaultdict(list)
for r in seg:
    sess[r.rsplit("_", 1)[0]].append(r)


def rms(path, a, b):
    w = wave.open(path); sr = w.getframerate(); w.setpos(int(a * sr))
    x = np.frombuffer(w.readframes(max(1, int((b - a) * sr))), dtype=np.int16).astype(np.float32) / 32768
    return float(np.sqrt(np.mean(x ** 2)) + 1e-8)


rows = []; C = Counter()
for r in sorted(seg):
    if not os.path.exists(f"{ddir}/{r}.npz"):
        continue
    p = np.load(f"{sdir}/{r}.npy").astype(np.float32); sp = p >= 0.5
    ed = np.diff(np.r_[0, (~sp).astype(int), 0]); s0, s1 = np.where(ed == 1)[0], np.where(ed == -1)[0]
    lg = (s1 - s0) * H >= 0.4; ls0 = s0[lg] * H
    d = np.load(f"{ddir}/{r}.npz")
    ev = [float(x["t"]) for x in csv.DictReader(open(f"{rdir}/{r}.tsv"), delimiter="\t")]
    other = [(b, e) for rr in sess[r.rsplit("_", 1)[0]] if rr != r for b, e, _ in seg[rr]]
    lst = sorted(seg[r])
    for k, (b, e, u) in enumerate(lst):
        if tag.get(u) != 1:
            continue
        g = lst[k + 1][0] - e if k + 1 < len(lst) else np.inf
        if g < 2.0 or not sp[int(b / H):int(e / H) + 1].any():
            continue
        j = np.where((ls0 >= e - 0.3) & (ls0 <= e + 5))[0]
        if not len(j) or ls0[j[0]] - e < 1.0:
            continue
        s = ls0[j[0]]
        ov = sum(max(0.0, min(s, oe) - max(e, ob)) for ob, oe in other) / (s - e)
        win = e + min(g, 5.0)
        resp = [t for t in ev if e <= t <= win]
        late = [t for t in ev if win < t <= win + 3]
        m = (d["st"] >= s) & (d["st"] <= s + 3.1)
        pmax = float(d["ss"][m].max()) if m.any() else 0.0
        ratio = rms(wav[r], e, s) / rms(wav[r], b, e)
        kind = "相手の声が重なる（半分以上）" if ov >= 0.5 else ("相手の声が少し重なる" if ov > 0 else "相手の声なし")
        C[(kind, bool(resp))] += 1
        rows.append((r, u, f"{b:.2f}", f"{e:.2f}", f"{s:.2f}", f"{s - e:.2f}", f"{ov:.2f}", f"{ratio:.2f}", f"{pmax:.2f}", "応答した" if resp else ("窓の後に応答" if late else "応答しなかった"), txt.get(u, "")))
with open(out, "w") as f:
    f.write("録音\t区間\t頭\t区間末\t無音の頭\t遅れ\t相手の声の重なり\t音の大きさの比\tP完了の最大\t応答\t書き起こし\n")
    for x in rows:
        f.write("\t".join(x) + "\n")
n = len(rows); print("区間", n)
for kind in ["相手の声が重なる（半分以上）", "相手の声が少し重なる", "相手の声なし"]:
    a, m_ = C[(kind, True)], C[(kind, False)]
    print(f"{kind}\t{a + m_}（{(a + m_) / n:.1%}）\t応答しなかった {m_ / max(1, a + m_):.1%}")
R = [x for x in rows if x[9] != "応答した"]
print("応答しなかった", len(R), "うち窓の後に応答", sum(x[9] == "窓の後に応答" for x in R))
for name, idx in [("遅れ", 5), ("音の大きさの比", 7), ("P完了の最大", 8)]:
    v = np.array([float(x[idx]) for x in R]); w = np.array([float(x[idx]) for x in rows if x[9] == "応答した"])
    print(f"{name}：応答しなかった 中央値 {np.median(v):.2f}／応答した {np.median(w):.2f}")
