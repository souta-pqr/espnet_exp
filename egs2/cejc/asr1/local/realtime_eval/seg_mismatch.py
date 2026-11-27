"""書き起こしの区間と，実時間と同じ規則で作った silero-vad の音声区間のずれを数える（2026-11-19）．
音声区間：確率 0.5 以上を音声，0.4 秒以上の無音で区切る，頭は音声の始まりの 0.2 秒前（epd/models/encac.py と同じ）．
  python seg_mismatch.py <data dir> <silero dir>
"""
import sys
from collections import Counter, defaultdict
import numpy as np

d, sd = sys.argv[1], sys.argv[2]
H = 512 / 16000
tag = dict(l.split() for l in open(f"{d}/tag"))
segs = defaultdict(list)
for l in open(f"{d}/segments"):
    u, r, b, e = l.split()
    segs[r].append((float(b), float(e), tag[u]))

N = Counter(); inner = Counter(); start_d = defaultdict(list); sil_d = defaultdict(list); nosil = Counter(); nspeech = Counter()
multi = Counter()
for r, ss in segs.items():
    try:
        p = np.load(f"{sd}/{r}.npy").astype(np.float32)
    except FileNotFoundError:
        N["録音なし"] += len(ss); continue
    sp = p >= 0.5
    # 無音の区間 [s, e)（フレーム）
    edges = np.diff(np.r_[0, (~sp).astype(int), 0])
    s0, s1 = np.where(edges == 1)[0], np.where(edges == -1)[0]
    long_ = (s1 - s0) * H >= 0.4
    ls0, ls1 = s0[long_] * H, s1[long_] * H            # 0.4 秒以上の無音の頭と終わり（秒）
    allsil0 = s0 * H                                   # すべての無音の頭（スコアを出し始める所）
    # 音声区間の頭：0.4 秒以上の無音の終わり − 0.2 秒（録音の最初の音声も）
    on = np.where(np.diff(np.r_[0, sp.astype(int)]) == 1)[0]
    heads = np.r_[on[:1] * H, ls1] - 0.2
    for b, e, t in ss:
        N[t] += 1
        if not sp[int(b / H):int(e / H) + 1].any():
            nspeech[t] += 1; continue
        # 区間の中（頭から区間末の 0.1 秒前まで）にある 0.4 秒以上の無音（実時間ではここで判定する）
        k = np.sum((ls0 > b + 0.1) & (ls0 + 0.4 < e - 0.1))
        inner[t] += k > 0
        # 区間末の後で最初に 0.4 秒続く無音の頭（区間末の 0.3 秒前から 5 秒後まで）
        j = np.where((ls0 >= e - 0.3) & (ls0 <= e + 5))[0]
        if len(j) == 0:
            nosil[t] += 1; continue
        sil_d[t].append(ls0[j[0]] - e)
        # その時点の音声区間の頭（判定の時刻 = 無音の頭 + 0.4 秒より前の最後の頭）
        hh = heads[heads < ls0[j[0]]]
        h = hh[-1] if len(hh) else 0.0
        start_d[t].append(h - b)
        # 音声区間の中に入る書き起こしの区間（同じ話者）の数
        multi[t] += sum(1 for bb, ee, _ in ss if ee <= ls0[j[0]] + 0.05 and ee > h + 0.05 and ee < e - 0.05) > 0


def q(v):
    v = np.array(v)
    return f"中央値 {np.median(v):+.2f}  10% {np.percentile(v, 10):+.2f}  90% {np.percentile(v, 90):+.2f}"


for t in ["0", "1", "2"]:
    n = N[t]
    print(f"== ラベル {t}（{n} 区間）")
    print(f"  silero が音声を一度も出さない  {nspeech[t] / n:.1%}")
    print(f"  区間の中に 0.4 秒以上の無音がある  {inner[t] / n:.1%}")
    print(f"  区間末の後 5 秒以内に 0.4 秒の無音が来ない  {nosil[t] / n:.1%}")
    print(f"  無音の頭 − 区間末（秒） {q(sil_d[t])}   1 秒以上遅れる {np.mean(np.array(sil_d[t]) >= 1):.1%}")
    print(f"  音声区間の頭 − 区間の頭（秒） {q(start_d[t])}   0.5 秒以上前から {np.mean(np.array(start_d[t]) <= -0.5):.1%}  0.5 秒以上後から {np.mean(np.array(start_d[t]) >= 0.5):.1%}")
    print(f"  音声区間に同じ話者の前の区間が入る  {multi[t] / max(1, len(start_d[t])):.1%}")
print(N["録音なし"], "録音なし")
