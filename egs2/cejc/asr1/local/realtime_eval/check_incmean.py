"""ボットの設計（値が確定したフレームだけ漸化式の平均 m に入れ，末尾は出力のたびに足すだけ）が，今の h.mean(0) と一致するかを確かめる（11/27）"""
import sys, wave, numpy as np, torch
sys.path.insert(0, ".")
from epd.models.encac import load_model
torch.set_num_threads(4)
m = load_model(sys.argv[1], "cpu")
E = m.encoder; BS, HOP, LA = E.block_size, E.hop_size, E.look_ahead

def conf(T):
    """長さ T のとき，後から値が変わらないフレームの数（ESPnet の塊の区切り方：最後の塊の出力だけが変わる）"""
    if T <= BS:
        return 0
    cur, left = BS - LA, HOP
    while left + BS < T:
        cur += HOP; left += HOP
    return cur

def enc(a):
    t = torch.from_numpy(a)[None]
    f, fl = m.frontend(t, torch.tensor([t.size(1)])); f, fl = m.normalize(f, fl)
    h, hl, _ = E(f, fl)
    return h[0, :hl[0]]

dv = dp = dfr = 0.0; steps = 0
with torch.no_grad():
    for rec, off in (("K008/K008_001/K008_001a_IC01", 20), ("C001/C001_001/C001_001_IC01", 60), ("K008/K008_001/K008_001a_IC02", 100)):
        w = wave.open(f"/autofs/diamond2/share/corpus/CEJC_safia/data/{rec}.wav"); w.setpos(16000 * off)
        x = np.frombuffer(w.readframes(16000 * 14), dtype=np.int16).astype(np.float32) / 32768
        for sp in (0.5, 1.0, 1.3, 2.0, 5.0, 9.0):            # 音声区間の頭から無音に入るまでの長さ（秒）
            n, mm, keep = 0, None, None                        # 状態：確定したフレーム数，その平均，（確かめ用）確定したフレームの値
            cut = int(sp * 16000)
            for k in range(1, 32):                             # 無音 0.1〜3.1 秒
                h = enc(x[:cut + 1600 * k]); T = len(h); c = conf(T)
                if keep is not None:                           # 前に確定したフレームが変わっていないか
                    dfr = max(dfr, float((h[:n] - keep).abs().max()))
                if c > n:
                    new = h[n:c]
                    mm = new.mean(0) if n == 0 else mm + (new - mm).sum(0) / c   # m += Σ(x − m)/k（k は更新後の枚数）
                    n = c; keep = h[:n].clone()
                v = (h.sum(0) if n == 0 else n * mm + h[n:].sum(0)) / T          # 末尾は足すだけ（保存しない）
                v0 = h.mean(0)
                dv = max(dv, float((v - v0).abs().max()))
                dp = max(dp, float((m.head(v).softmax(-1) - m.head(v0).softmax(-1)).abs().max()))
                steps += 1
print(f"block_size {BS}・hop_size {HOP}・look_ahead {LA}．試した回数 {steps}（録音 3 つ × 音声の長さ 6 通り × 無音 0.1〜3.1 秒の 31 回）")
print(f"漸化式の平均と h.mean(0) の差の最大 {dv:.2e}．確率の差の最大 {dp:.2e}．確定したフレームが後で変わった量の最大 {dfr:.2e}")
