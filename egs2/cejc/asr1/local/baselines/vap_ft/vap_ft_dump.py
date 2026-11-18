"""CEJC で学習し直した VAP（train_vap.py の best.pth．2026-10-29）で，各区間の頭から区間末 +5 秒までの「相手（システム）に交替する確率」を出す（2026-10-21）。

maai の実時間処理（maai/model.py の Maai，KV キャッシュなし）をそのままなぞる：
  0.1 秒ごとに「直前 320 サンプル＋新しい 1600 サンプル」を符号化し（CPC の GRU の状態は持ち越す），
  埋め込みの直近 200 フレームを VAP に通して最後の時点の出力を取る。録音（チャネル）をまとめて一括で流す。
ch1＝判定対象の話者のマイク，ch2＝無音（システムはまだ話していない．提案法・silero と同じく本人の音声だけを使う）。
スコアは p_now の ch2（この先 0〜0.6 秒に話しているのが ch2 である確率）と，p_future の ch2（0.6〜2.0 秒）の 2 つ。
  out には p_now を，out の .tsv を _fut.tsv にしたファイルに p_future を書く。k 回目の出力は 0.02 + 0.1×(k+1) 秒までの音声で出る。
出力は eot_eval.py の --dump 形式（utt true pos elapsed p_cont p_end p_bc）．p_end＝スコア，p_bc＝0（相槌の確率は無い）。

中身は ../baselines_20261021/vap_dump2.py と同じで，重みだけ best.pth にする。
  python vap_ft_dump.py data/eval_g tt_preds/vap_ft_eval_g_post5.0.tsv
"""
import sys, glob, wave, collections, time
import numpy as np, torch
import types
pkg = types.ModuleType("maai"); pkg.__path__ = ["/home/kobori/tt_base/maai_src/maai"]; sys.modules["maai"] = pkg   # maai/__init__.py（マイク入力）を通さない
from maai.models.config import VapConfig
from maai.models.vap import VapGPT

POST, STEP, PAD, CTX, SR = 5.0, 1600, 320, 200, 16000
data, out = sys.argv[1], sys.argv[2]
dev = "cuda"
p = "/home/kobori/tt_base/vap_jp/vap_state_dict_jp_10hz_20000msec.pt"
sd = torch.load(p, map_location="cpu")
c = VapConfig(); c.frame_hz = 10.0; c.context_limit = CTX; c.load_pretrained = 1; c.encoder_type = "cpc"
m = VapGPT(c); m.load_encoder(cpc_model="/home/kobori/.cache/cpc/60k_epoch4-d0f474de.pt")
r = m.load_state_dict(sd, strict=False)
for e in (m.encoder1, m.encoder2):          # downsample は VAP の重み（maai/model.py と同じ）
    e.downsample[1].weight = torch.nn.Parameter(sd["encoder.downsample.1.weight"]); e.downsample[1].bias = torch.nn.Parameter(sd["encoder.downsample.1.bias"])
    e.downsample[2].ln.weight = torch.nn.Parameter(sd["encoder.downsample.2.ln.weight"]); e.downsample[2].ln.bias = torch.nn.Parameter(sd["encoder.downsample.2.ln.bias"])
bad = [k for k in r.missing_keys if not k.startswith(("encoder1.encoder.", "encoder2.encoder.", "encoder1.downsample.", "encoder2.downsample."))]
assert not bad, bad
m.load_state_dict(torch.load("/home/kobori/2025/espnet/egs2/cejc/asr1/exp/vap_ft_20261029/best.pth", map_location="cpu"))   # 学習し直した重み
m = m.to(dev).eval()


def p_now_ch2(x1, x2):
    """VapGPT.forward と同じ計算で，バッチ全体の最後の時点の p_now・p_future（ch2）を返す。"""
    o1 = m.ar_channel(x1); o2 = m.ar_channel(x2)
    out = m.ar(o1["x"], o2["x"])
    probs = m.vap_head(out["x"][:, -1:]).softmax(-1)
    pn = m.objective.probs_next_speaker_aggregate(probs, from_bin=m.BINS_P_NOW[0], to_bin=m.BINS_P_NOW[-1])[:, 0, 1]
    pf = m.objective.probs_next_speaker_aggregate(probs, from_bin=m.BINS_PFUTURE[0], to_bin=m.BINS_PFUTURE[1])[:, 0, 1]
    return pn, pf


def read(path):
    w = wave.open(path); assert w.getframerate() == SR and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


wavs = dict(l.split() for l in open(f"{data}/wav.scp"))
segs = collections.defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); segs[rec].append((u, float(b), float(e)))
recs = sorted(segs, key=lambda r: wave.open(wavs[r]).getnframes())
B = 24                                       # 長さの近いチャネルをまとめて流す
t0 = time.time()
f = open(out, "w"); ff = open(out.replace(".tsv", "_fut.tsv"), "w")
for fo in (f, ff):
    fo.write("utt\ttrue\tpos\telapsed\tp_cont\tp_end\tp_bc\n")
for g0 in range(0, len(recs), B):
    grp = recs[g0:g0 + B]
    X = [read(wavs[r]) for r in grp]
    n = (max(len(x) for x in X) - PAD) // STEP
    A = np.zeros((len(X), PAD + n * STEP), np.float32)
    for i, x in enumerate(X):
        A[i, :min(len(x), A.shape[1])] = x[:A.shape[1]]
    A = torch.from_numpy(A).to(dev)
    Z = torch.zeros(len(X), 1, PAD + STEP, device=dev)
    P = np.zeros((len(X), n), np.float32); PF = np.zeros((len(X), n), np.float32)
    for e in (m.encoder1, m.encoder2):
        e.reset_streaming_state()
    E1 = E2 = None
    with torch.no_grad():
        for k in range(n):
            ch = A[:, None, k * STEP:k * STEP + PAD + STEP]
            e1, e2 = m.encode_audio(ch, Z)
            E1 = e1 if E1 is None else torch.cat([E1, e1], 1)[:, -CTX:]
            E2 = e2 if E2 is None else torch.cat([E2, e2], 1)[:, -CTX:]
            pn, pf = p_now_ch2(E1, E2)
            P[:, k] = pn.float().cpu().numpy(); PF[:, k] = pf.float().cpu().numpy()
    tt = (PAD + STEP * (np.arange(n) + 1)) / SR
    for i, r in enumerate(grp):
        for u, b, e in segs[r]:
            k = np.where((tt > b) & (tt <= e + POST))[0]
            for j, kk in enumerate(k):
                f.write(f"{u}\t-1\t{j}\t{tt[kk] - b:.4f}\t{1 - P[i, kk]:.6f}\t{P[i, kk]:.6f}\t0\n")
                ff.write(f"{u}\t-1\t{j}\t{tt[kk] - b:.4f}\t{1 - PF[i, kk]:.6f}\t{PF[i, kk]:.6f}\t0\n")
    print(f"{g0 + len(grp)}/{len(recs)} チャネル（{n / 600:.1f} 分ぶん）{time.time() - t0:.0f} 秒", flush=True)
    del A
f.close(); ff.close()
print(f"ok {time.time() - t0:.0f} 秒")
