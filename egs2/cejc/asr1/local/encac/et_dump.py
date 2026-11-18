"""CEJC で学習し直した Easy Turn（train_et.py の best.pt）のダンプ（2026-11-01，silver13）。

Easy Turn は「書き起こし→タグ」を順に出すので，0.1 秒ごとに書き起こしから出し直すと重すぎる。そこで
  - 区切り（区間の途中の間の始まり，区間末）ごとに，そこまでの音声で書き起こしを 1 回出す（貪欲）
  - 区切り c から c + 3.1 秒まで（次の区切りが先ならそこまで）0.1 秒ごとに，区間の頭から t までの音声と
    c での書き起こしを与えてタグの確率を出す（et_model.tag_probs）．p_end＝<complete>，p_bc＝<backchannel>
  - それ以外の時刻（話している間・最初の区切りより前）は確率 0．区切り c + 3.1 秒より後は最後の値を続ける
応答の規則が使うのは無音が続いている間のスコアだけ（無音 0.05〜1.5 秒・長い無音 2.5〜3 秒）なので，評価の結果は変わらない。
区切りは make_prefix.py（ASR ＋ Namo）と同じ：silero の無音の始まりのうち，区間の頭から 0.1 秒より後・区間末の 0.3 秒より前．
時刻 t の出力は t までの音声だけで出る（先読みなし）．書き起こしを出す時間は入れていない。
環境変数 NOEND=1 のときは区間末（書き起こしの正解の時刻）を区切りに使わず，silero の無音の始まりだけを区切りにする（11/02）。
環境変数 AC=1 のときは音響だけのモデル（train_ac.py の best.pt．ac_model.py）で確率を出す．書き起こしは使わない（11/03）。
環境変数 NOTEXT=1 のときは書き起こしを出さず，空の書き起こしでタグの確率を出す（train_et.py --no_text で学習したもの用．11/02）。
  python et_dump.py <best.pt> dev_g out.tsv [区間の数]
"""
import sys, os, wave, time, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_model as E
ASR1 = os.environ.get("ASR1", "/home/kobori/2025/espnet/egs2/cejc/asr1")   # silver12 では ~/2026/espnet_tt/…
sys.path.insert(0, f"{ASR1}/local/turntaking")
import eot_eval as EV

ck, d, out = sys.argv[1:4]
lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
STEP, POST, HOLD, SR, BS, GB = 0.1, 5.0, 3.1, 16000, int(os.environ.get("BS", 48)), 32   # BS=1：詰め物で確率が変わるモデル用（2026-11-12）
AC = os.environ.get("AC") == "1"
if AC:
    import ac_model
    m = ac_model.load(ck)
else:
    m, r = E.load(ck)
seg, _, _ = EV.load_segments(f"{ASR1}/data/{d}")
utts = list(seg)[:lim] if lim else list(seg)
runs = EV.silence_runs(f"{ASR1}/tt_analysis/vad_silero/{d}", utts, seg, POST)
wavs = dict(l.split() for l in open(f"{ASR1}/data/{d}/wav.scp"))
cache = {}


def audio(rec, b, t):
    if rec not in cache:
        cache.clear()
        w = wave.open(wavs[rec]); cache[rec] = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = cache[rec][int(round(b * SR)): int(round((b + t) * SR))].astype(np.float32) / 32768
    return torch.from_numpy(x if len(x) >= 400 else np.pad(x, (0, 400 - len(x))))


@torch.no_grad()
def transcribe(feats):
    """貪欲に書き起こしを出す（まとめて）．タグより前の文字列を返す"""
    pre = E._prefix(m, feats)
    L = max(len(p) for p in pre)
    Em = torch.zeros(len(pre), L, pre[0].shape[1], device=pre[0].device, dtype=torch.bfloat16)
    A = torch.zeros(len(pre), L, dtype=torch.long, device=Em.device)
    for i, p in enumerate(pre):                       # 左詰めで足りない分を前に置く（生成のため）
        Em[i, L - len(p):] = p; A[i, L - len(p):] = 1
    o = m.llama_model.generate(inputs_embeds=Em, attention_mask=A, max_new_tokens=80, num_beams=1, do_sample=False,
                               repetition_penalty=m.repetition_penalty, eos_token_id=151643, pad_token_id=151643)
    return [s.split("<")[0] for s in m.tokenizer.batch_decode(o, skip_special_tokens=True)]


t0 = time.time()
with open(out, "w") as fo:
    fo.write("utt\ttrue\tpos\telapsed\tp_cont\tp_end\tp_bc\n")
    for g in range(0, len(utts), GB):
        grp = utts[g:g + GB]
        cuts = {}
        for u in grp:
            rec, b, e = seg[u]; t, run = runs[u]
            st = t[(run > 0) & (np.r_[0, run[:-1]] == 0)] - EV.SIL_HOP
            if os.environ.get("NOEND") == "1":
                cuts[u] = [float(s) for s in st[st > 0.1]]
            else:
                cuts[u] = [float(s) for s in st[(st > 0.1) & (st < e - b - 0.3)]] + [e - b]
        keys = [(u, c) for u in grp for c in cuts[u]]
        txt = {}
        if AC or os.environ.get("NOTEXT") == "1":
            txt = {key: "" for key in keys}
            keys = []
        for k in range(0, len(keys), BS):
            bt = keys[k:k + BS]
            for key, s in zip(bt, transcribe([E.logmel(audio(seg[u][0], seg[u][1], c)) for u, c in bt])):
                txt[key] = s
        jobs = []                                     # (u, 行番号, t, 区切り)
        rows = {}
        for u in grp:
            rec, b, e = seg[u]
            ts = np.arange(1, int(round((e + POST - b) / STEP)) + 1) * STEP
            rows[u] = np.zeros((len(ts), 2))
            cs = cuts[u]
            for j, t in enumerate(ts):
                k = max([i for i, c in enumerate(cs) if c <= t + 1e-9], default=-1)
                if k < 0:
                    continue
                nxt = cs[k + 1] if k + 1 < len(cs) else np.inf
                if t <= min(cs[k] + HOLD, nxt) + 1e-9:
                    jobs.append((u, j, t, cs[k]))
        for k in range(0, len(jobs), BS):
            bt = jobs[k:k + BS]
            fs = [(ac_model.feat if AC else E.logmel)(audio(seg[u][0], seg[u][1], t)) for u, _, t, _ in bt]
            p = (ac_model.probs(m, fs) if AC else E.tag_probs(m, fs, [txt[(u, c)] for u, _, _, c in bt])).numpy()
            for (u, j, _, _), pp in zip(bt, p):
                rows[u][j] = pp[0], pp[2]
        for u in grp:
            rec, b, e = seg[u]
            R = rows[u]; last = (0.0, 0.0)
            ts = np.arange(1, len(R) + 1) * STEP
            for j, t in enumerate(ts):
                if R[j].any():
                    last = tuple(R[j])
                elif t > (e - b) + HOLD:                  # 区間末 + 3.1 秒より後は最後の値を続ける
                    R[j] = last
                fo.write(f"{u}\t-1\t{j}\t{t:.4f}\t{1 - R[j][0] - R[j][1]:.6f}\t{R[j][0]:.6f}\t{R[j][1]:.6f}\n")
        print(f"{g + len(grp)}/{len(utts)} 区間  書き起こし {len(keys)}・タグ {len(jobs)}  {time.time() - t0:.0f} 秒", flush=True)
print("ok")
