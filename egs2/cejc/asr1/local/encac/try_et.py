"""公開の重みがそのまま動くかを確かめる：付属の中国語の例と，CEJC の区間をいくつか（2026-11-01）"""
import sys, glob, wave, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_model as E
m, r = E.load("/home/kobori/2026/work/easy_turn/ckpt/checkpoint.pt")
print("足りない重み:", len(r.missing_keys), r.missing_keys[:5]); print("余った重み:", len(r.unexpected_keys), r.unexpected_keys[:5])
print("タグのトークン:", [m.tokenizer(t, add_special_tokens=False)["input_ids"] for t in E.TAGS])


def read(p, b=0.0, e=None):
    w = wave.open(p); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if w.getnchannels() > 1: x = x.reshape(-1, w.getnchannels())[:, 0]
    if sr != 16000: return None
    return torch.from_numpy(x[int(b * sr): None if e is None else int(e * sr)].copy())


for p in sorted(glob.glob("/home/kobori/2026/work/easy_turn/samples/*_real_*.wav")):
    x = read(p)
    if x is None: continue
    f = E.logmel(x)
    out = E.generate(m, f)
    txt = out.split("<")[0]
    print(os.path.basename(p), "|", out, "|", np.round(E.tag_probs(m, [f], [txt]).numpy()[0], 3))

A = "/home/kobori/2025/espnet/egs2/cejc/asr1/data/dev_g"
wav = dict(l.split() for l in open(f"{A}/wav.scp")); tag = dict(l.split() for l in open(f"{A}/tag"))
text = dict(l.rstrip("\n").split(" ", 1) for l in open(f"{A}/text"))
segs = [l.split() for l in open(f"{A}/segments")][1000:1012]
for u, rec, b, e in segs:
    f = E.logmel(read(wav[rec], float(b), float(e)))
    out = E.generate(m, f)
    print(u, "正解", tag[u], text[u][:40], "|", out[:60], "|", np.round(E.tag_probs(m, [f], [out.split("<")[0]]).numpy()[0], 3))
