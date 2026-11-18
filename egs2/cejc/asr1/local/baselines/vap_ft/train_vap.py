"""VAP（maai-kyoto/vap_jp，10 Hz・文脈 20 秒）を CEJC で学習し直す（公開の重みから続けて学習．2026-10-29，silver11）。

従来手法との比較を公平にするため，提案法と同じデータ（学習 data/train_gx，検証 data/dev_g）で学ぶ。
使い方は評価（../baselines_20261021/vap_dump2.py）と同じにする：
  - ch1＝本人のマイク，ch2＝無音（システムはまだ話していない．提案法と同じく本人の音声だけを使う）
  - 符号化は maai の実時間処理と同じ：0.1 秒ごとに「直前 320 サンプル＋新しい 1600 サンプル」を CPC の畳み込みに通し，
    GRU の状態は持ち越し，ダウンサンプル（kernel 5・stride 2）はまとまりごとにかける（1 まとまり 3 フレーム）。
    ここではそれをまとめて計算する（逐次に回したものと同じになることを --check で確かめる）
  - 0.1 秒ごとの出力は，そのまとまりの最後のフレーム．文脈は直近 200 フレーム（ALiBi の窓）
正解：VAP の目的関数（この先 2 秒を 0.2・0.4・0.6・0.8 秒の区切りで，各話者が話しているか．256 通り）．
  0.1 秒ごとの各区切りで，ch1＝本人が話しているか（本人の区間），ch2＝同じ会話の他の参加者の誰かが話しているか（その区間）。
  時刻 t_k（k 番目の出力が見た音声の終わり）から先の 2 秒を見る。
学習するのは Transformer（各チャネル 1 層・2 チャネルをまたぐ 3 層）と出力層．エンコーダ（CPC・ダウンサンプル）は凍結。
20 秒の窓を切り出して流し，GRU と文脈がそろう最初の 7 秒は損失に入れない．検証の損失が最小の重みを残す。

  python train_vap.py <出力ディレクトリ> [--check] [--max_steps N]
"""
import sys, os, types, math, time, wave, argparse, collections
import numpy as np, torch, torch.nn.functional as F
pkg = types.ModuleType("maai"); pkg.__path__ = ["/home/kobori/tt_base/maai_src/maai"]; sys.modules["maai"] = pkg   # maai/__init__.py（マイク入力）を通さない
from maai.models.config import VapConfig
from maai.models.vap import VapGPT

ASR1 = "/home/kobori/2025/espnet/egs2/cejc/asr1"
SR, STEP, PAD, CTX, HOR = 16000, 1600, 320, 200, 20       # 0.1 秒・先頭の重なり・文脈のフレーム数・先の 2 秒（ステップ数）
NWIN, WARM_STEPS = 200, 70                                 # 窓 20 秒（200 ステップ）．最初の 7 秒は損失に入れない
LR, EPOCHS, BS, WD = 1e-4, 5, 16, 0.01

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--check", action="store_true")
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cuda")

# ---- モデル（vap_dump2.py と同じ読み込み） ----
sd = torch.load("/home/kobori/tt_base/vap_jp/vap_state_dict_jp_10hz_20000msec.pt", map_location="cpu")
c = VapConfig(); c.frame_hz = 10.0; c.context_limit = CTX; c.load_pretrained = 1; c.encoder_type = "cpc"
m = VapGPT(c); m.load_encoder(cpc_model="/home/kobori/.cache/cpc/60k_epoch4-d0f474de.pt")
r = m.load_state_dict(sd, strict=False)
for e in (m.encoder1, m.encoder2):
    e.downsample[1].weight = torch.nn.Parameter(sd["encoder.downsample.1.weight"]); e.downsample[1].bias = torch.nn.Parameter(sd["encoder.downsample.1.bias"])
    e.downsample[2].ln.weight = torch.nn.Parameter(sd["encoder.downsample.2.ln.weight"]); e.downsample[2].ln.bias = torch.nn.Parameter(sd["encoder.downsample.2.ln.bias"])
bad = [k for k in r.missing_keys if not k.startswith(("encoder1.encoder.", "encoder2.encoder.", "encoder1.downsample.", "encoder2.downsample."))]
assert not bad, bad
m = m.to(dev)
for e in (m.encoder1, m.encoder2):
    e.eval()
    for p in e.parameters():
        p.requires_grad_(False)


def encode(enc, a):
    """maai の実時間処理と同じ符号化をまとめて行う。a：(B, PAD + STEP × N) → (B, 3N, 256)．"""
    B = a.shape[0]
    ch = a.unfold(1, PAD + STEP, STEP)                        # (B, N, 1920)
    N = ch.shape[1]
    z = enc.encoder.gEncoder(ch.reshape(B * N, 1, -1))         # (BN, 256, 12)
    z = z[:, :, 1:-1].reshape(B, N, 256, -1).permute(0, 1, 3, 2).reshape(B, -1, 256)   # (B, 10N, 256)
    enc.encoder.gAR.hidden = None
    z = enc.encoder.gAR(z)                                     # GRU は続けて流す（状態の持ち越しと同じ）
    enc.encoder.gAR.hidden = None
    z = enc.downsample(z.reshape(B * N, -1, 256))              # まとまりごと：(BN, 3, 256)
    return z.reshape(B, -1, 256)


def logits_steps(a):
    """(B, PAD + STEP × N) → 0.1 秒ごとの VAP のロジット (B, N, 256)．"""
    with torch.no_grad():
        x1 = encode(m.encoder1, a)
        x2 = encode(m.encoder2, torch.zeros_like(a))
    o1 = m.ar_channel(x1); o2 = m.ar_channel(x2)
    out = m.ar(o1["x"], o2["x"])
    k = x1.shape[1] // a.unfold(1, PAD + STEP, STEP).shape[1]
    return m.vap_head(out["x"][:, k - 1::k])                  # 各まとまりの最後のフレーム


if args.check:                                                 # 逐次（vap_dump2.py と同じ）とまとめての一致を確かめる
    m.eval()
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(2, PAD + STEP * 80, generator=g) * 0.1).to(dev)
    with torch.no_grad():
        L = logits_steps(a)
        for e in (m.encoder1, m.encoder2):
            e.reset_streaming_state()
        E1 = E2 = None; Z = torch.zeros(2, 1, PAD + STEP, device=dev); seq = []
        for k in range(80):
            e1, e2 = m.encode_audio(a[:, None, k * STEP:k * STEP + PAD + STEP], Z)
            E1 = e1 if E1 is None else torch.cat([E1, e1], 1)[:, -CTX:]
            E2 = e2 if E2 is None else torch.cat([E2, e2], 1)[:, -CTX:]
            o1 = m.ar_channel(E1); o2 = m.ar_channel(E2)
            seq.append(m.vap_head(m.ar(o1["x"], o2["x"])["x"][:, -1]))
        S = torch.stack(seq, 1)
    d = (L.softmax(-1) - S.softmax(-1)).abs()
    print(f"逐次との差：確率の最大 {d.max().item():.2e}（ステップ 0〜69 {d[:, :70].max().item():.2e}，70〜 {d[:, 70:].max().item():.2e}）", flush=True)
    sys.exit(0)


# ---- データ ----
def load(d):
    wav = dict(l.split() for l in open(f"{ASR1}/data/{d}/wav.scp"))
    segs = collections.defaultdict(list)
    for l in open(f"{ASR1}/data/{d}/segments"):
        u, rec, b, e = l.split(); segs[rec].append((float(b), float(e)))
    conv = collections.defaultdict(list)
    for rec in wav:
        conv[rec.rsplit("_", 1)[0]].append(rec)
    items, va = [], {}
    for rec, path in sorted(wav.items()):
        w = wave.open(path); n = w.getnframes(); w.close()
        ns = (n - PAD) // STEP                                 # このチャネルのステップ数
        t = PAD / SR + STEP / SR * (np.arange(ns) + 0.5)       # 各ステップの区切りの真ん中の時刻
        def act(recs):
            v = np.zeros(ns, np.float32)
            for r2 in recs:
                for b, e in segs.get(r2, []):
                    v[(t >= b) & (t < e)] = 1
            return v
        va[rec] = np.stack([act([rec]), act([r for r in conv[rec.rsplit("_", 1)[0]] if r != rec])], 1)
        items.append((rec, path, ns))
    return items, va


class DS(torch.utils.data.Dataset):
    def __init__(self, items, va, fixed):
        self.va, self.fixed = va, fixed
        span = NWIN + HOR
        self.win = []                                          # (rec, path, 先頭のステップ)
        for rec, path, ns in items:
            for s in range(0, ns - span, NWIN):
                self.win.append((rec, path, s, ns))

    def __len__(self):
        return len(self.win)

    def __getitem__(self, i):
        rec, path, s, ns = self.win[i]
        if not self.fixed:                                     # 学習は位置をずらす
            s = int(np.random.randint(0, max(1, ns - NWIN - HOR)))
        w = wave.open(path); w.setpos(s * STEP)
        x = np.frombuffer(w.readframes(PAD + STEP * NWIN), dtype=np.int16).astype(np.float32) / 32768.0
        w.close()
        return torch.from_numpy(x), torch.from_numpy(self.va[rec][s:s + NWIN + HOR])


def loader(ds, shuffle):
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle, num_workers=args.workers, pin_memory=True,
                                       drop_last=shuffle, persistent_workers=True)


def step_loss(a, va):
    lg = logits_steps(a)                                       # (B, N, 256)
    # 正解：ステップ k の先の 2 秒＝区切り k+1〜k+20（ObjectiveVAP.get_labels はフレーム i に va[i+1:] の窓を使う）
    lab = m.objective.get_labels(va)                           # (B, N, ...)
    lab = lab[:, :lg.shape[1]]
    return F.cross_entropy(lg[:, WARM_STEPS:].reshape(-1, lg.shape[-1]).float(), lab[:, WARM_STEPS:].reshape(-1))


tr_items, tr_va = load("train_gx"); dv_items, dv_va = load("dev_g")
ds_tr, ds_dv = DS(tr_items, tr_va, False), DS(dv_items, dv_va, True)
print(f"学習 {len(tr_items)} チャネル {sum(n for *_, n in tr_items) / 36000:.0f} 時間（1 エポック {len(ds_tr)} 窓）．"
      f"検証 {len(dv_items)} チャネル {len(ds_dv)} 窓．本人が話している割合 {np.mean(np.concatenate([v[:, 0] for v in tr_va.values()])):.3f}・"
      f"他の参加者 {np.mean(np.concatenate([v[:, 1] for v in tr_va.values()])):.3f}", flush=True)
dl_tr, dl_dv = loader(ds_tr, True), loader(ds_dv, False)
params = [p for p in m.parameters() if p.requires_grad]
print(f"学習するパラメータ {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
total = args.max_steps or EPOCHS * len(dl_tr)
warm = int(0.05 * total)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def evaluate():
    m.eval(); tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a, va in dl_dv:
            l = step_loss(a.to(dev, non_blocking=True), va.to(dev, non_blocking=True))
            tot += l.item() * len(a); n += len(a)
    m.train()
    for e in (m.encoder1, m.encoder2):
        e.eval()
    return tot / n


if not args.max_steps:
    best = evaluate()
    print(f"[学習前（公開の重み）] 検証の損失 {best:.4f}", flush=True)
    torch.save({k: v for k, v in m.state_dict().items()}, f"{args.out}/best.pth")
else:
    best = math.inf
m.train()
for e in (m.encoder1, m.encoder2):
    e.eval()
step, t0, run, done = 0, time.time(), [], False
for ep in range(EPOCHS):
    for a, va in dl_tr:
        a, va = a.to(dev, non_blocking=True), va.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = step_loss(a, va)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run.append(loss.item()); step += 1
        if step % 100 == 0 or (args.max_steps and step % 10 == 0):
            print(f"エポック {ep + 1} 更新 {step}/{total}  損失 {np.mean(run[-300:]):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:.0f} 秒  GPU メモリ最大 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if step >= total:
            done = True; break
    if not args.max_steps:
        v = evaluate(); mark = ""
        if v < best:
            best = v; torch.save(m.state_dict(), f"{args.out}/best.pth"); mark = "  ← 最良"
        print(f"[エポック {ep + 1}] 検証の損失 {v:.4f}{mark}", flush=True)
    if done:
        break
torch.save(m.state_dict(), f"{args.out}/last.pth")
print(f"完了．最良の検証の損失 {best:.4f}  {time.time() - t0:.0f} 秒", flush=True)
