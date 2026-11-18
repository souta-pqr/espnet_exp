"""Easy Turn（ASLP-lab）を CEJC で学習し直す（公開の重みから続けて学習．2026-11-01，silver13）。

従来手法との比較を公平にするため，Smart Turn の学び直し（smartturn_ft_20261028/train_st.py）と同じデータ・ラベルを使う。
  - データ：学習 data/train_gx，検証 data/dev_g（提案法と同じ）
  - ラベル：gemma のラベル（data/*/tag：0=継続，1=完了，2=相槌）→ <incomplete>・<complete>・<backchannel>（<wait> は使わない）
  - 出力の文字列：書き起こし＋タグ（Easy Turn の形のまま）．書き起こしは data/*/text から (F …)・(D …) の印と空白を除き，
    <mask>・<unclear> は「＊」にしたもの
  - 読み出し位置：区間末の 0〜+2.0 秒から一様に 1 つ．提案法・Smart Turn は −0.6 秒からだが，区間末より前では
    その時点までの書き起こしが分からない（単語の時刻が無い）ので，区間末より後だけにした
  - 入力：区間の頭から読み出し位置までの本人の音声（Easy Turn の使い方どおり発話全体．区間は最長 28 秒なので切らない）
学習の設定は Easy Turn の conf/train.yaml に合わせる（AdamW lr 5e-5・betas (0.9, 0.99)・weight decay 0.01・warmup・
勾配のクリップ 5）．プロンプトは公式と同じく 5 つから毎回ランダム．メモリのため全体を学習しつつ bf16 で動かす。
検証は区間ごとに位置を固定（seed 0）した dev_g の 4000 区間で，損失とタグの正解率（正しい書き起こしを与えたとき）を見て，
損失が最小の重みを残す。
  python train_et.py <出力ディレクトリ> [--epochs 2] [--max_steps N]
--no_text（11/02）：書き起こしを出さずにタグだけを出すように学習する（Easy Turn の論文の「Easy Turn-only-state」に当たるアブレーション）
"""
import sys, os, re, math, time, wave, random, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_model as E

ASR1 = "/home/kobori/2025/espnet/egs2/cejc/asr1"
SR, POST = 16000, 2.0
TAG = {"0": "<incomplete>", "1": "<complete>", "2": "<backchannel>"}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--acc", type=int, default=4)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--nval", type=int, default=4000)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--no_text", action="store_true", help="書き起こしを出さずにタグだけを出す")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0); random.seed(0)


def norm(t):
    t = re.sub(r"<(mask|unclear)>", "＊", t)
    t = re.sub(r"\([A-Z?]+ ?", "", t).replace(")", "")
    return t.replace(" ", "")


def load(d):
    wav = dict(l.split() for l in open(f"{ASR1}/data/{d}/wav.scp"))
    tag = dict(l.split() for l in open(f"{ASR1}/data/{d}/tag"))
    txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(f"{ASR1}/data/{d}/text"))
    return [(wav[rec], float(b), float(e), norm(txt[u]), TAG[tag[u]]) for u, rec, b, e in (l.split() for l in open(f"{ASR1}/data/{d}/segments"))]


class DS(torch.utils.data.Dataset):
    def __init__(self, items, fixed):
        self.items = items
        self.off = np.random.default_rng(0).uniform(0, POST, len(items)) if fixed else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, b, e, txt, tag = self.items[i]
        off = self.off[i] if self.off is not None else np.random.uniform(0, POST)
        w = wave.open(path)
        s0 = int(round(b * SR)); s1 = min(int(round((e + off) * SR)), w.getnframes())
        w.setpos(s0)
        x = np.frombuffer(w.readframes(max(s1 - s0, 400)), dtype=np.int16).astype(np.float32) / 32768
        prompt = random.choice(E.PROMPTS[E.TASK]) if self.off is None else E.PROMPT
        return E.logmel(torch.from_numpy(x.copy())), txt, tag, prompt


def collate(b):
    return list(zip(*b))


m, r = E.load("/home/kobori/2026/work/easy_turn/ckpt/checkpoint.pt")
assert not r.missing_keys and not r.unexpected_keys
m.llama_model.float()                        # 読み込みは bf16．学習は float32 の重み＋bf16 の autocast（更新が丸められないように）
m.llama_model.gradient_checkpointing_enable()
m.encoder.gradient_checkpointing = True
m.train()
tok = m.tokenizer
EOS = 151643


def loss_batch(feats, txts, tags, prompts):
    """書き起こし＋タグ＋<|endoftext|> の交差エントロピー（Easy Turn の forward と同じく，プロンプト・音声は損失に入れない）"""
    dev = next(m.parameters()).device
    lens = torch.tensor([len(f) for f in feats], device=dev)
    x = torch.nn.utils.rnn.pad_sequence(list(feats), batch_first=True).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sp, mask = m.get_embedding_from_wav(x, lens)
        sp, mask, _ = m._add_bos_eos(m.speech_token_num, 1 + m.speech_token_num, sp, mask, None)
    sp = sp.to(m.embed_tokens.weight.dtype)
    seqs, tg = [], []
    for i in range(len(feats)):
        pe = m.embed_tokens(tok([prompts[i]], return_tensors="pt")["input_ids"].to(dev))[0]
        ids = tok(("" if args.no_text else txts[i]) + tags[i], add_special_tokens=False)["input_ids"] + [EOS]
        ids = torch.tensor(ids, device=dev)
        s = torch.cat([pe, sp[i][mask[i]], m.embed_tokens(ids)], 0)
        t = torch.full((len(s),), -100, device=dev); t[-len(ids):] = ids
        seqs.append(s); tg.append(t)
    L = max(len(s) for s in seqs)
    Em = torch.zeros(len(seqs), L, seqs[0].shape[1], device=dev, dtype=seqs[0].dtype)
    A = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    T = torch.full((len(seqs), L), -100, device=dev)
    for i, (s, t) in enumerate(zip(seqs, tg)):
        Em[i, :len(s)] = s; A[i, :len(s)] = 1; T[i, :len(s)] = t
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m.llama_model(inputs_embeds=Em, attention_mask=A, labels=T).loss


@torch.no_grad()
def validate(dl):
    torch.cuda.empty_cache()
    m.eval(); ls, ok, n = [], 0, 0
    for feats, txts, tags, prompts in dl:
        ls.append(loss_batch(feats, txts, tags, prompts).item())
        p = E.tag_probs(m, list(feats), [""] * len(txts) if args.no_text else list(txts)).argmax(1)
        ok += sum(E.TAGS[int(k)] == t for k, t in zip(p, tags)); n += len(tags)
    m.train()
    return float(np.mean(ls)), ok / n


tr = load("train_gx")
va = load("dev_g")
va = [va[i] for i in np.random.default_rng(0).permutation(len(va))[:args.nval]]
dl = torch.utils.data.DataLoader(DS(tr, False), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                 collate_fn=collate, drop_last=True, persistent_workers=True)
dv = torch.utils.data.DataLoader(DS(va, True), batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate)
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.01)
total = args.max_steps or args.epochs * len(dl) // args.acc
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / args.warmup, max(0.0, (total - s) / max(1, total - args.warmup))))
print(f"学習 {len(tr)} 区間・検証 {len(va)} 区間・更新 {total} 回・学習するパラメータ {sum(p.numel() for p in params) / 1e6:.0f}M", flush=True)
v0 = validate(dv)
print(f"[公開の重み] 検証の損失 {v0[0]:.4f}  タグの正解率 {v0[1]:.4f}", flush=True)
best, step, t0 = math.inf, 0, time.time()
for ep in range(1, args.epochs + 1):
    run = []
    for k, (feats, txts, tags, prompts) in enumerate(dl):
        loss = loss_batch(feats, txts, tags, prompts)
        (loss / args.acc).backward(); run.append(loss.item())
        if (k + 1) % args.acc:
            continue
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 100 == 0:
            print(f"エポック {ep} 更新 {step}/{total}  損失 {np.mean(run[-100 * args.acc:]):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:.0f} 秒  GPU メモリ最大 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if args.max_steps and step >= args.max_steps:
            break
        if step % 2000 == 0 or (k + 1) // args.acc == len(dl) // args.acc:
            vl, va_acc = validate(dv)
            mark = ""
            if vl < best:
                best = vl; mark = "  ← 最良"
                torch.save({k: v for k, v in m.state_dict().items()}, f"{args.out}/best.pt")
            print(f"[エポック {ep} 更新 {step}] 検証の損失 {vl:.4f}  タグの正解率 {va_acc:.4f}{mark}", flush=True)
    if args.max_steps and step >= args.max_steps:
        break
print(f"完了．最良の検証の損失 {best:.4f}  {time.time() - t0:.0f} 秒", flush=True)
