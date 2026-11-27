"""Easy Turn のアブレーション 2：音響だけ（エンコーダ＋平均＋線形層）を CEJC で学習する（2026-11-03，silver13）。
データ・ラベル・読み出し位置（区間末の 0〜+2.0 秒から一様に 1 つ）・入力（区間の頭から読み出し位置まで）・最適化の設定は
train_et.py と同じ（線形層だけ lr 1e-3）。検証も同じ dev_g の 4000 区間で，損失が最小の重みを残す。
  python train_ac.py <出力ディレクトリ> [--epochs 2]
"""
import sys, os, math, time, random, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_model as E
import ac_model as A
from train_et_data import load, DS, collate
import train_et_data
train_et_data.FEAT = A.feat     # 11/04：エンコーダごとの入力（enc_plug.py）

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--acc", type=int, default=4)
ap.add_argument("--freeze_enc", action="store_true", help="エンコーダを固めて線形層だけ学習する（BatchNorm の統計も固める．11/05）")
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--nval", type=int, default=4000)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--train", default="train_gx", help="学習データ（train_gxac：前の区間とつないだ音声も足したもの．11/11）")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0); random.seed(0)
CLS = {"<incomplete>": 0, "<complete>": 1, "<backchannel>": 2}

m = A.load_pretrained().cuda()
m.encoder.gradient_checkpointing = True
m.train()
if args.freeze_enc:
    for p_ in m.encoder.parameters():
        p_.requires_grad_(False)
    _train = m.train
    def _train_fz(mode=True):     # validate() の m.train() の後も，エンコーダは eval のまま
        _train(mode); m.encoder.eval(); return m
    m.train = _train_fz
    m.train()


HARDW = train_et_data.HARDW
NEXT = train_et_data.NEXT        # 11/15：追加の課題（同じ話者の次の声までの時間を当てる．Next-Turn, arXiv 2606.18094 にならう）．推論では使わないので重みには残さない
if NEXT:
    aux = torch.nn.Linear(m.head.in_features, 1).cuda(); pooled = {}
    m.head.register_forward_hook(lambda mod, i, o: pooled.__setitem__("x", i[0]))


CTCW = train_et_data.CTCW        # 11/16：追加の課題（区間の文字を CTC で当てる．文字の手がかりを contextual block conformer エンコーダに残す）．推論では使わないので重みには残さない
if CTCW:
    import yaml, enc_plug
    TOK = {t: i for i, t in enumerate(yaml.safe_load(open(enc_plug.GTCONF))["token_list"])}
    aux = torch.nn.Linear(m.head.in_features, len(TOK)).cuda(); enc_out = {}
    m.encoder.register_forward_hook(lambda mod, i, o: enc_out.__setitem__("x", o))


def ctc_loss(txts, segs):
    """区間の終わりまでのフレームで，その区間の文字を当てる（区間の後の無音や次の声は入れない）"""
    h, hl = enc_out["x"][0], enc_out["x"][1]
    use = [i for i, (t, s) in enumerate(zip(txts, segs)) if s > 0 and 0 < len(t) <= int(s * 16000 / 528)]
    if not use:
        return h.sum() * 0.0
    n = torch.tensor([min(int(hl[i]), int(segs[i] * 16000 / 528) + 1) for i in use])
    y = [torch.tensor([TOK.get(c, 1) for c in txts[i]]) for i in use]
    lp = aux(h[use]).log_softmax(-1).transpose(0, 1)
    return torch.nn.functional.ctc_loss(lp, torch.cat(y), n, torch.tensor([len(v) for v in y]), blank=0, zero_infinity=True)


def loss_batch(feats, tags, rem=None, txts=None):
    y = torch.tensor([CLS[t] for t in tags], device="cuda")
    lg = m(feats)
    l = torch.nn.functional.cross_entropy(lg, y)
    if rem is not None and HARDW:    # 間が 1 秒以上ある「まだ続く」区間を重くする（実時間で割り込みになるのはこの区間）
        w = torch.where((y == 0) & (torch.tensor(rem, device="cuda") >= 1.0), HARDW, 1.0)
        l = (torch.nn.functional.cross_entropy(lg, y, reduction="none") * w).sum() / w.sum()
    elif rem is not None and CTCW:
        l = l + CTCW * ctc_loss(txts, rem)
    elif rem is not None:
        l = l + torch.nn.functional.mse_loss(2.0 * torch.sigmoid(aux(pooled["x"]).squeeze(1)), torch.tensor(rem, device="cuda", dtype=torch.float32))
    return l, (lg.argmax(1) == y).sum().item()


@torch.no_grad()
def validate(dl):
    m.eval(); ls, ok, n = [], 0, 0
    for feats, _, tags, *_ in dl:
        l, k = loss_batch(feats, tags); ls.append(l.item()); ok += k; n += len(tags)
    m.train()
    return float(np.mean(ls)), ok / n


tr = load(args.train)
va = load("dev_g")
va = [va[i] for i in np.random.default_rng(0).permutation(len(va))[:args.nval]]
dl = torch.utils.data.DataLoader(DS(tr, False), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                 collate_fn=collate, drop_last=True, persistent_workers=True)
dv = torch.utils.data.DataLoader(DS(va, True), batch_size=args.bs, shuffle=False, num_workers=args.workers, collate_fn=collate)
opt = torch.optim.AdamW([{"params": m.encoder.parameters(), "lr": args.lr}, {"params": list(m.head.parameters()) + (list(aux.parameters()) if NEXT or CTCW else []), "lr": 1e-3}],
                        betas=(0.9, 0.99), eps=1e-6, weight_decay=0.01)
total = args.epochs * len(dl) // args.acc
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / args.warmup, max(0.0, (total - s) / max(1, total - args.warmup))))
print(f"学習 {len(tr)} 区間・検証 {len(va)} 区間・更新 {total} 回", flush=True)
best, step, t0 = math.inf, 0, time.time()
for ep in range(1, args.epochs + 1):
    run = []
    for k, (feats, txts, tags, _, *rem) in enumerate(dl):
        loss, _ = loss_batch(feats, tags, *rem, txts=txts)
        (loss / args.acc).backward(); run.append(loss.item())
        if (k + 1) % args.acc:
            continue
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 100 == 0:
            print(f"エポック {ep} 更新 {step}/{total}  損失 {np.mean(run[-100 * args.acc:]):.4f}  {time.time() - t0:.0f} 秒  "
                  f"GPU メモリ最大 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if step % 2000 == 0 or (k + 1) // args.acc == len(dl) // args.acc:
            vl, acc = validate(dv)
            mark = ""
            if vl < best:
                best = vl; mark = "  ← 最良"
                torch.save(m.state_dict(), f"{args.out}/best.pt")
                if CTCW:   # 11/26：音声認識の性能（CER）を後で測るため，補助タスクの出力層も残す
                    torch.save(aux.state_dict(), f"{args.out}/best_aux.pt")
            print(f"[エポック {ep} 更新 {step}] 検証の損失 {vl:.4f}  正解率 {acc:.4f}{mark}", flush=True)
print(f"完了．最良の検証の損失 {best:.4f}  {time.time() - t0:.0f} 秒", flush=True)
