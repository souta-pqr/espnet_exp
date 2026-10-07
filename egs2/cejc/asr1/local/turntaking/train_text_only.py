#!/usr/bin/env python
"""テキストのみで発話区間末を予測する Stage2 相当モデル（音声モデルとの対照実験）。

狙い: 「音声の情報を入れないとダメなのか / テキストだけでどこまで届くか」を、
音声モデル（run_xfmr_n0.sh / run_xfmr.sh）と**同一のヘッド・同一のハイパラ**で測る。

  音声版 (espnet_model_turntaking_xfmr.py):
      現発話 ─[凍結ASRエンコーダ]→ c1..cT (T,256)   ← 33ms フレーム列
      過去発話 ─[同上＋max-pool]→ v1..vN (N,256)
  本スクリプト (テキスト版):
      現発話 ─[トークン埋め込み]→ e1..eL (L,256)    ← 文字トークン列
      過去発話 ─[同上＋max-pool]→ v1..vN (N,256)

  どちらも同じ `TurnTakingXfmrHead`（self-attention 1層 → 現発話の最終位置 → MLP）に入る。
  ＝ 差分は「入力が音響フレーム列か文字トークン列か」だけ。

入力表現は 2 通り選べる（--embed）:
  trainable: 埋め込み(256次元)を乱数初期化して学習する。★既定。
             テキスト側に最良の表現を学ばせる＝「テキストだけでどこまで届くか」の上限。
  frozen   : Stage1 純粋ASR の transducer decoder 埋め込みを**凍結**して使い、
             線形射影で 256 次元に落とす。音声版の「凍結エンコーダ」と対称な条件。
             （decoder 埋め込みは 512 次元なので射影が要る＝学習パラメータは完全一致しない）

公平性として揃えているもの: ヘッドのクラスと構成 / クラス重み / 最適化器・学習率・warmup /
accum_grad・grad_clip / エポック数 / 上位Nエポック平均 / seed / データ分割 / 評価指標。
揃わないもの: 入力系列の中身（音響フレーム列 vs 文字トークン列）と、入力表現の学習パラメータ数。

ラベルは data/<set>/tag（0=継続, 1=完了, 2=相槌）。評価指標は音声版と同じ report() を使う。

注意: data/<set>/text は**正解書き起こし**（(F …)(D …) の付与も含む）。したがって本条件は
「テキストが完璧に取れた場合の上限」であり、ASR 出力を使う実運用条件ではない。
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_multitask import report  # noqa: E402  音声版と同一の指標計算

from espnet2.asr.espnet_model_turntaking_xfmr import TurnTakingXfmrHead  # noqa: E402


# ---------------------------------------------------------------- データ読み込み
def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1].strip()
    return m


def read_segments(path):
    """utt -> (reco, start)"""
    seg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                seg[p[0]] = (p[1], float(p[2]))
    return seg


def session_of(reco):
    return re.sub(r"_IC\d+$", "", reco)


class TagTextDataset(torch.utils.data.Dataset):
    """data/<set> から (現発話トークン列, 過去N発話のトークン列, ラベル) を作る。

    過去発話の選び方は local/extract_past_vectors.py と完全に同じ規則:
    同一 reco（scope=same）または同一セッション（scope=session）内を start 昇順に並べ、
    直前から最大 n_past 本、ただし max_past_sec より古くなったらそこで打ち切り。
    """

    def __init__(self, data_dir, token2id, n_past=0, scope="same", max_past_sec=30.0,
                 strip_tags=False):
        d = Path(data_dir)
        text = read_map(d / "text")
        tag = {k: int(v) for k, v in read_map(d / "tag").items()}
        unk = token2id["<unk>"]

        def encode(s):
            toks = s.split()
            if strip_tags:
                toks = [t for t in toks if t not in ("(F", "(D", ")")]
            return [token2id.get(t, unk) for t in toks]

        ids = {u: encode(t) for u, t in text.items()}

        self.items = []
        self.n_skip_empty = 0
        if n_past > 0:
            seg = read_segments(d / "segments")
            groups = defaultdict(list)
            for u in ids:
                if u in seg:
                    reco = seg[u][0]
                    groups[reco if scope == "same" else session_of(reco)].append(u)
            for k in groups:
                groups[k].sort(key=lambda x: seg[x][1])
            idx = {u: i for k in groups for i, u in enumerate(groups[k])}

        for u, cur in ids.items():
            if u not in tag:
                continue
            if not cur:                       # 空テキストは音声版の短すぎ発話 skip に相当
                self.n_skip_empty += 1
                continue
            past = []
            if n_past > 0 and u in idx:
                reco, s = seg[u]
                sib = groups[reco if scope == "same" else session_of(reco)]
                i = idx[u]
                for j in range(i - 1, max(-1, i - 1 - n_past), -1):
                    pu = sib[j]
                    if seg[pu][1] >= s - max_past_sec and ids.get(pu):
                        past.append(ids[pu])
                    else:
                        break
                past.reverse()                # 古→新
            self.items.append((u, cur, past, tag[u]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    """現発話・過去発話をそれぞれパディングして返す。"""
    utts = [b[0] for b in batch]
    curs = [b[1] for b in batch]
    pasts = [b[2] for b in batch]
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)

    B = len(batch)
    L = max(len(c) for c in curs)
    cur = torch.zeros(B, L, dtype=torch.long)
    cur_lens = torch.tensor([len(c) for c in curs], dtype=torch.long)
    for i, c in enumerate(curs):
        cur[i, : len(c)] = torch.tensor(c, dtype=torch.long)

    N = max((len(p) for p in pasts), default=0)
    if N == 0:
        return utts, cur, cur_lens, None, None, labels
    Lp = max((len(x) for p in pasts for x in p), default=1)
    past = torch.zeros(B, N, Lp, dtype=torch.long)
    past_tok_lens = torch.zeros(B, N, dtype=torch.long)
    past_lens = torch.tensor([len(p) for p in pasts], dtype=torch.long)
    for i, p in enumerate(pasts):
        for j, x in enumerate(p):
            past[i, j, : len(x)] = torch.tensor(x, dtype=torch.long)
            past_tok_lens[i, j] = len(x)
    return utts, cur, cur_lens, (past, past_tok_lens, past_lens), None, labels


# ---------------------------------------------------------------- モデル
class TextOnlyTurnTaking(nn.Module):
    """トークン埋め込み → （音声版と同一の）TurnTakingXfmrHead。

    音声版の enc_out (B,T,256) の代わりに埋め込み列 (B,L,256) を流すだけで、
    ヘッド以降は同じクラス・同じ設定を使う。
    """

    def __init__(self, vocab_size, head_conf, freeze_embed=False, init_embed=None):
        super().__init__()
        d = head_conf["d_model"]
        d_emb = init_embed.shape[1] if init_embed is not None else d
        self.embed = nn.Embedding(vocab_size, d_emb, padding_idx=0)
        if init_embed is not None:
            self.embed.weight.data.copy_(init_embed)
        if freeze_embed:
            self.embed.weight.requires_grad_(False)
        # 凍結埋め込みが d_model と異なる次元のときだけ線形射影を挟む
        self.proj = nn.Linear(d_emb, d) if d_emb != d else nn.Identity()
        self.head = TurnTakingXfmrHead(**head_conf)

    def _embed(self, x):
        return self.proj(self.embed(x))

    def _pool_past(self, past, past_tok_lens):
        """過去発話ごとに埋め込みを有効トークンで max-pool → (B,N,D)。音声版の max-pool と同じ。"""
        B, N, Lp = past.shape
        e = self._embed(past.view(B * N, Lp))                       # (B*N, Lp, D)
        mask = (torch.arange(Lp, device=past.device)[None, :]
                < past_tok_lens.view(B * N, 1))                     # (B*N, Lp)
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        v = e.max(dim=1).values                                     # (B*N, D)
        # 長さ 0 の過去（パディング分）は -inf のままなので 0 に潰す
        return torch.nan_to_num(v, neginf=0.0).view(B, N, -1)

    def forward(self, cur, cur_lens, past_pack):
        e = self._embed(cur)                                        # (B, L, D)
        if past_pack is None:
            past_vec = e.new_zeros(e.shape[0], 0, e.shape[-1])
            past_lens = torch.zeros(e.shape[0], dtype=torch.long, device=e.device)
        else:
            past, past_tok_lens, past_lens = past_pack
            past_vec = self._pool_past(past, past_tok_lens)
        return self.head(past_vec, past_lens, e, cur_lens)


# ---------------------------------------------------------------- 学習
class WarmupLR:
    """ESPnet の WarmupLR と同一式。"""

    def __init__(self, optimizer, base_lr, warmup_steps):
        self.opt, self.base_lr, self.w = optimizer, base_lr, warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        n = self.step_num
        lr = self.base_lr * self.w ** 0.5 * min(n ** -0.5, n * self.w ** -1.5)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


@torch.no_grad()
def evaluate(model, loader, device, class_weight):
    model.eval()
    preds, labels, tot_loss, n = [], [], 0.0, 0
    for _, cur, cur_lens, past_pack, _, y in loader:
        cur, cur_lens, y = cur.to(device), cur_lens.to(device), y.to(device)
        if past_pack is not None:
            past_pack = tuple(t.to(device) for t in past_pack)
        logits = model(cur, cur_lens, past_pack)
        tot_loss += F.cross_entropy(logits, y, weight=class_weight,
                                    reduction="sum").item()
        n += y.numel()
        preds += logits.argmax(-1).tolist()
        labels += y.tolist()
    return tot_loss / max(n, 1), float(np.mean(np.array(preds) == np.array(labels))), \
        preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="myconf/train_asr_turntaking_xfmr.yaml",
                    help="音声版の学習 config。ヘッド構成・最適化設定をここから読む（公平性）")
    ap.add_argument("--token_list", default="data/jp_token_list/word/tokens.txt")
    ap.add_argument("--train_dir", default="data/train_nodup")
    ap.add_argument("--valid_dir", default="data/train_dev")
    ap.add_argument("--eval_dir", default="data/eval")
    ap.add_argument("--embed", choices=["trainable", "frozen"], default="trainable",
                    help="trainable=乱数初期化して学習（既定・テキストの上限）/ "
                         "frozen=Stage1 ASR decoder 埋め込みを凍結して射影")
    ap.add_argument("--stage1_model", default="exp/asr_20260713-pureasr/valid.loss.ave.pth",
                    help="--embed frozen のとき decoder.embed を読む Stage1 モデル")
    ap.add_argument("--n_past", type=int, default=0)
    ap.add_argument("--scope", choices=["same", "session"], default="same")
    ap.add_argument("--max_past_sec", type=float, default=30.0)
    ap.add_argument("--strip_tags", action="store_true",
                    help="(F …)(D …) の非流暢タグを落として学習/評価する")
    ap.add_argument("--batch_size", type=int, default=14,
                    help="音声版 batch_bins=700000 の実測 ≈14 発話/minibatch に合わせる")
    ap.add_argument("--outdir", default="exp/textonly")
    ap.add_argument("--max_epoch", type=int, default=0, help="0=config の値を使う")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- 音声版 config からヘッド構成と最適化設定を取り出す（手打ちしない＝ズレ防止）
    conf = yaml.safe_load(open(args.config, encoding="utf-8"))
    mc = conf["model_conf"]
    d_model = conf["encoder_conf"]["output_size"]
    head_conf = dict(
        d_model=d_model,
        head_type=mc.get("tt_head_type", "selfattn"),
        n_layers=mc.get("tt_xfmr_layers", 1),
        n_heads=mc.get("tt_xfmr_heads", 4),
        d_ff=mc.get("tt_xfmr_ff", 1024),
        dropout=mc.get("tt_dropout", 0.1),
        n_classes=mc.get("num_tag_classes", 3),
        d_hidden=mc.get("tt_d_hidden", 128),
    )
    class_w = mc.get("tag_class_weight")
    max_epoch = args.max_epoch or conf.get("max_epoch", 30)
    accum_grad = conf.get("accum_grad", 8)
    grad_clip = conf.get("grad_clip", 5)
    base_lr = conf["optim_conf"]["lr"]
    warmup = conf["scheduler_conf"]["warmup_steps"]
    keep_nbest = conf.get("keep_nbest_models", 10)
    seed = conf.get("seed", 30)

    torch.manual_seed(seed)
    np.random.seed(seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 語彙
    tokens = [l.rstrip("\n") for l in open(args.token_list, encoding="utf-8")]
    token2id = {t: i for i, t in enumerate(tokens)}
    print(f"語彙: {len(tokens)}  head_conf={head_conf}")

    # ---- データ
    ds_kw = dict(token2id=token2id, n_past=args.n_past, scope=args.scope,
                 max_past_sec=args.max_past_sec, strip_tags=args.strip_tags)
    tr = TagTextDataset(args.train_dir, **ds_kw)
    va = TagTextDataset(args.valid_dir, **ds_kw)
    te = TagTextDataset(args.eval_dir, **ds_kw)
    print(f"train={len(tr)} (空テキスト除外 {tr.n_skip_empty})  "
          f"valid={len(va)} ({va.n_skip_empty})  eval={len(te)} ({te.n_skip_empty})")

    mk = lambda d, sh: torch.utils.data.DataLoader(  # noqa: E731
        d, batch_size=args.batch_size, shuffle=sh, collate_fn=collate,
        num_workers=args.num_workers, drop_last=False)
    tr_ld, va_ld, te_ld = mk(tr, True), mk(va, False), mk(te, False)

    # ---- モデル
    init_embed, freeze = None, False
    if args.embed == "frozen":
        sd = torch.load(args.stage1_model, map_location="cpu", weights_only=False)
        key = next(k for k in sd if k.endswith("decoder.embed.weight"))
        init_embed = sd[key]
        freeze = True
        if init_embed.shape[0] != len(tokens):
            raise ValueError(f"語彙数が一致しません: {init_embed.shape[0]} != {len(tokens)}")
        print(f"Stage1 decoder 埋め込みを凍結利用: {key} {tuple(init_embed.shape)}"
              + (f" → 線形射影で {d_model} 次元へ" if init_embed.shape[1] != d_model else ""))

    device = args.device
    model = TextOnlyTurnTaking(len(tokens), head_conf, freeze, init_embed).to(device)
    n_head = sum(p.numel() for p in model.head.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"パラメータ: ヘッド {n_head/1e6:.3f}M / 学習対象 合計 {n_train/1e6:.3f}M")

    class_weight = (torch.tensor(class_w, dtype=torch.float, device=device)
                    if class_w else None)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=base_lr)
    sched = WarmupLR(opt, base_lr, warmup)

    # ---- 学習ループ（accum_grad / grad_clip / warmuplr は音声版と同じ）
    ckpts = []          # (tag_acc, epoch, state_dict)
    for ep in range(1, max_epoch + 1):
        model.train()
        run_loss = run_acc = seen = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (_, cur, cur_lens, past_pack, _, y) in enumerate(tr_ld, 1):
            cur, cur_lens, y = cur.to(device), cur_lens.to(device), y.to(device)
            if past_pack is not None:
                past_pack = tuple(t.to(device) for t in past_pack)
            logits = model(cur, cur_lens, past_pack)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            (loss / accum_grad).backward()
            if it % accum_grad == 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            bs = y.numel()
            run_loss += loss.item() * bs
            run_acc += (logits.argmax(-1) == y).sum().item()
            seen += bs
        vl, vacc, _, _ = evaluate(model, va_ld, device, class_weight)
        print(f"epoch {ep:3d}: train loss={run_loss/seen:.4f} acc={run_acc/seen:.4f} | "
              f"valid loss={vl:.4f} tag_acc={vacc:.4f} | lr={opt.param_groups[0]['lr']:.2e}",
              flush=True)
        ckpts.append((vacc, ep, {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}))
        ckpts.sort(key=lambda x: -x[0])
        ckpts = ckpts[:keep_nbest]

    # ---- 音声版の valid.tag_acc.{ave,best}.pth と同じ 2 つを作って両方評価する。
    # 学習が不安定なとき（frozen 埋め込みなど）、異なる盆地のエポックを重み平均すると
    # 壊れることがある。best 単体も出しておかないと平均の失敗を性能と誤読してしまう。
    print(f"\n上位 {len(ckpts)} エポックの valid tag_acc: "
          f"{[(e, round(a, 4)) for a, e, _ in ckpts]}")
    header = f"テキストのみ / embed={args.embed} / N={args.n_past}"

    best = ckpts[0]
    torch.save(best[2], outdir / "tag_acc.best.pth")
    model.load_state_dict(best[2])
    _, _, preds, labels = evaluate(model, te_ld, device, class_weight)
    print(f"\n===== eval ({args.eval_dir}) : {header} / best単体 (epoch {best[1]}) =====")
    report(preds, labels)

    avg = {k: torch.stack([c[2][k].float() for c in ckpts]).mean(0)
           for k in ckpts[0][2]}
    torch.save(avg, outdir / "tag_acc.ave.pth")
    model.load_state_dict(avg)
    _, _, preds, labels = evaluate(model, te_ld, device, class_weight)
    print(f"\n===== eval ({args.eval_dir}) : {header} / 上位{len(ckpts)}平均 =====")
    report(preds, labels)


if __name__ == "__main__":
    main()
