#!/usr/bin/env python
"""utterance-level の発話区間末モデルを評価する（発話を最後まで聞いて 1 区間 1 予測）。

学習済み turntaking モデルをロードし、各区間の音声をエンコード → 発話全体プール →
発話区間末ヘッドで no/yes/other を 1 つ予測。accuracy・クラス別 prec/recall/F1・
macro-F1・no/yes 二値分離を出す。早期確定（SPRT/TEASER）は使わない。

入力データは dump/raw/<set>（utt 単位・共有不要）。過去文脈モデル（ctx_dim>0）は
--ctx_scp で ctx_vec.scp を渡す。
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def read_segments(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                rows.append((p[0], p[1], float(p[2]), float(p[3])))
    return rows


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1].strip()
    return m


@torch.no_grad()
def predict_utt(model, speech, device, ctx=None):
    """waveform(numpy) -> 予測クラス int（発話全体プール）。"""
    x = torch.from_numpy(speech.astype(np.float32)).unsqueeze(0).to(device)
    xl = torch.tensor([x.shape[1]], device=device)
    enc = model.encode(x, xl)
    eo = enc[0]
    if isinstance(eo, tuple):
        eo = eo[0]
    el = enc[1]
    ctx_t = None
    if getattr(model, "ctx_dim", 0) > 0 and ctx is not None:
        ctx_t = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0).to(device)
    logits = model.turntaking_head(eo, el, ctx_t)   # (1, C)
    return int(logits[0].argmax())


@torch.no_grad()
def predict_utt_xfmr(model, wav, device, past_mat, return_prob=False):
    """Stage2(xfmr): 凍結encoder出力 c1..cT ＋ 過去max-poolベクトル V1..VN を
    self-attention に通し、現発話の最終フレーム位置 C'_T を MLP へ。"""
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    xl = torch.tensor([x.shape[1]], device=device)
    enc = model.encode(x, xl)
    eo = enc[0]
    if isinstance(eo, tuple):
        eo = eo[0]
    el = enc[1]
    D = eo.shape[-1]
    if past_mat is None:
        # N=0（過去なし）。ダミートークンを挿さず長さ 0 で渡す（学習時と同じ扱い）。
        pv = eo.new_zeros(1, 0, D)
        pl = torch.zeros(1, dtype=torch.long, device=eo.device)
    else:
        pm = np.asarray(past_mat, dtype=np.float32)
        if pm.ndim == 1:
            pm = pm[None, :]
        pv = torch.from_numpy(pm).unsqueeze(0).to(eo.device)
        pl = torch.tensor([pm.shape[0]], dtype=torch.long, device=eo.device)
    logits = model.turntaking_head(pv, pl, eo, el)     # (1, C)
    if return_prob:
        return int(logits[0].argmax()), logits[0].softmax(-1).cpu().numpy()
    return int(logits[0].argmax())


@torch.no_grad()
def predict_utt_xfmr_pool(model, wav, past_wav, device, past_bounds=None):
    """過去音声 past_wav を凍結エンコーダで符号化し、指定の圧縮で要約 → head。

    past_bounds は連結した各過去発話のサンプル数（tt_past_pool="utt" のときのみ使う）。
    """
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    enc = model.encode(x, torch.tensor([x.shape[1]], device=device))
    eo = enc[0][0] if isinstance(enc[0], tuple) else enc[0]
    el = enc[1]
    D = eo.shape[-1]
    # 過去音声が短すぎると Conformer の畳み込みが通らない（無音のみ＝過去なし等）。
    # 十分長い場合のみ符号化し、失敗/短い場合は過去なし扱い。
    po = None
    if past_wav is not None and len(past_wav) >= 4000:      # 0.25s 以上
        try:
            px = torch.from_numpy(past_wav.astype(np.float32)).unsqueeze(0).to(device)
            penc = model.encode(px, torch.tensor([px.shape[1]], device=device))
            po = penc[0][0] if isinstance(penc[0], tuple) else penc[0]
            pl_frames = penc[1].to(po.device)
        except Exception:
            po = None
    if po is None:
        pv = eo.new_zeros(1, 0, D); pl = torch.zeros(1, dtype=torch.long, device=eo.device)
    else:
        if model.tt_past_pool == "frames":
            pv = po                                     # pooling せず全フレーム
            pl = pl_frames
        elif model.tt_past_pool in ("conv", "query", "topk"):
            pv, pl = model.past_compressor(po, pl_frames)   # 圧縮した記憶スロット列
        elif model.tt_past_pool == "utt":
            b = torch.tensor([past_bounds or [len(past_wav)]],
                             dtype=torch.long, device=po.device)
            pv, pl = model.past_compressor(
                po, pl_frames, b, torch.tensor([b.shape[1]], device=po.device))
        elif model.tt_past_pool == "max":
            m = torch.arange(po.shape[1], device=po.device)[None, :] >= pl_frames[:, None]
            pooled = po.masked_fill(m.unsqueeze(-1), float("-inf")).max(dim=1).values
            pv = pooled.unsqueeze(1); pl = torch.ones(1, dtype=torch.long, device=eo.device)
        else:
            pooled = model.past_attn_pool(po, pl_frames)
            pv = pooled.unsqueeze(1); pl = torch.ones(1, dtype=torch.long, device=eo.device)
    return int(model.turntaking_head(pv, pl, eo, el)[0].argmax())


@torch.no_grad()
def predict_utt_xfmr_text(model, wav, past_ids, device):
    """過去テキスト past_ids（トークンID列）を凍結デコーダ埋め込み→射影 → head。"""
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    enc = model.encode(x, torch.tensor([x.shape[1]], device=device))
    eo = enc[0][0] if isinstance(enc[0], tuple) else enc[0]
    el = enc[1]; D = eo.shape[-1]
    if not past_ids:
        pv = eo.new_zeros(1, 0, D); pl = torch.zeros(1, dtype=torch.long, device=eo.device)
    else:
        tok = torch.tensor([past_ids], dtype=torch.long, device=eo.device).clamp(min=0)
        emb = model.decoder.embed(tok)
        pv = model.past_text_proj(emb)
        pl = torch.tensor([len(past_ids)], dtype=torch.long, device=eo.device)
    return int(model.turntaking_head(pv, pl, eo, el)[0].argmax())


def eval_xfmr_text(model, wav_scp, tags, args):
    """過去テキスト版の評価。past_text.scp（uttid id1 id2 ...）を読む。"""
    past_map = {}
    if args.past_text_scp:
        for line in open(args.past_text_scp, encoding="utf-8"):
            p = line.split()
            if len(p) >= 2:
                past_map[p[0]] = [int(t) for t in p[1:]]
        print(f"past_text 読込: {len(past_map)} 件", flush=True)
    utts = [u for u in wav_scp if u in tags]
    if args.max_utts > 0:
        utts = utts[: args.max_utts]
    preds, labels = [], []
    n_done = n_skip = 0
    for u in utts:
        p = wav_scp[u]
        if "|" in p:
            n_skip += 1; continue
        try:
            wav, _ = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            n_skip += 1; continue
        if wav.ndim > 1: wav = wav[:, 0]
        if len(wav) < 320:
            n_skip += 1; continue
        ids = past_map.get(u, [])
        ids = [i for i in ids if i > 0]                 # 0(=過去なし埋め) は落とす
        preds.append(predict_utt_xfmr_text(model, wav, ids, args.device))
        labels.append(tags[u]); n_done += 1
        if n_done % 2000 == 0:
            print(f"  ...{n_done} 区間", flush=True)
    print(f"処理: {n_done} (skip={n_skip})")
    report(preds, labels)


def eval_xfmr_pool(model, wav_scp, tags, args):
    """past_speech.scp（1行に過去発話の音声パスを空白区切り）を読み、max/attn pooling で評価。"""
    past_map = {}
    if args.past_speech_scp:
        for line in open(args.past_speech_scp, encoding="utf-8"):
            p = line.split()
            if len(p) >= 2:
                past_map[p[0]] = p[1:]
        print(f"past_speech 読込: {len(past_map)} 件 / pool={model.tt_past_pool}", flush=True)
    utts = [u for u in wav_scp if u in tags]
    if args.max_utts > 0:
        utts = utts[: args.max_utts]
    preds, labels = [], []
    n_done = n_skip = 0
    for u in utts:
        p = wav_scp[u]
        if "|" in p:
            n_skip += 1; continue
        try:
            wav, _ = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            n_skip += 1; continue
        if wav.ndim > 1: wav = wav[:, 0]
        if len(wav) < 320:
            n_skip += 1; continue
        past_wav = None
        past_bounds = None
        paths = past_map.get(u)
        if paths:
            arrs = []
            for pp in paths:
                try:
                    a, _ = sf.read(pp, dtype="float32", always_2d=False)
                    if a.ndim > 1: a = a[:, 0]
                    arrs.append(a)
                except Exception:
                    pass
            if arrs:
                cat = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
                # 無音のみ（過去なし）は None 扱い
                if len(cat) >= 320:
                    past_wav = cat
                    past_bounds = [len(a) for a in arrs]   # 発話境界（連結順）
        preds.append(predict_utt_xfmr_pool(model, wav, past_wav, args.device,
                                           past_bounds))
        labels.append(tags[u]); n_done += 1
        if n_done % 2000 == 0:
            print(f"  ...{n_done} 区間", flush=True)
    print(f"処理: {n_done} (skip={n_skip})")
    report(preds, labels)


def eval_xfmr(model, wav_scp, tags, args):
    """xfmr モデル評価: past_vec.scp（(n,256) 行列）を渡して予測。"""
    import kaldiio
    if args.past_vec_scp:
        past_map = dict(kaldiio.load_scp(args.past_vec_scp).items())
        print(f"past_vec 読込: {len(past_map)} 件", flush=True)
    else:
        past_map = {}
        print("past_vec 未指定 → N=0（過去なし）として評価", flush=True)
    utts = [u for u in wav_scp if u in tags]
    if args.max_utts > 0:
        utts = utts[: args.max_utts]
    preds, labels = [], []
    dump_f = open(args.dump_preds, "w", encoding="utf-8") if args.dump_preds else None
    if dump_f:
        dump_f.write("utt\ttrue\tpred\tp_cont\tp_end\tp_bc\tn_samp\n")
    n_done = n_skip = 0
    for u in utts:
        p = wav_scp[u]
        if "|" in p:
            n_skip += 1
            continue
        try:
            wav, _ = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            n_skip += 1
            continue
        if wav.ndim > 1:
            wav = wav[:, 0]
        if len(wav) < 320:
            n_skip += 1
            continue
        if dump_f:
            pr, prob = predict_utt_xfmr(model, wav, args.device, past_map.get(u),
                                        return_prob=True)
            dump_f.write(f"{u}\t{tags[u]}\t{pr}\t{prob[0]:.4f}\t{prob[1]:.4f}"
                         f"\t{prob[2]:.4f}\t{len(wav)}\n")
        else:
            pr = predict_utt_xfmr(model, wav, args.device, past_map.get(u))
        preds.append(pr)
        labels.append(tags[u])
        n_done += 1
        if n_done % 2000 == 0:
            print(f"  ...{n_done} 区間", flush=True)
    if dump_f:
        dump_f.close()
        print(f"予測ダンプ: {args.dump_preds}")
    print(f"処理: {n_done} (skip={n_skip})")
    if n_done:
        report(preds, labels)


@torch.no_grad()
def predict_utt_seqcat(model, cur_wav, past_wav, device):
    """seqcat 方式: [past ; current] を連結→エンコーダ→現発話フレーム pool→MLP。"""
    cur = torch.from_numpy(cur_wav.astype(np.float32)).unsqueeze(0).to(device)
    cur_len = torch.tensor([cur.shape[1]], device=device)
    enc = model.encode(cur, cur_len)                # 現発話のみ（cur_enc_lens 用）
    cur_enc_lens = enc[1]
    if past_wav is not None and len(past_wav) > 0:
        past = torch.from_numpy(past_wav.astype(np.float32)).unsqueeze(0).to(device)
        past_len = torch.tensor([past.shape[1]], device=device)
        logits = model._seqcat_head(cur, cur_len, cur_enc_lens, past, past_len)
    else:
        logits = model._seqcat_head(cur, cur_len, cur_enc_lens, None, None)
    return int(logits[0].argmax())


def report(preds, labels, num_classes=3):
    names = ["<継続>", "<終了>", "<相槌>"]   # index0=発話区間末でない, 1=発話区間末, 2=相槌
    conf = np.zeros((num_classes, num_classes), dtype=int)   # [true, pred]
    noyes_c = noyes_n = 0
    for y, p in zip(labels, preds):
        conf[y, p] += 1
        if y in (0, 1):
            noyes_n += 1
            noyes_c += int(p == y)   # no/yes のうち正しく当てた割合（other 予測は不正解扱い）
    total = conf.sum()
    acc = np.trace(conf) / max(total, 1)
    rec = [conf[c, c] / max(conf[c].sum(), 1) for c in range(num_classes)]
    prec = [conf[c, c] / max(conf[:, c].sum(), 1) for c in range(num_classes)]
    f1 = [2 * prec[c] * rec[c] / max(prec[c] + rec[c], 1e-8) for c in range(num_classes)]
    print("=" * 64)
    print(f"処理 {total} 区間   3クラス acc = {acc:.3f}   macro-F1 = {sum(f1)/3:.3f}")
    print(f"{'class':>8} | {'prec':>6} {'recall':>6} {'F1':>6} | {'件数':>6}")
    for c, nm in enumerate(names):
        print(f"{nm:>8} | {prec[c]:6.3f} {rec[c]:6.3f} {f1[c]:6.3f} | {conf[c].sum():6d}")
    print(f"継続/終了 二値分離 acc = {noyes_c/max(noyes_n,1):.3f}  (対象 {noyes_n} 区間)")
    print("混同行列 [真→予測] 継続/終了/相槌:")
    for c, nm in enumerate(names):
        print(f"  {nm:>8}: {conf[c].tolist()}")


def eval_seqcat(model, wav_scp, tags, args):
    """seqcat モデル評価: 各発話の直前N発話(同一話者)の音声を連結して past として渡す。"""
    import re
    seg_rows = read_segments(Path(args.seg_dir) / "segments")
    segmap = {u: (reco, st, en) for u, reco, st, en in seg_rows}
    utts = [u for u in wav_scp if u in segmap and u in tags]

    def sess(r):
        return re.sub(r"_IC\d+$", "", r)

    groups = defaultdict(list)
    for u in utts:
        reco = segmap[u][0]
        key = reco if args.scope == "same" else sess(reco)
        groups[key].append(u)
    for k in groups:
        groups[k].sort(key=lambda x: segmap[x][1])       # start 昇順
    idx = {u: i for k in groups for i, u in enumerate(groups[k])}

    if args.max_utts > 0:
        utts = utts[: args.max_utts]

    cache = {}

    def load(u):
        if u in cache:
            return cache[u]
        p = wav_scp.get(u)
        if p is None or "|" in p:
            return None
        try:
            w, _ = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            return None
        if w.ndim > 1:
            w = w[:, 0]
        cache[u] = w
        return w

    preds, labels = [], []
    n_done = n_skip = 0
    for u in utts:
        cur = load(u)
        if cur is None or len(cur) < 320:
            n_skip += 1
            continue
        reco, s, en = segmap[u]
        key = reco if args.scope == "same" else sess(reco)
        sib = groups[key]
        i = idx[u]
        past_list = []
        for j in range(i - 1, max(-1, i - 1 - args.n_past), -1):
            pu = sib[j]
            if segmap[pu][1] >= s - args.max_past_sec:
                past_list.append(pu)
            else:
                break
        past_list.reverse()                              # 古→新
        chunks = [load(pu) for pu in past_list]
        chunks = [c for c in chunks if c is not None and len(c) > 0]
        past_wav = np.concatenate(chunks) if chunks else None
        preds.append(predict_utt_seqcat(model, cur, past_wav, args.device))
        labels.append(tags[u])
        n_done += 1
        if n_done % 2000 == 0:
            print(f"  ...{n_done} 区間", flush=True)
    print(f"処理: {n_done} (skip={n_skip})")
    if n_done:
        report(preds, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True, help="dump/raw/<set>（wav.scp, tag）")
    ap.add_argument("--ctx_scp", default="", help="過去文脈 ctx_vec.scp（ctx_dim>0 モデル用）")
    ap.add_argument("--xfmr", action="store_true", help="Stage2(xfmr) モデルの評価")
    ap.add_argument("--past_vec_scp", default="", help="xfmr: past_vec.scp（(n,256) 行列）")
    ap.add_argument("--past_pool", action="store_true",
                    help="過去音声プーリング版(tt_past_pool=max/attn)の評価")
    ap.add_argument("--past_speech_scp", default="", help="past_pool: past_speech.scp")
    ap.add_argument("--past_text", action="store_true", help="過去テキスト版の評価")
    ap.add_argument("--past_text_scp", default="", help="past_text: past_text.scp（uttid id...）")
    ap.add_argument("--seqcat", action="store_true", help="seqcat モデルの評価（過去音声を連結）")
    ap.add_argument("--seg_dir", default="", help="seqcat: data/<set>（segments で過去選択）")
    ap.add_argument("--scope", choices=["same", "session"], default="same")
    ap.add_argument("--n_past", type=int, default=2)
    ap.add_argument("--max_past_sec", type=float, default=30.0)
    ap.add_argument("--max_utts", type=int, default=0)
    ap.add_argument("--dump_preds", default="",
                    help="xfmr: 区間ごとの正解/予測/確率を TSV に出す（誤り分析用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from espnet2.tasks.asr import TurnTakingASRTask
    model, _ = TurnTakingASRTask.build_model_from_file(args.config, args.model, args.device)
    model.eval()

    data_dir = Path(args.data_dir)
    wav_scp = read_map(data_dir / "wav.scp")
    tags = {k: int(v) for k, v in read_map(data_dir / "tag").items()}
    seg_path = data_dir / "segments"

    if args.past_text:
        eval_xfmr_text(model, wav_scp, tags, args)
        return

    if args.past_pool:
        eval_xfmr_pool(model, wav_scp, tags, args)
        return

    if args.xfmr:
        eval_xfmr(model, wav_scp, tags, args)
        return

    if args.seqcat:
        eval_seqcat(model, wav_scp, tags, args)
        return

    ctx_map = None
    if args.ctx_scp:
        import kaldiio
        ctx_map = {k: v for k, v in kaldiio.load_scp(args.ctx_scp).items()}
        print(f"ctx_vec 読込: {len(ctx_map)} 件")

    # 読み込み単位 (utt, path, start, stop)
    jobs = []
    if seg_path.is_file():
        segs = read_segments(seg_path)
        if args.max_utts > 0:
            segs = segs[: args.max_utts]
        for utt, reco, st, en in segs:
            jobs.append((utt, wav_scp.get(reco), st, en))
    else:
        items = list(wav_scp.items())
        if args.max_utts > 0:
            items = items[: args.max_utts]
        for utt, path in items:
            jobs.append((utt, path, None, None))

    preds, labels = [], []
    n_done = n_skip = 0
    for utt, path, st, en in jobs:
        if path is None or "|" in path or utt not in tags:
            n_skip += 1
            continue
        try:
            if st is None:
                wav, _ = sf.read(path, dtype="float32", always_2d=False)
            else:
                sr = sf.info(path).samplerate
                wav, _ = sf.read(path, start=int(st * sr), stop=int(en * sr),
                                 dtype="float32", always_2d=False)
        except Exception:
            n_skip += 1
            continue
        if wav.ndim > 1:
            wav = wav[:, 0]
        if len(wav) < 320:
            n_skip += 1
            continue
        ctx = ctx_map.get(utt) if ctx_map is not None else None
        preds.append(predict_utt(model, wav, args.device, ctx))
        labels.append(tags[utt])
        n_done += 1
        if n_done % 2000 == 0:
            print(f"  ...{n_done} 区間", flush=True)

    print(f"処理: {n_done} (skip={n_skip})")
    if n_done:
        report(preds, labels)


if __name__ == "__main__":
    main()
