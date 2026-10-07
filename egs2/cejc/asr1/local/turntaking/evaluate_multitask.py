#!/usr/bin/env python
"""utterance-level の発話区間末モデルを評価する（発話を最後まで聞いて 1 区間 1 予測）。

学習済み turntaking モデルをロードし、各区間の音声をエンコード → 発話全体プール →
発話区間末ヘッドで no/yes/other を 1 つ予測。accuracy・クラス別 prec/recall/F1・
macro-F1・no/yes 二値分離を出す。早期確定（SPRT/TEASER）は使わない。

入力データは dump/raw/<set>（utt 単位・共有不要）。過去文脈モデル（ctx_dim>0）は
--ctx_scp で ctx_vec.scp を渡す。
"""
import argparse
import math
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
def predict_utt_xfmr_pool(model, wav, past_wav, device, past_bounds=None,
                          return_prob=False, dec_ids=None):
    """過去音声 past_wav を凍結エンコーダで符号化し、指定の圧縮で要約 → head。

    past_bounds は連結した各過去発話のサンプル数（tt_past_pool="utt" のときのみ使う）。
    """
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    enc = model.encode(x, torch.tensor([x.shape[1]], device=device))
    eo = enc[0][0] if isinstance(enc[0], tuple) else enc[0]
    el = enc[1]
    D = eo.shape[-1]
    # 学習時は短い過去もミニバッチ内でパディングされて必ず符号化される（長さはマスクで扱う）。
    # 単発推論では入力が短いと Conformer の畳み込み・ブロック処理が通らないため、
    # **最小長までゼロ詰めしつつ長さは真値を渡す**ことで学習時と同じ条件にする。
    # （以前は 0.25s 未満を「過去なし」に落としていて train/eval が食い違っていた）
    # encode() は渡した長さで波形を切るため、長さに真値を渡すとパディングが消えてしまう。
    # 学習時と同じ「パディングごと符号化し、有効フレーム数だけを使う」形にする。
    MIN_PAST = 16000        # 1 s（block_size 18 フレーム ≒ 0.6 s を確実に上回る長さ）
    po = None
    if past_wav is not None and len(past_wav) > 0:
        n = len(past_wav)
        pw = past_wav if n >= MIN_PAST else np.pad(past_wav, (0, MIN_PAST - n))
        try:
            px = torch.from_numpy(pw.astype(np.float32)).unsqueeze(0).to(device)
            penc = model.encode(px, torch.tensor([len(pw)], device=device))
            po = penc[0][0] if isinstance(penc[0], tuple) else penc[0]
            # 有効フレーム数 = 真の長さ相当（ゼロ詰め部分は捨てる）
            fps = penc[1].item() / float(len(pw))            # フレーム/サンプル
            nf = max(1, min(int(math.ceil(n * fps)), po.shape[1]))
            pl_frames = torch.tensor([nf], dtype=torch.long, device=po.device)
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
    # tt_use_dec: Transducer デコーダ状態を head へ。学習時は正解トークンだが、
    # 推論時は ASR 仮説のトークン列を渡すのが実運用に即した条件（--dec_text_scp）。
    dv = dl = None
    if getattr(model, "tt_use_dec", False):
        ids = dec_ids or []
        t = torch.tensor([ids], dtype=torch.long, device=eo.device)
        dv, dl = model._decoder_states(t, torch.tensor([len(ids)], device=eo.device),
                                       eo.dtype)
    logits, hor = model.turntaking_head(pv, pl, eo, el, dv, dl, return_horizon=True)
    if return_prob:
        prob = logits[0].softmax(-1).cpu().numpy()
        # 先読みヘッド（tt_horizons）があれば各 h の「h 秒以内に完了で終わる」確率も返す
        ph = hor[0].sigmoid().cpu().numpy() if hor is not None else np.zeros(0)
        return int(logits[0].argmax()), prob, ph
    return int(logits[0].argmax())


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


@torch.no_grad()
def eval_frame_level(model, wav_scp, tags, args):
    """1 回の符号化で、現発話の**全フレーム位置**での判定をまとめて出す。

    これまでの前向きグリッドは、時刻ごとに波形を切って符号化し直していた。
    しかしエンコーダはストリーミング（contextual block）なので、
    **同じ絶対位置の表現は後続を 6 フレーム（約 0.2 秒）与えた時点で完全に確定する**
    （実測 cos 類似 1.000）。つまり長い音声を 1 回符号化して位置 p を読めば、
    実際のストリーミングが時刻 p に持っている表現と一致する。

    一方、波形を切る方式は切った端が**境界として処理される**ため、
    同じ時刻でも表現が変わる（cos 類似 0.46）。
    実運用では発話の途中に境界は生じないので、こちらの方式のほうが実態に近い。

    出力は 1 行 = (区間, 位置)。elapsed は発話開始からの経過秒。
    """
    full_map, seg_map = {}, {}
    for line in open(args.full_wav_scp):
        q = line.split(maxsplit=1)
        if len(q) == 2:
            full_map[q[0]] = q[1].strip()
    for line in open(args.segments):
        q = line.split()
        if len(q) == 4:
            seg_map[q[0]] = (q[1], float(q[2]), float(q[3]))
    past_map = {}
    if args.past_speech_scp:
        for line in open(args.past_speech_scp):
            q = line.split()
            if len(q) >= 2:
                past_map[q[0]] = q[1:]
    utts = [u for u in wav_scp if u in tags and u in seg_map]
    if args.max_utts > 0:
        utts = utts[: args.max_utts]
    keep = int(args.frame_max_sec * 16000)

    f = open(args.dump_preds, "w", encoding="utf-8")
    f.write("utt\ttrue\tpos\telapsed\tp_cont\tp_end\tp_bc\n")
    n_done = n_skip = 0
    for u in utts:
        rec, b, e = seg_map[u]
        if rec not in full_map:
            n_skip += 1; continue
        try:                      # 発話開始から frame_max_sec 秒ぶんを元録音から読む
            wav = sf.read(full_map[rec], dtype="float32",
                          start=int(b * 16000), frames=keep, always_2d=False)[0]
        except Exception:
            n_skip += 1; continue
        if wav.ndim > 1:
            wav = wav[:, 0]
        if len(wav) < 3200:
            n_skip += 1; continue
        x = torch.from_numpy(wav).unsqueeze(0).to(args.device)
        enc = model.encode(x, torch.tensor([len(wav)], device=args.device))
        eo = enc[0][0] if isinstance(enc[0], tuple) else enc[0]
        F = int(enc[1].item()); D = eo.shape[-1]
        sec_per_frame = (len(wav) / 16000.0) / F

        # 過去発話（従来と同じ扱い）
        pv = eo.new_zeros(1, 0, D)
        pl = torch.zeros(1, dtype=torch.long, device=eo.device)
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
            cat = np.concatenate(arrs) if len(arrs) > 1 else (arrs[0] if arrs else None)
            if cat is not None and len(cat) >= 320:
                MIN_PAST = 16000
                pw = cat if len(cat) >= MIN_PAST else np.pad(cat, (0, MIN_PAST - len(cat)))
                px = torch.from_numpy(pw.astype(np.float32)).unsqueeze(0).to(args.device)
                penc = model.encode(px, torch.tensor([len(pw)], device=args.device))
                po = penc[0][0] if isinstance(penc[0], tuple) else penc[0]
                fps = penc[1].item() / float(len(pw))
                nf = max(1, min(int(math.ceil(len(cat) * fps)), po.shape[1]))
                plf = torch.tensor([nf], dtype=torch.long, device=po.device)
                if model.tt_past_pool == "max":
                    msk = torch.arange(po.shape[1], device=po.device)[None, :] >= plf[:, None]
                    pooled = po.masked_fill(msk.unsqueeze(-1), float("-inf")).max(dim=1).values
                else:
                    pooled = model.past_attn_pool(po, plf)
                pv = pooled.unsqueeze(1)
                pl = torch.ones(1, dtype=torch.long, device=eo.device)

        # 全位置をバッチ次元に展開して head を 1 回だけ呼ぶ
        lens = torch.arange(1, F + 1, device=eo.device)
        eo_b = eo.expand(F, -1, -1)
        pv_b = pv.expand(F, -1, -1) if pv.shape[1] > 0 else pv.new_zeros(F, 0, D)
        pl_b = pl.expand(F)
        out = model.turntaking_head(pv_b, pl_b, eo_b, lens)
        prob = out.softmax(-1).cpu().numpy()
        for i in range(F):
            el = (i + 1) * sec_per_frame
            if el > args.frame_max_sec + 1e-6:
                break
            f.write(f"{u}\t{tags[u]}\t{i}\t{el:.4f}"
                    f"\t{prob[i,0]:.6f}\t{prob[i,1]:.6f}\t{prob[i,2]:.6f}\n")
        n_done += 1
        if n_done % 1000 == 0:
            print(f"  ...{n_done} 区間", flush=True)
    f.close()
    print(f"処理: {n_done} (skip={n_skip}) → {args.dump_preds}")


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


def eval_xfmr_pool(model, wav_scp, tags, args, token_list=None):
    """past_speech.scp（1行に過去発話の音声パスを空白区切り）を読み、max/attn pooling で評価。"""
    # tt_use_dec 用のトークン列。--dec_text_scp に ASR 仮説 text を渡すのが実運用条件、
    # data/<set>/text（正解）を渡すと理想カスケード相当の上限が測れる。
    dec_map = {}
    if args.dec_text_scp and token_list:
        w2i = {w: i for i, w in enumerate(token_list)}
        unk = w2i.get("<unk>", 1)
        for line in open(args.dec_text_scp, encoding="utf-8"):
            p = line.rstrip("\n").split(" ", 1)
            dec_map[p[0]] = [w2i.get(w, unk) for w in p[1].split()] if len(p) > 1 else []
        print(f"dec_text 読込: {len(dec_map)} 件", flush=True)
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
    n_done = n_skip = n_clip = 0
    # 早期確定：現発話を「発話末の trunc_sec 秒手前」で打ち切って渡す。head の読み出しは
    # 常に渡された音声の最終フレームなので、これがその時点での判定になる。
    # 短い発話は打ち切ると消えてしまうので MIN_KEEP で下限を設け、評価区間数を t によらず
    # 一定に保つ（下限に当たった件数は n_clip として報告する）。
    trunc = int(getattr(args, "trunc_sec", 0.0) * 16000)
    # 前向きグリッド：発話「開始から」keep_sec 秒だけ渡す。trunc_sec が発話末を基準に
    # 切るのに対し、こちらは経過時間を基準に切るので、実行時に観測できる状態と一致する。
    # 発話がそれより短ければ全体を渡す（その時刻には発話が既に終わっているため）。
    keep_fw = int(getattr(args, "keep_sec", 0.0) * 16000)
    MIN_KEEP = 4000                                    # 0.25 秒
    # 元録音からの切り出し：セグメント終端を越えて音声を伸ばす。
    # 発話が終わったあとの無音（や次の発話）が入るので、実運用のストリーミングに一致する。
    full_map, seg_map = {}, {}
    if getattr(args, "full_wav_scp", "") and getattr(args, "segments", ""):
        for line in open(args.full_wav_scp):
            q = line.split(maxsplit=1)
            if len(q) == 2: full_map[q[0]] = q[1].strip()
        for line in open(args.segments):
            q = line.split()
            if len(q) == 4: seg_map[q[0]] = (q[1], float(q[2]), float(q[3]))
        print(f"元録音から切り出す: 録音 {len(full_map)} / セグメント {len(seg_map)}", flush=True)
    dump_f = open(args.dump_preds, "w", encoding="utf-8") if args.dump_preds else None
    horizons = list(getattr(model, "tt_horizons", []) or [])
    if dump_f:
        # 先読みヘッドの列は末尾に足す（既存の読み込み側は先頭 7 列しか見ないので互換）
        hcols = "".join(f"\tp_h{h:g}" for h in horizons)
        dump_f.write("utt\ttrue\tpred\tp_cont\tp_end\tp_bc\tn_samp" + hcols + "\n")
    for u in utts:
        p = wav_scp[u]
        if "|" in p:
            n_skip += 1; continue
        try:
            wav, _ = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            n_skip += 1; continue
        if wav.ndim > 1: wav = wav[:, 0]
        if keep_fw > 0 and u in seg_map and seg_map[u][0] in full_map:
            rec, b, e = seg_map[u]
            try:                                       # 元録音から [開始, 開始+keep] を読む
                wav = sf.read(full_map[rec], dtype="float32",
                              start=int(b * 16000), frames=keep_fw, always_2d=False)[0]
                if wav.ndim > 1: wav = wav[:, 0]
                if keep_fw > int((e - b) * 16000): n_clip += 1   # 発話後の無音を含む
            except Exception:
                n_skip += 1; continue
        elif keep_fw > 0:
            if keep_fw >= len(wav):
                n_clip += 1                            # その時刻には発話が終わっている
            wav = wav[:min(keep_fw, len(wav))]
        elif trunc > 0:
            keep = len(wav) - trunc
            if keep < MIN_KEEP:
                keep = min(MIN_KEEP, len(wav)); n_clip += 1
            wav = wav[:keep]
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
        di = dec_map.get(u)
        if dump_f:
            pr, prob, ph = predict_utt_xfmr_pool(model, wav, past_wav, args.device,
                                                 past_bounds, return_prob=True, dec_ids=di)
            dump_f.write(f"{u}\t{tags[u]}\t{pr}\t{prob[0]:.6f}\t{prob[1]:.6f}"
                         f"\t{prob[2]:.6f}\t{len(wav)}"
                         + "".join(f"\t{x:.6f}" for x in ph) + "\n")
        else:
            pr = predict_utt_xfmr_pool(model, wav, past_wav, args.device, past_bounds,
                                       dec_ids=di)
        preds.append(pr)
        labels.append(tags[u]); n_done += 1
        if n_done % 2000 == 0:
            print(f"  ...{n_done} 区間", flush=True)
    if dump_f:
        dump_f.close()
        print(f"予測ダンプ: {args.dump_preds}")
    print(f"処理: {n_done} (skip={n_skip}"
          + (f", 打ち切り下限に到達={n_clip}" if trunc > 0 else "")
          + (f", 発話が既に終了={n_clip}" if keep_fw > 0 else "") + ")")
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
    ap.add_argument("--dec_text_scp", default="",
                    help="tt_use_dec: デコーダに渡すトークン列。ASR仮説 text=実運用条件 / "
                         "data/<set>/text=理想カスケード相当の上限")
    ap.add_argument("--trunc_sec", type=float, default=0.0,
                    help="早期確定: 現発話を発話末の t 秒手前で打ち切って評価する（過去発話は不変）")
    ap.add_argument("--full_wav_scp", default="",
                    help="元録音の wav.scp（チャネル単位）。--segments と併せて keep_sec に使うと、"
                         "セグメント終端を越えて音声を伸ばせる（発話後の無音を含む実運用条件）")
    ap.add_argument("--segments", default="",
                    help="--full_wav_scp と対で使う segments ファイル")
    ap.add_argument("--keep_sec", type=float, default=0.0,
                    help="前向きグリッド: 発話開始から t 秒だけ渡して評価する"
                         "（経過時間ベース＝実行時に観測できる状態。trunc_sec と排他）")
    ap.add_argument("--frame_level", action="store_true",
                    help="1 回の符号化で全フレーム位置の判定を出す（--full_wav_scp/--segments 必須）")
    ap.add_argument("--frame_max_sec", type=float, default=3.0,
                    help="frame_level: 発話開始から何秒ぶんを符号化するか")
    ap.add_argument("--dump_preds", default="",
                    help="xfmr: 区間ごとの正解/予測/確率を TSV に出す（誤り分析用）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from espnet2.tasks.asr import TurnTakingASRTask
    model, train_args = TurnTakingASRTask.build_model_from_file(
        args.config, args.model, args.device)
    model.eval()

    data_dir = Path(args.data_dir)
    wav_scp = read_map(data_dir / "wav.scp")
    tags = {k: int(v) for k, v in read_map(data_dir / "tag").items()}
    seg_path = data_dir / "segments"

    if args.frame_level:
        eval_frame_level(model, wav_scp, tags, args)
        return

    if args.past_text:
        eval_xfmr_text(model, wav_scp, tags, args)
        return

    if args.past_pool:
        eval_xfmr_pool(model, wav_scp, tags, args,
                       getattr(train_args, "token_list", None))
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
