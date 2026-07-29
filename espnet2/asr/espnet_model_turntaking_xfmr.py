"""凍結エンコーダ ＋ Transformer による発話区間末予測（Stage2 モデル）。

Stage1: 現発話のみで **純粋 ASR**（transducer）を学習 → エンコーダ完成。
Stage2: **エンコーダを凍結**し、以下だけを学習する。

    過去発話1..N ─(凍結Encoder・事前計算)→ 各発話ごとに **max pooling** → v1..vN (各 D 次元)
    現発話       ─(凍結Encoder)──────────→ c1..cT (T×D, **pooling しない**)

        Transformer 入力: [ v1, …, vN, c1, …, cT ]   （長さ N+T）
                 │  ← ここだけ学習（Encoder は凍結＝ASR は数値的に完全不変）
            Transformer
                 │
        出力の **現発話の最終フレーム位置 (cT の位置)** → MLP → 継続/完了/相槌

過去ベクトル v1..vN は `past_vec`（kaldi_ark, 形状 (n,D)）としてデータから与える（事前計算・可変長）。
ASR 損失は計算しない（凍結済みのため）。総損失 = loss_tag のみ。
"""
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable


def sinusoidal_pe(length: int, d_model: int, device, dtype):
    """位置符号（Transformer の順序情報用・パラメータ不要）。"""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(length, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe.to(dtype).unsqueeze(0)                     # (1, L, D)


def sinusoidal_pe_at(pos, d_model: int, dtype):
    """任意の整数位置 pos:(B,L) に対する位置符号 → (B,L,D)。

    相対位置符号用。絶対 index ではなく「読み出し位置からの距離」を渡すことで、
    過去スロット数 Np（バッチ内の最大過去長）が変わっても符号がずれなくなる。
    """
    i = torch.arange(0, d_model, 2, device=pos.device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    ang = pos.float().unsqueeze(-1) * div                # (B,L,D/2)
    pe = torch.zeros(*pos.shape, d_model, device=pos.device, dtype=torch.float32)
    pe[..., 0::2] = torch.sin(ang)
    pe[..., 1::2] = torch.cos(ang)[..., : pe[..., 1::2].shape[-1]]
    return pe.to(dtype)                                  # (B, L, D)


class TurnTakingXfmrHead(nn.Module):
    """[過去 max-pool ベクトル ; 現発話 encoder 出力] を混ぜ、**現発話の最終フレーム位置**→ MLP。

    head_type:
      "selfattn"    : **self-attention（multi-head）1 層のみ**（FFN・残差・LayerNorm なし）★既定
      "transformer" : self-attention + FFN + 残差 + LayerNorm を積んだ Transformer エンコーダ
    """

    def __init__(self, d_model=256, head_type="selfattn", n_layers=1, n_heads=4,
                 d_ff=1024, dropout=0.1, n_classes=3, d_hidden=128, rel_pe=False):
        super().__init__()
        self.head_type = head_type
        self.rel_pe = rel_pe
        if head_type == "selfattn":
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
            )
        elif head_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.xfmr = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            raise ValueError(f"unknown head_type: {head_type}")
        # セグメント埋め込み（0=過去ベクトル, 1=現発話フレーム）
        self.seg = nn.Embedding(2, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, past_vec, past_lens, enc_out, enc_lens):
        """past_vec:(B,Np,D) padded / past_lens:(B,) / enc_out:(B,T,D) / enc_lens:(B,)"""
        B, Np, D = past_vec.shape
        T = enc_out.shape[1]
        seq = torch.cat([past_vec, enc_out], dim=1)                 # (B, Np+T, D)

        # セグメント埋め込み（過去 v / 現発話 c の区別）＋ 位置符号
        seg_id = torch.cat([
            torch.zeros(B, Np, dtype=torch.long, device=seq.device),
            torch.ones(B, T, dtype=torch.long, device=seq.device),
        ], dim=1)
        seq = seq + self.seg(seg_id)
        if self.rel_pe:
            # 相対位置符号：読み出し位置（現発話の最終フレーム）からの距離で符号化する。
            # 絶対 index だと Np がバッチごとに変わり現発話の位置がずれるため。
            readout = (Np + enc_lens.to(seq.device) - 1).clamp(min=0)      # (B,)
            idx = torch.arange(Np + T, device=seq.device)[None, :]
            dist = (readout[:, None] - idx).clamp(min=0)                   # (B, Np+T)
            seq = seq + sinusoidal_pe_at(dist, D, seq.dtype)
        else:
            seq = seq + sinusoidal_pe(Np + T, D, seq.device, seq.dtype)

        # 有効位置マスク（過去は past_lens まで、現発話は enc_lens まで）
        ar_p = torch.arange(Np, device=seq.device)[None, :] < past_lens.to(seq.device)[:, None]
        ar_c = torch.arange(T, device=seq.device)[None, :] < enc_lens.to(seq.device)[:, None]
        valid = torch.cat([ar_p, ar_c], dim=1)                      # (B, Np+T)

        if self.head_type == "selfattn":
            # self-attention のみ（Q=K=V=seq）。FFN・残差・LayerNorm は入れない。
            out, _ = self.attn(seq, seq, seq, key_padding_mask=~valid, need_weights=False)
        else:
            out = self.xfmr(seq, src_key_padding_mask=~valid)       # (B, Np+T, D)

        # 現発話の最終フレーム位置（現発話は必ず index Np から始まる）
        last = (Np + enc_lens.to(out.device) - 1).clamp(min=0, max=Np + T - 1)
        vec = out[torch.arange(B, device=out.device), last]         # (B, D)
        return self.mlp(vec)                                        # (B, C)


class BottleneckAdapter(nn.Module):
    """Houlsby 型ボトルネック Adapter: down(d→r) → ReLU → up(r→d)。

    up を零初期化するため、学習開始時点では出力が 0 ＝ 元のモデルと完全に同一挙動。
    （LoRA の B=0 初期化と同じ考え方）

    use_ln=False（既定）では**入力に LayerNorm をかけない**。LN を入れると補正量が
    入力の大きさと無関係になり、活性が小さい領域で補正が支配して ASR が壊れる
    （実測 CER 23.3 → 46.7）。LN なしなら補正は入力に比例し、LoRA と同じ性質になる。
    use_ln=True は旧実装の再現用。
    """

    def __init__(self, d_model: int, bottleneck: int = 32, dropout: float = 0.0,
                 use_ln: bool = False, max_ratio: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model) if use_ln else None
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, d_model)
        self.drop = nn.Dropout(dropout)
        self.max_ratio = float(max_ratio)   # >0 で補正量の上限を入力の max_ratio 倍に制限
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.norm(x) if self.norm is not None else x
        out = self.drop(self.up(self.act(self.down(h))))
        if self.max_ratio > 0:
            # 各時刻ごとに ||out|| <= max_ratio * ||x|| へ射影（超えたときだけ縮める）。
            # 学習開始時は up=0 → out=0 なので係数は 1（恒等）で勾配も素直に流れる。
            xn = x.norm(dim=-1, keepdim=True)
            on = out.norm(dim=-1, keepdim=True)
            scale = torch.clamp(self.max_ratio * xn / (on + 1e-6), max=1.0)
            out = out * scale
        return out


class PastAttnPool(nn.Module):
    """過去発話フレーム列を1本のベクトルに要約する attention pooling。

    学習可能なクエリ q との内積で各フレームの重みを softmax で作り、加重平均する。
    max pooling（パラメータ無し・非学習）との比較用。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=0.02)
        self.scale = d_model ** -0.5

    def forward(self, frames, lengths):
        # frames:(B,T,D) padded / lengths:(B,)
        B, T, D = frames.shape
        score = (frames @ self.query) * self.scale               # (B,T)
        mask = torch.arange(T, device=frames.device)[None, :] >= lengths[:, None]
        score = score.masked_fill(mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)            # (B,T,1)
        return (w * frames).sum(dim=1)                            # (B,D)


class PastCompressor(nn.Module):
    """【固定率・soft】Compressive Transformer (Rae+ ICLR2020) 型の過去圧縮。

    過去フレーム列 (B,Tp,D) を圧縮率 c ごとのブロックにまとめ、(B,ceil(Tp/c),D) にする。
    Transformer-XL が過去を「キャッシュしてそのまま繋ぐ」のに対し、こちらはその先で
    古い状態を **固定率で圧縮した記憶スロット** に落とす（＝出力長が過去長に比例）。

    既存の pooling は本モジュールの両端に対応する:
        c = 1   → 圧縮なし（tt_past_pool="frames" と同じ）
        c >= Tp → 1 本に潰す（tt_past_pool="max"/"attn" と同じ粒度）

    fn は圧縮関数（論文の compression function に相当）:
        "conv" : kernel=stride=c の 1D 畳み込み（学習あり）★既定
        "mean" : ブロック平均（非学習）
        "max"  : ブロック最大（非学習）
    """

    def __init__(self, d_model: int, ratio: int, fn: str = "conv"):
        super().__init__()
        if ratio < 1:
            raise ValueError(f"tt_compress_ratio は 1 以上: {ratio}")
        if fn not in ("conv", "mean", "max"):
            raise ValueError(f"unknown tt_compress_fn: {fn}")
        self.ratio = ratio
        self.fn = fn
        if fn == "conv":
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=ratio, stride=ratio)

    def forward(self, frames, lengths):
        """frames:(B,Tp,D) padded / lengths:(B,) → (B,ceil(Tp/c),D), (B,)"""
        B, T, D = frames.shape
        c = self.ratio
        lengths = lengths.to(frames.device)
        if c == 1:
            return frames, lengths

        # パディング位置は 0 にしてから圧縮する（無効フレームを混ぜない）
        valid = torch.arange(T, device=frames.device)[None, :] < lengths[:, None]
        x = frames * valid.unsqueeze(-1)
        pad = (c - T % c) % c
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            valid = F.pad(valid, (0, pad))
        out_lens = (lengths + c - 1) // c

        if self.fn == "conv":
            out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        elif self.fn == "mean":
            cnt = valid.view(B, -1, c).sum(-1).clamp(min=1).to(x.dtype)
            out = x.view(B, -1, c, D).sum(dim=2) / cnt.unsqueeze(-1)
        else:
            out = x.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            out = out.view(B, -1, c, D).max(dim=2).values
            out = torch.nan_to_num(out, neginf=0.0)   # 全パディングのブロック → 0
        return out, out_lens


class PastQueryPool(nn.Module):
    """【固定長・soft】学習クエリ M 本で過去を要約する（出力は常に M スロット）。

    Set Transformer の PMA (Lee+ ICML2019) / Perceiver の latent bottleneck
    (Jaegle+ ICML2021) と同じ機構。学習された M 本のクエリが過去フレーム列に
    cross-attention し、過去がどれだけ長くても固定サイズ M に落とす。

    既存の attention pooling (PastAttnPool) は **M=1・単一ヘッド** の特殊ケース。
    M を振ることで「1 本では足りないのか」を直接検証できる。
    固定率の PastCompressor と違い、**過去長が伸びてもスロット数が増えない**のが対照点。
    """

    def __init__(self, d_model: int, slots: int, n_heads: int = 4):
        super().__init__()
        if slots < 1:
            raise ValueError(f"tt_compress_slots は 1 以上: {slots}")
        self.slots = slots
        self.query = nn.Parameter(torch.zeros(slots, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, frames, lengths):
        B, T, D = frames.shape
        lengths = lengths.to(frames.device).clamp(min=1, max=T)   # 全マスク→NaN を防ぐ
        q = self.query.unsqueeze(0).expand(B, -1, -1).to(frames.dtype)
        pad = torch.arange(T, device=frames.device)[None, :] >= lengths[:, None]
        out, _ = self.attn(q, frames, frames, key_padding_mask=pad, need_weights=False)
        out_lens = torch.full((B,), self.slots, dtype=torch.long, device=frames.device)
        return out, out_lens


class PastTopKPool(nn.Module):
    """【固定長・hard】重要度上位 M フレームだけを残す（選択型の圧縮）。

    Compressive Transformer 論文の "most-used"（注意質量の大きい状態を残し、
    残りを捨てる）に相当する hard compression。上の 2 つが元の値を混ぜ合わせる
    soft compression なのに対し、こちらは **元フレームをそのまま残す**ので、
    「混ぜて要約する」のと「選んで残す」のどちらが効くかを切り分けられる。

    選択自体は微分できないため、選ばれたフレームに重要度 sigmoid を掛けて
    スコア関数に勾配を通す。時系列順は保つ。
    """

    def __init__(self, d_model: int, slots: int):
        super().__init__()
        if slots < 1:
            raise ValueError(f"tt_compress_slots は 1 以上: {slots}")
        self.slots = slots
        self.scorer = nn.Linear(d_model, 1)

    def forward(self, frames, lengths):
        B, T, D = frames.shape
        lengths = lengths.to(frames.device)
        m = min(self.slots, T)
        score = self.scorer(frames).squeeze(-1)                   # (B,T)
        pad = torch.arange(T, device=frames.device)[None, :] >= lengths[:, None]
        score = score.masked_fill(pad, float("-inf"))
        idx = score.topk(m, dim=1).indices
        idx, _ = idx.sort(dim=1)                                  # 時系列順に戻す
        out = frames.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        w = torch.sigmoid(score.gather(1, idx)).unsqueeze(-1)     # 無効位置は 0
        out = out * w
        # 無効フレームは常に有効フレームより後ろの index なので、ソート後は末尾に来る
        out_lens = lengths.clamp(max=m)
        return out, out_lens


class PastUttPool(nn.Module):
    """【発話単位】過去発話ごとに k スロットへ要約する（中間審査スライドの構造）。

    past_speech は過去N発話を連結した1本の波形なので、符号化後は発話の切れ目が
    失われている。past_bounds（連結順の各発話サンプル数）から境界フレームを復元し、
    **発話ごとに** pooling する。k=1 なら出力はスライドどおり V_1..V_N の N 本。

    上の 3 手法が固定率・固定長という「音響的に無意味な区切り」なのに対し、
    こちらは会話の単位で区切る点が対照になる。
    """

    def __init__(self, d_model: int, per_utt: int = 1, fn: str = "max"):
        super().__init__()
        if per_utt < 1:
            raise ValueError(f"tt_utt_slots は 1 以上: {per_utt}")
        if fn not in ("max", "mean", "attn"):
            raise ValueError(f"unknown tt_compress_fn for utt: {fn}")
        self.per_utt = per_utt
        self.fn = fn
        if fn == "attn":
            self.pool = PastQueryPool(d_model, per_utt, n_heads=4)

    def forward(self, frames, lengths, bounds, bound_lengths):
        """frames:(B,Tp,D) / lengths:(B,) / bounds:(B,Nmax) 各過去発話のサンプル数"""
        B, T, D = frames.shape
        dev = frames.device
        lengths = lengths.to(dev)
        bounds = bounds.to(dev).clamp(min=0).float()
        nmax = bounds.shape[1]

        # サンプル数の累積比 → フレーム境界（フレームレートは一様なので比例配分でよい）
        total = bounds.sum(dim=1, keepdim=True).clamp(min=1.0)
        cum = bounds.cumsum(dim=1) / total                          # (B,Nmax) 0<..<=1
        end = (cum * lengths[:, None].float()).round().long()       # 各発話の終端フレーム
        start = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev),
                           end[:, :-1]], dim=1)

        # 有効な過去発話（サンプル数>0 かつ 1フレーム以上）
        valid_utt = (bounds > 0) & (end > start)
        if bound_lengths is not None:
            ar = torch.arange(nmax, device=dev)[None, :]
            valid_utt &= ar < bound_lengths.to(dev)[:, None]

        # 発話 u のフレーム集合マスク: start_u <= t < end_u
        ar_t = torch.arange(T, device=dev)[None, None, :]           # (1,1,T)
        m = (ar_t >= start[:, :, None]) & (ar_t < end[:, :, None])  # (B,Nmax,T)
        m &= valid_utt[:, :, None]

        outs = []
        for k in range(self.per_utt):
            if self.per_utt == 1:
                sub = m
            else:
                # 発話内をさらに k 等分（per_utt>1 のとき）
                pos = ar_t - start[:, :, None]
                span = (end - start).clamp(min=1)[:, :, None]
                sub = m & ((pos * self.per_utt) // span == k)
            if self.fn == "mean":
                cnt = sub.sum(-1, keepdim=True).clamp(min=1).to(frames.dtype)
                o = torch.einsum("but,btd->bud", sub.to(frames.dtype), frames) / cnt
            else:   # max（attn も発話内 max で代表させ、後段で query pooling する）
                o = frames[:, None].masked_fill(~sub[:, :, :, None], float("-inf"))
                o = torch.nan_to_num(o.max(dim=2).values, neginf=0.0)
            outs.append(o)
        # (B, Nmax, per_utt, D) → 発話順に並べて (B, Nmax*per_utt, D)
        out = torch.stack(outs, dim=2).reshape(B, nmax * self.per_utt, D)
        out = out * valid_utt.repeat_interleave(self.per_utt, dim=1)[:, :, None]

        out_lens = valid_utt.sum(dim=1) * self.per_utt
        # 有効発話が先頭に詰まっている前提（bounds は連結順・パディングは末尾）
        return out, out_lens.clamp(min=0)


class TurnTakingXfmrASRModel(ESPnetASRModel):
    """凍結エンコーダ ＋ Transformer で発話区間末を予測（Stage2）。ASR は学習しない。"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: Optional[dict] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        # --- Stage2 固有 ---
        num_tag_classes: int = 3,
        tag_class_weight: Optional[List[float]] = None,
        tt_head_type: str = "selfattn",   # "selfattn"（self-attention 1層のみ）| "transformer"
        tt_xfmr_layers: int = 1,          # transformer 選択時の層数
        tt_xfmr_heads: int = 4,           # self-attention のヘッド数
        tt_xfmr_ff: int = 1024,           # transformer 選択時の FFN 次元
        tt_dropout: float = 0.1,
        tt_d_hidden: int = 128,
        report_turntaking_accuracy: bool = True,
        # --- エンコーダ部分適応（PEFT）。"none"=完全凍結（既定） / "lora" / "adapter" ---
        # "lora"   : LoRA の挿入は use_adapter+adapter_conf（abs_task）で行う。
        # "adapter": 本クラスが各 Conformer ブロックの FFN 出力に Bottleneck Adapter を
        #            forward hook で加算する（エンコーダのパラメータ名は変えない）。
        # いずれもエンコーダを勾配ありで通す（＝適応パラメータに勾配が流れる）。
        tt_adapt: str = "none",
        # 過去発話の要約方法。"none"=事前計算 past_vec を使う（既定・従来）／
        # "max"/"attn"=過去音声 past_speech を凍結エンコーダで符号化し max/attn pooling で1本に要約／
        # "frames"=過去音声を符号化し **pooling せず全フレーム**を self-attention に入れる（圧縮しない要約）／
        # --- ここから圧縮手法（過去音声を凍結エンコーダで符号化した後の要約方法）---
        # "conv" =【固定率・soft】Compressive Transformer 型。ceil(Tp/ratio) スロット。
        # "query"=【固定長・soft】学習クエリ M 本による PMA/Perceiver 型。常に slots スロット。
        # "topk" =【固定長・hard】重要度上位 M フレームを選択。常に slots スロット。
        # "utt"  =【発話単位】past_bounds から境界を復元し発話ごとに要約（スライドの構造）。
        tt_past_pool: str = "none",
        # tt_past_pool="utt" の 1 発話あたりスロット数（1 なら V_1..V_N と同じ）。
        tt_utt_slots: int = 1,
        # tt_past_pool="conv" の圧縮率 c（過去 Tp フレーム → ceil(Tp/c) スロット）。
        tt_compress_ratio: int = 8,
        # 同じく圧縮関数。"conv"（学習あり）/ "mean" / "max"（非学習）。
        tt_compress_fn: str = "conv",
        # tt_past_pool="query"/"topk" のスロット数 M（過去長に依らず固定）。
        tt_compress_slots: int = 8,
        # True で head の位置符号を **相対**（読み出し位置からの距離）にする。
        # False（既定）は従来の絶対位置符号＝過去長 Np で現発話の位置がずれる。
        tt_rel_pe: bool = False,
        # False で **過去音声には SpecAugment を掛けない**（記憶を学習/評価で揃える）。
        # True（既定）は従来どおり過去音声にも増強が掛かる。
        tt_past_specaug: bool = True,
        # True で過去発話を **テキスト**（past_text のトークン列）で入れる。
        # Stage1 の Transducer デコーダ埋め込み（凍結）で埋め込み → 256次元へ射影して head に渡す。
        tt_past_text: bool = False,
        tt_adapter_bottleneck: int = 32,
        tt_adapter_dropout: float = 0.0,
        # Adapter を挿す FFN。既定は各ブロックの主 FFN のみ（12 箇所, Pfeiffer 型）。
        # ["feed_forward_macaron", "feed_forward"] にすると旧実装と同じ 24 箇所。
        tt_adapter_targets: List[str] = ["feed_forward"],
        tt_adapter_ln: bool = False,   # True で旧実装（入力 LayerNorm あり）を再現
        # 補正量の上限。>0 で「補正の大きさ <= max_ratio × 入力の大きさ」に制限する。
        # 0 は無制限（＝上限なし）。適応の強さを直接調整するつまみ。
        tt_adapter_max_ratio: float = 0.0,
    ):
        super().__init__(
            vocab_size=vocab_size, token_list=token_list, frontend=frontend,
            specaug=specaug, normalize=normalize, preencoder=preencoder,
            encoder=encoder, postencoder=postencoder, decoder=decoder, ctc=ctc,
            joint_network=joint_network, aux_ctc=aux_ctc, ctc_weight=ctc_weight,
            interctc_weight=interctc_weight, ignore_id=ignore_id, lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss, report_cer=report_cer,
            report_wer=report_wer, sym_space=sym_space, sym_blank=sym_blank,
            transducer_multi_blank_durations=transducer_multi_blank_durations,
            transducer_multi_blank_sigma=transducer_multi_blank_sigma,
            sym_sos=sym_sos, sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )
        self.num_tag_classes = num_tag_classes
        self.report_turntaking_accuracy = report_turntaking_accuracy
        self.tt_adapt = tt_adapt  # "none" | "lora" | "houlsby"
        if tag_class_weight is not None:
            self.register_buffer(
                "tag_class_weight", torch.tensor(tag_class_weight, dtype=torch.float)
            )
        else:
            self.tag_class_weight = None

        self.turntaking_head = TurnTakingXfmrHead(
            d_model=encoder.output_size(), head_type=tt_head_type,
            n_layers=tt_xfmr_layers, n_heads=tt_xfmr_heads, d_ff=tt_xfmr_ff,
            dropout=tt_dropout, n_classes=num_tag_classes, d_hidden=tt_d_hidden,
            rel_pe=tt_rel_pe,
        )
        self.tt_past_specaug = tt_past_specaug
        if tt_rel_pe:
            logging.info("[turntaking_xfmr] head 位置符号 = 相対（読み出し位置からの距離）")
        if not tt_past_specaug:
            logging.info("[turntaking_xfmr] 過去音声には SpecAugment を掛けない")

        self.tt_past_pool = tt_past_pool
        if tt_past_pool == "attn":
            self.past_attn_pool = PastAttnPool(encoder.output_size())
            logging.info("[turntaking_xfmr] 過去要約 = attention pooling（学習あり）")
        elif tt_past_pool == "max":
            logging.info("[turntaking_xfmr] 過去要約 = max pooling（過去音声から・非学習）")
        elif tt_past_pool == "frames":
            logging.info("[turntaking_xfmr] 過去 = 全フレーム（pooling なし）")
        elif tt_past_pool == "conv":
            self.past_compressor = PastCompressor(
                encoder.output_size(), tt_compress_ratio, tt_compress_fn
            )
            logging.info(f"[turntaking_xfmr] 過去要約 = Compressive 型圧縮（固定率・soft）"
                         f"(ratio={tt_compress_ratio}, fn={tt_compress_fn})")
        elif tt_past_pool == "query":
            self.past_compressor = PastQueryPool(
                encoder.output_size(), tt_compress_slots, tt_xfmr_heads
            )
            logging.info(f"[turntaking_xfmr] 過去要約 = 学習クエリ PMA/Perceiver 型"
                         f"（固定長・soft）(slots={tt_compress_slots})")
        elif tt_past_pool == "topk":
            self.past_compressor = PastTopKPool(
                encoder.output_size(), tt_compress_slots
            )
            logging.info(f"[turntaking_xfmr] 過去要約 = 重要度 top-k 選択（固定長・hard）"
                         f"(slots={tt_compress_slots})")
        elif tt_past_pool == "utt":
            self.past_compressor = PastUttPool(
                encoder.output_size(), tt_utt_slots, tt_compress_fn
                if tt_compress_fn in ("max", "mean", "attn") else "max"
            )
            logging.info(f"[turntaking_xfmr] 過去要約 = 発話単位（境界復元）"
                         f"(per_utt={tt_utt_slots}, fn={tt_compress_fn})")

        self.tt_past_text = tt_past_text
        if tt_past_text:
            emb_dim = self.decoder.embed.embedding_dim   # Transducer デコーダ埋め込み（凍結）
            self.past_text_proj = nn.Linear(emb_dim, encoder.output_size())
            logging.info(f"[turntaking_xfmr] 過去 = テキスト（decoder.embed {emb_dim}次元→"
                         f"{encoder.output_size()}次元 射影・埋め込みは凍結）")

        if tt_adapt == "adapter":
            n = self._install_encoder_adapters(
                tt_adapter_bottleneck, tt_adapter_dropout,
                tt_adapter_targets, tt_adapter_ln, tt_adapter_max_ratio,
            )
            logging.info(f"[turntaking_xfmr] Bottleneck Adapter を {n} 箇所に挿入 "
                         f"(bottleneck={tt_adapter_bottleneck}, targets={tt_adapter_targets}, "
                         f"input_ln={tt_adapter_ln}, max_ratio={tt_adapter_max_ratio})")

    # ------------------------------------------------------------------
    def _install_encoder_adapters(self, bottleneck: int, dropout: float,
                                  targets: List[str], use_ln: bool,
                                  max_ratio: float = 0.0) -> int:
        """指定した FFN の出力に Adapter を forward hook で加算する。

        モジュールを差し替えず hook で足すため、**エンコーダのパラメータ名は不変**。
        そのため Stage1 チェックポイントがそのまま読め、freeze_param: encoder も
        従来どおり base 重みだけを凍結できる（Adapter は encoder_adapters 配下＝学習対象）。
        """
        d = self.encoder.output_size()
        adapters = []

        def _make_hook(adapter):
            def _hook(module, inputs, output):
                return output + adapter(output)
            return _hook

        for layer in self.encoder.encoders:
            for name in targets:
                base = getattr(layer, name, None)
                if base is None:
                    continue
                ad = BottleneckAdapter(d, bottleneck, dropout, use_ln=use_ln,
                                       max_ratio=max_ratio)
                base.register_forward_hook(_make_hook(ad))
                adapters.append(ad)

        if not adapters:
            raise ValueError(f"Adapter の挿入先が見つかりません: targets={targets}")
        self.encoder_adapters = nn.ModuleList(adapters)
        return len(adapters)

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """学習モードにしても、凍結部は常に eval のままにする。

        freeze_param で勾配は止まるが、**BatchNorm の running 統計は forward で更新**されて
        しまう（勾配とは無関係）。凍結部を eval に固定して、ASR をビット単位で不変にする。
        SpecAug は self.training（モデル本体）を見るので、学習時の増強はそのまま効く。
        """
        super().train(mode)
        # 常に eval に固定する凍結部（ASR 側）
        for m in (self.frontend, self.normalize, self.decoder,
                  getattr(self, "joint_network", None), getattr(self, "ctc", None)):
            if m is not None:
                m.eval()
        if self.tt_adapt == "none":
            # 完全凍結：エンコーダも eval（BN 統計も止め、ASR をビット単位で不変にする）
            self.encoder.eval()
        else:
            # 部分適応：LoRA/Adapter に勾配を流すためエンコーダは train のまま。
            # ただし BatchNorm の running 統計だけは更新させない（適応対象を PEFT に限定）。
            from torch.nn.modules.batchnorm import _BatchNorm
            for sub in self.encoder.modules():
                if isinstance(sub, _BatchNorm):
                    sub.eval()
        return self

    def _frozen_encode(self, speech, speech_lengths):
        """エンコーダ出力を得る。

        tt_adapt=="none": 勾配も BN 統計も更新しない＝ASR は数値的に完全不変。
        それ以外(PEFT): エンコーダを勾配ありで通す（LoRA/Adapter に勾配を流す）。
        いずれも base の重みは freeze_param で凍結済み。
        """
        if self.tt_adapt == "none":
            with torch.no_grad():
                enc_out, enc_lens = self.encode(speech, speech_lengths)
                if isinstance(enc_out, tuple):
                    enc_out = enc_out[0]
                return enc_out.detach(), enc_lens.detach()
        enc_out, enc_lens = self.encode(speech, speech_lengths)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]
        return enc_out, enc_lens

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        tag_label: Optional[torch.Tensor] = None,
        tag_label_lengths: Optional[torch.Tensor] = None,
        past_vec: Optional[torch.Tensor] = None,
        past_vec_lengths: Optional[torch.Tensor] = None,
        past_speech: Optional[torch.Tensor] = None,
        past_speech_lengths: Optional[torch.Tensor] = None,
        past_text: Optional[torch.Tensor] = None,
        past_text_lengths: Optional[torch.Tensor] = None,
        past_bounds: Optional[torch.Tensor] = None,
        past_bounds_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = speech.shape[0]

        # 1. 凍結エンコーダ（現発話・pooling しない）
        enc_out, enc_lens = self._frozen_encode(speech, speech_lengths)
        B, T, D = enc_out.shape

        # 2a. 過去音声を凍結エンコーダで符号化して過去表現を作る（pooling / 全フレーム）
        if (self.tt_past_pool in ("max", "attn", "frames", "conv", "query", "topk", "utt")
                and past_speech is not None):
            with torch.no_grad():
                # encode() は self.training を見て SpecAug を掛ける。過去は「記憶」なので、
                # tt_past_specaug=False のときだけ一時的に外して符号化する（現発話には掛かる）。
                sa = self.specaug
                if not self.tt_past_specaug:
                    self.specaug = None
                try:
                    p_out, p_lens = self.encode(past_speech, past_speech_lengths)
                finally:
                    self.specaug = sa
                if isinstance(p_out, tuple):
                    p_out = p_out[0]
                p_out = p_out.detach()
            p_lens = p_lens.to(p_out.device)
            if self.tt_past_pool == "frames":
                # pooling なし：過去の全フレームをそのまま渡す
                past_vec = p_out
                past_vec_lengths = p_lens
            elif self.tt_past_pool in ("conv", "query", "topk"):
                # 記憶スロット列に圧縮する（conv は可変長、query/topk は固定長）
                past_vec, past_vec_lengths = self.past_compressor(p_out, p_lens)
            elif self.tt_past_pool == "utt":
                if past_bounds is None:
                    raise RuntimeError(
                        "tt_past_pool='utt' には past_bounds.scp が必要です。"
                        "local/build_past_audio.py を再実行して境界を生成してください。"
                    )
                past_vec, past_vec_lengths = self.past_compressor(
                    p_out, p_lens, past_bounds, past_bounds_lengths
                )
            else:
                if self.tt_past_pool == "max":
                    pmask = torch.arange(p_out.shape[1], device=p_out.device)[None, :] \
                        >= p_lens[:, None]
                    pooled = p_out.masked_fill(pmask.unsqueeze(-1), float("-inf")).max(dim=1).values
                else:
                    pooled = self.past_attn_pool(p_out, p_lens)
                past_vec = pooled.unsqueeze(1)                    # (B,1,D)
                past_vec_lengths = torch.ones(B, dtype=torch.long, device=enc_out.device)

        # 2b. 過去テキスト：凍結デコーダ埋め込み → 射影 → past_vec 列にする
        if self.tt_past_text and past_text is not None:
            tok = past_text.clamp(min=0).long()                  # (B, L) パディングは 0 扱い
            with torch.no_grad():
                emb = self.decoder.embed(tok).detach()            # (B, L, 512)
            past_vec = self.past_text_proj(emb.to(enc_out.dtype)) # (B, L, D)
            past_vec_lengths = past_text_lengths

        # 2. 過去ベクトル（事前計算の max-pool 済み・可変本数）
        if past_vec is None:
            # N=0（過去発話なし）。ダミートークンを挿さず長さ 0 で渡すため、
            # self-attention は現発話フレームのみを見る（位置符号のずれも生じない）。
            past_vec = enc_out.new_zeros(B, 0, D)
            past_lens = torch.zeros(B, dtype=torch.long, device=enc_out.device)
        else:
            past_vec = past_vec.to(enc_out.dtype)
            if past_vec.dim() == 2:                    # (B,D) → (B,1,D)
                past_vec = past_vec.unsqueeze(1)
            past_lens = (
                past_vec_lengths
                if past_vec_lengths is not None
                else torch.full((B,), past_vec.shape[1], dtype=torch.long,
                                device=enc_out.device)
            )

        # 3. Transformer → 現発話の最終フレーム位置 → MLP
        tag_logits = self.turntaking_head(past_vec, past_lens, enc_out, enc_lens)

        stats: Dict[str, torch.Tensor] = {}
        loss_tag = torch.tensor(0.0, device=enc_out.device)
        tag_acc = 0.0
        if tag_label is not None:
            label = (tag_label[:, 0] if tag_label.dim() == 2 else tag_label).long()
            loss_tag = F.cross_entropy(tag_logits, label, weight=self.tag_class_weight)
            tag_acc = (tag_logits.argmax(-1) == label).float().mean().item()

        # 4. ASR は凍結済み＝学習しない（総損失は区間末のみ）
        loss = loss_tag
        stats["loss_tag"] = loss_tag.detach()
        if self.report_turntaking_accuracy:
            stats["tag_acc"] = tag_acc
        stats["loss"] = loss.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    # ------------------------------------------------------------------
    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}
