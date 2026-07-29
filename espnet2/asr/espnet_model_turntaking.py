"""マルチタスク turn-taking（発話区間末予測）付き ASR モデル（utterance-level 版）。

タスク: **発話を最後まで聞いて**、その区間の発話区間末タグ（0=<no>/1=<yes>/2=<other>）を
1 区間につき 1 つ予測する。早期確定・未来VA・時間重み等は使わない（素直な分類）。

    音声 → 共有 encoder（ASR と同時学習）→ encoder_out (B,T,256)
              │                                  │
              ▼                                  ▼
       transducer デコーダ（ASR）        発話区間末ヘッド
        → loss_transducer                 encoder_out を発話全体で平均プール
                                          (+ 任意で過去文脈ベクトル ctx_vec を concat)
                                          → MLP → 発話区間末タグ確率 (B,3)
    総損失  L = loss_asr + turntaking_weight · loss_tag      （loss_tag = クラス重み付き CE）

ctx_dim>0 のとき、直前N発話の文脈ベクトル ctx_vec を head に与えて条件付ける（ASR は不変）。
"""
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


class TurnTakingHead(nn.Module):
    """発話全体を聞いて区間末タグを 1 つ予測する utterance-level 分類ヘッド。

    encoder_out を有効フレームで平均プール → (任意で ctx を concat) → MLP → n_classes。
    """

    def __init__(self, d_in=256, d_hidden=128, n_classes=3, ctx_dim=0, dropout=0.1):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_in + ctx_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, encoder_out, encoder_out_lens, ctx=None):
        # 発話全体の平均プール（有効フレームのみ）
        B, T, _ = encoder_out.shape
        mask = (
            torch.arange(T, device=encoder_out.device)[None, :]
            < encoder_out_lens.to(encoder_out.device)[:, None]
        ).float().unsqueeze(-1)                                  # (B,T,1)
        pooled = (encoder_out * mask).sum(1) / mask.sum(1).clamp(min=1.0)   # (B,D)
        if self.ctx_dim > 0:
            if ctx is None:
                ctx = pooled.new_zeros(B, self.ctx_dim)
            pooled = torch.cat([pooled, ctx], dim=-1)
        return self.mlp(pooled)                                  # (B, n_classes)


class TurnTakingESPnetASRModel(ESPnetASRModel):
    """transducer ASR と発話区間末分類を encoder 共有で同時学習するモデル（utterance-level）。"""

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
        # --- turn-taking 固有パラメータ（yaml の model_conf から渡す）---
        num_tag_classes: int = 3,          # 0=<no>, 1=<yes>, 2=<other>
        turntaking_weight: float = 0.3,    # α: ASR 優先で小さめから
        tt_d_hidden: int = 128,
        tt_dropout: float = 0.1,
        # クラス重み: None で均一、[w_no, w_yes, w_other] で指定（<no> が少ない場合に上げる）
        tag_class_weight: Optional[List[float]] = None,
        report_turntaking_accuracy: bool = True,
        ctx_dim: int = 0,                  # 過去文脈ベクトルの次元（0=使わない）
        # --- seqcat 方式（過去音声を現発話の前に連結してエンコーダに通す）---
        context_mode: str = "ctxvec",      # "ctxvec"（従来: 平均プール+ctx）| "seqcat"（連結→pool）
        tt_pool: str = "mean",             # seqcat時のプール: "max" | "mean"
        tt_pool_range: str = "current",    # seqcat時のプール範囲: "whole"（過去+現発話全体）| "current"（現発話のみ）
        # --- 早期確定(TEASER)用: プレフィックス学習（どの長さでも当たる予測器にする）---
        tag_prefix_train: bool = False,    # 学習時に区間末ヘッドをランダムなプレフィックス長でプール
        tag_prefix_min: float = 0.3,       # プレフィックス最小割合（[min,1.0]×T からサンプル）
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            transducer_multi_blank_durations=transducer_multi_blank_durations,
            transducer_multi_blank_sigma=transducer_multi_blank_sigma,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        assert self.use_transducer_decoder, (
            "TurnTakingESPnetASRModel は transducer ASR を前提とする"
            "（decoder: transducer / joint_network 必須）"
        )

        self.num_tag_classes = num_tag_classes
        self.turntaking_weight = turntaking_weight
        self.report_turntaking_accuracy = report_turntaking_accuracy
        self.ctx_dim = ctx_dim
        self.context_mode = context_mode
        self.tt_pool = tt_pool
        self.tt_pool_range = tt_pool_range
        self.tag_prefix_train = tag_prefix_train
        self.tag_prefix_min = tag_prefix_min

        if tag_class_weight is not None:
            assert len(tag_class_weight) == num_tag_classes, (
                len(tag_class_weight), num_tag_classes,
            )
            self.register_buffer(
                "tag_class_weight", torch.tensor(tag_class_weight, dtype=torch.float)
            )
        else:
            self.tag_class_weight = None

        self.turntaking_head = TurnTakingHead(
            d_in=encoder.output_size(),
            d_hidden=tt_d_hidden,
            n_classes=num_tag_classes,
            ctx_dim=ctx_dim,
            dropout=tt_dropout,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        tag_label: Optional[torch.Tensor] = None,
        tag_label_lengths: Optional[torch.Tensor] = None,
        ctx_vec: Optional[torch.Tensor] = None,
        ctx_vec_lengths: Optional[torch.Tensor] = None,
        past_speech: Optional[torch.Tensor] = None,
        past_speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0] == speech_lengths.shape[0]
            == text.shape[0] == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id
        text = text[:, : text_lengths.max()]

        # 1. Encoder（1 回だけ）
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        stats: Dict[str, torch.Tensor] = {}

        # 2. ASR 損失（CTC は weight>0 のときのみ／transducer は常時）
        loss_ctc = None
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            stats["loss_ctc"] = loss_ctc.detach()
            stats["cer_ctc"] = cer_ctc

        loss_transducer, cer_transducer, wer_transducer = self._calc_transducer_loss(
            encoder_out, encoder_out_lens, text
        )
        if loss_ctc is not None:
            loss_asr = loss_transducer + self.ctc_weight * loss_ctc
        else:
            loss_asr = loss_transducer
        stats["loss_transducer"] = loss_transducer.detach()
        stats["cer_transducer"] = cer_transducer
        stats["wer_transducer"] = wer_transducer

        # 3. 発話区間末分類（発話全体プール → 1 区間 1 予測）
        if self.context_mode == "seqcat":
            # [過去発話 + 現発話] を連結してエンコーダに通し、現発話フレームのみをプール
            tag_logits = self._seqcat_head(
                speech, speech_lengths, encoder_out_lens,
                past_speech, past_speech_lengths,
            )                                                                    # (B, C)
        else:
            ctx = ctx_vec.to(encoder_out.dtype) if (self.ctx_dim > 0 and ctx_vec is not None) else None
            tag_lens = encoder_out_lens
            if self.training and self.tag_prefix_train:
                # 各サンプルで [min,1.0]×有効長 のランダムなプレフィックス長でプール（ASR はフルで不変）
                r = torch.rand_like(encoder_out_lens.float()) * (1.0 - self.tag_prefix_min) + self.tag_prefix_min
                tag_lens = (encoder_out_lens.float() * r).long().clamp(min=1)
            tag_logits = self.turntaking_head(encoder_out, tag_lens, ctx)   # (B, C)

        loss_tag = torch.tensor(0.0, device=encoder_out.device)
        tag_acc = 0.0
        if tag_label is not None:
            label = (tag_label[:, 0] if tag_label.dim() == 2 else tag_label).long()
            loss_tag = F.cross_entropy(tag_logits, label, weight=self.tag_class_weight)
            tag_acc = (tag_logits.argmax(-1) == label).float().mean().item()

        # 4. 総損失
        loss = loss_asr + self.turntaking_weight * loss_tag
        stats["loss_tag"] = loss_tag.detach()
        if self.report_turntaking_accuracy:
            stats["tag_acc"] = tag_acc
        stats["loss"] = loss.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    # ------------------------------------------------------------------
    def _concat_wave(self, past, past_len, cur, cur_len):
        """波形レベルで [past ; cur] を各バッチ要素ごとに連結（現発話が最後）。"""
        if past.dim() == 3:      # (B,T,1) → (B,T)
            past = past.squeeze(-1)
        if cur.dim() == 3:
            cur = cur.squeeze(-1)
        B = cur.shape[0]
        total = (past_len + cur_len).long()
        Tmax = int(total.max().item())
        out = cur.new_zeros(B, Tmax)
        for i in range(B):
            pl = int(past_len[i].item())
            cl = int(cur_len[i].item())
            if pl > 0:
                out[i, :pl] = past[i, :pl]
            out[i, pl:pl + cl] = cur[i, :cl]
        return out, total

    def _seqcat_head(self, speech, speech_lengths, cur_enc_lens,
                     past_speech, past_speech_lengths):
        """[過去+現発話] を連結→エンコーダ→現発話フレーム(末尾)をプール→MLP。"""
        if past_speech is not None and past_speech_lengths is not None:
            cat, cat_len = self._concat_wave(
                past_speech, past_speech_lengths, speech, speech_lengths
            )
        else:
            cat, cat_len = speech, speech_lengths
        enc_cat, enc_cat_lens = self.encode(cat, cat_len)
        if isinstance(enc_cat, tuple):
            enc_cat = enc_cat[0]
        B = enc_cat.shape[0]
        vecs = []
        for i in range(B):
            end = int(enc_cat_lens[i].item())
            if self.tt_pool_range == "whole":
                frames = enc_cat[i, :max(1, end)]              # 連結全体（過去+現発話）を pool
            else:
                n_cur = int(cur_enc_lens[i].item())
                start = max(0, end - n_cur)                    # 現発話は連結系列の末尾 n_cur フレーム
                frames = enc_cat[i, start:end] if end > start else enc_cat[i, :max(1, end)]
            vec = frames.max(dim=0).values if self.tt_pool == "max" else frames.mean(dim=0)
            vecs.append(vec)
        vec = torch.stack(vecs, 0)             # (B, D)
        return self.turntaking_head.mlp(vec)   # (B, C)  ※ctx_dim=0 前提

    # ------------------------------------------------------------------
    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        tag_label: Optional[torch.Tensor] = None,
        tag_label_lengths: Optional[torch.Tensor] = None,
        ctx_vec: Optional[torch.Tensor] = None,
        ctx_vec_lengths: Optional[torch.Tensor] = None,
        past_speech: Optional[torch.Tensor] = None,
        past_speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}
