"""CIF-based section tag detection モデル（MultitaskTransformerModel 拡張）。

アーキテクチャ:
    Audio → Encoder → α predictor → cif_function
                                          ↓
                              cif_out (トークンごとの積算音響表現)
                                   ↓                      ↓
                             ASR デコーダ          各 fire の表現
                           (+disfluency)                  ↓
                                                tag_classifier (slave)
                                                      ↓
                                           確率 p_i (C,)
                                                      ↓
                                            SPRT: LLR を累積
                                                      ↓
                                           LLR >= sprt_upper → 早期確定

停止判定（推論時）:
    SPRT（Sequential Probability Ratio Test）を使用。
    fire ごとに対数尤度比（LLR）を累積し、いずれかのクラスの LLR が
    sprt_upper を超えた時点で確定する。oracle ラベルや追加の学習済みヘッドは不要。

データフォーマット（分離後）:
  text     : "uttid ふ ー ん"  (発話のみ, pyscripts/make_tag_split.py で生成)
  tag ファイル: "uttid 2"        (クラスインデックス: 0=<no>, 1=<yes>, 2=<other>)
  → text_int 型でロードされ、forward の tag_label 引数として渡される (shape (B, 1))
  → トークンリストに <no>/<yes>/<other> を含める必要はない
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.multitask_transformer_model import MultitaskTransformerModel

try:
    from torch_cif import cif_function as _cif_function_impl
    _CIF_AVAILABLE = True
except ImportError:
    _cif_function_impl = None
    _CIF_AVAILABLE = False


class CIFTagTransformerModel(MultitaskTransformerModel):
    """CIF 機構を使ったセクションタグ検出付き ASR モデル。

    MultitaskTransformerModel を継承し、エンコーダ出力に CIF を適用する。
    forward は encode を 1 回だけ呼ぶよう実装 (親クラスの super().forward() は
    使わずに全 loss を 1 パスで計算)。

    早期確定には SPRT（Sequential Probability Ratio Test）を使用する。
    追加の学習済みヘッドや oracle ラベルは不要で、tag_classifier の出力確率から
    fire ごとに LLR を累積し、閾値 sprt_upper を超えた時点で確定する。
    """

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend,
        specaug,
        normalize,
        preencoder,
        encoder,
        postencoder,
        decoder,
        ctc,
        # --- MultitaskTransformerModel パラメータ ---
        use_disfluency_detection: bool = True,
        disfluency_classes: int = 4,
        disfluency_weight: float = 1.0,
        ctc_weight: float = 0.3,
        ignore_id: int = -1,
        lsm_weight: float = 0.1,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = False,
        # --- CIF 固有パラメータ ---
        num_tag_classes: int = 3,   # 0=<no>, 1=<yes>, 2=<other>
        cif_weight: float = 0.5,
        tag_weight: float = 1.0,
        cif_threshold: float = 1.0,
        # クラス重み: None で均一、[w_no, w_yes, w_other] で指定
        # 例) <no> が少ない・精度が低い場合は [3.0, 1.0, 1.0] に設定する
        tag_class_weight: Optional[List[float]] = None,
        # --- SPRT パラメータ ---
        sprt_upper: float = 2.0,    # 停止境界。dev set で HM を最大化するよう調整する。
        **kwargs,
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
            use_disfluency_detection=use_disfluency_detection,
            disfluency_classes=disfluency_classes,
            disfluency_weight=disfluency_weight,
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            **kwargs,
        )

        if not _CIF_AVAILABLE:
            raise ImportError(
                "torch-cif が見つかりません。\n"
                "conda activate 0822-espnet && pip install torch-cif を実行してください。"
            )

        self.report_cer = report_cer
        self.report_wer = report_wer

        enc_dim = encoder.output_size()
        self.cif_weight = cif_weight
        self.tag_weight = tag_weight
        self.cif_threshold = cif_threshold
        self.num_tag_classes = num_tag_classes
        self.sprt_upper = sprt_upper

        # クラス重みを buffer として登録（GPU 移動に追従させる）
        if tag_class_weight is not None:
            self.register_buffer(
                "tag_class_weight",
                torch.tensor(tag_class_weight, dtype=torch.float),
            )
        else:
            self.tag_class_weight = None

        # フレームごとの重み α ∈ (0, 1) を予測
        self.alpha_predictor = nn.Sequential(
            nn.Linear(enc_dim, 1),
            nn.Sigmoid(),
        )

        # slave: 各 fire のタグ分類 (0=<no>, 1=<yes>, 2=<other>)
        self.tag_classifier = nn.Sequential(
            nn.Linear(enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_tag_classes),
        )

        logging.info(
            f"CIFTagTransformerModel: num_tag_classes={num_tag_classes}, "
            f"cif_weight={cif_weight}, tag_weight={tag_weight}, "
            f"sprt_upper={sprt_upper}, cif_threshold={cif_threshold}"
        )

    # ------------------------------------------------------------------
    # SPRT ユーティリティ
    # ------------------------------------------------------------------
    @staticmethod
    def _sprt_update(
        probs: torch.Tensor,
        llr: torch.Tensor,
    ) -> torch.Tensor:
        """SPRT の対数尤度比（LLR）を 1 fire 分更新する。

        各クラス k に対して:
            LLR_k += log( p(y=k) / max_{k'≠k} p(y=k') )

        p(y=k) が他クラスより高いほど LLR_k が増加し、
        低いほど LLR_k が減少する（そのクラスへの確信が下がる）。

        Args:
            probs: fire i での予測確率 (C,)
            llr  : 現在の LLR の累積値 (C,)

        Returns:
            更新後の llr (C,)
        """
        C = probs.shape[0]
        # 対角を 0 にして各クラスから自分自身を除外した最大確率を計算
        p_mat = probs.unsqueeze(0).expand(C, C).clone()
        mask = torch.eye(C, dtype=torch.bool, device=probs.device)
        p_mat = p_mat.masked_fill(mask, 0.0)
        p_others_max = p_mat.max(dim=1).values.clamp(min=1e-10)   # (C,)
        return llr + torch.log(probs.clamp(min=1e-10) / p_others_max)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        tag_label: Optional[torch.Tensor] = None,
        tag_label_lengths: Optional[torch.Tensor] = None,
        isdysfl: Optional[torch.Tensor] = None,
        isdysfl_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """学習時の forward。encode は 1 回のみ。

        tag_label が渡された場合はそれを使用（推奨）。
        渡されない場合は None を返す（タグ loss をスキップ）。
        """
        batch_size = speech.shape[0]

        # ======================================================
        # 1. エンコーダ（1 回だけ呼ぶ）
        # ======================================================
        encode_result = self.encode(speech, speech_lengths)
        encoder_out: torch.Tensor = encode_result[0]
        encoder_out_lens: torch.Tensor = encode_result[1]
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # ======================================================
        # 2. CTC loss
        # ======================================================
        loss_ctc = None
        if self.ctc_weight > 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)

        # ======================================================
        # 3. Attention decoder
        # ======================================================
        ys_in_pad, ys_out_pad = self._target_shift(text, self.ignore_id)
        ys_in_pad_safe = ys_in_pad.clamp(min=0)

        if self.use_disfluency_detection and isdysfl is not None:
            isdysfl_clean = isdysfl.clone()
            isdysfl_clean[isdysfl_clean == self.ignore_id] = 0
            disfluency_in_pad, disfluency_out_pad = self._target_shift(
                isdysfl_clean, self.ignore_id
            )
            decoder_out, disfluency_logits = self.decoder(
                hs_pad=encoder_out,
                hlens=encoder_out_lens,
                ys_in_pad=ys_in_pad_safe,
                ys_in_lens=text_lengths,
                disfluency_in_pad=disfluency_in_pad,
                return_disfluency=True,
            )
        else:
            decoder_result = self.decoder(
                hs_pad=encoder_out,
                hlens=encoder_out_lens,
                ys_in_pad=ys_in_pad_safe,
                ys_in_lens=text_lengths,
            )
            decoder_out = decoder_result[0]
            disfluency_logits = None

        loss_att = self.criterion_att(decoder_out, ys_out_pad)

        # ======================================================
        # 4. Disfluency loss
        # ======================================================
        loss_disfluency = None
        if self.use_disfluency_detection and disfluency_logits is not None:
            loss_disfluency = self.criterion_disfluency(
                disfluency_logits.view(-1, self.disfluency_classes),
                disfluency_out_pad.view(-1),
            )

        # ======================================================
        # 5. CIF forward
        # ======================================================
        alpha = self.alpha_predictor(encoder_out).squeeze(-1)  # (B, T)
        B, T = alpha.shape
        pad_mask = (
            torch.arange(T, device=alpha.device).unsqueeze(0) < encoder_out_lens.unsqueeze(1)
        )
        alpha = alpha * pad_mask.float()

        cif_result = _cif_function_impl(
            inputs=encoder_out,
            alpha=alpha,
            target_lengths=text_lengths.float(),
            beta=self.cif_threshold,
        )
        cif_out: torch.Tensor = cif_result["cif_out"][0]      # (B, T_cif, D)
        alpha_sum: torch.Tensor = cif_result["alpha_sum"][0]  # (B,)

        loss_quantity = torch.mean(torch.abs(alpha_sum - text_lengths.float()))

        # ======================================================
        # 6. タグ分類（slave）
        # ======================================================
        resolved_tag_label = self._resolve_tag_label(tag_label, text)

        loss_tag = torch.tensor(0.0, device=speech.device)
        tag_acc = 0.0

        if resolved_tag_label is not None:
            T_cif = cif_out.shape[1]

            # 全 fire の logits (B, T_cif, C)
            tag_logits_all = self.tag_classifier(cif_out)

            # 有効 fire マスク (B, T_cif)
            valid_fire = (
                torch.arange(T_cif, device=cif_out.device).unsqueeze(0)
                < text_lengths.unsqueeze(1)
            )

            tag_label_exp = resolved_tag_label.unsqueeze(1).expand(B, T_cif)
            ce_per_fire = F.cross_entropy(
                tag_logits_all.view(-1, self.num_tag_classes),
                tag_label_exp.reshape(-1),
                weight=self.tag_class_weight,   # None なら均一重み
                reduction='none',
            ).view(B, T_cif)

            # 有効 fire のみの平均（位置重みなし・全 fire を均等に学習）
            n_valid = valid_fire.float().sum().clamp(min=1e-8)
            loss_tag = (ce_per_fire * valid_fire.float()).sum() / n_valid

            # tag_acc: 最後の fire でモニタリング
            last_fire = _get_last_token(cif_out, text_lengths)
            tag_acc = (
                self.tag_classifier(last_fire).argmax(-1) == resolved_tag_label
            ).float().mean().item()

        # ======================================================
        # 7. 合計損失
        # ======================================================
        loss_asr: torch.Tensor = (
            (self.ctc_weight * loss_ctc if loss_ctc is not None else 0.0)
            + (1.0 - self.ctc_weight) * loss_att
        )
        if loss_disfluency is not None:
            loss_asr = loss_asr + self.disfluency_weight * loss_disfluency

        total_loss = (
            loss_asr
            + self.cif_weight * loss_quantity
            + self.tag_weight * loss_tag
        )

        # ======================================================
        # 8. CER/WER（validation 時のみ）
        # ======================================================
        cer, wer = None, None
        if not self.training and (self.report_cer or self.report_wer):
            cer, wer = self._calc_cer_wer(
                encoder_out, encoder_out_lens, text.cpu(), text_lengths
            )

        # ======================================================
        # 9. Stats
        # ======================================================
        stats: Dict[str, torch.Tensor] = {
            "loss": total_loss.detach(),
            "loss_att": loss_att.detach(),
            "loss_quantity": loss_quantity.detach(),
            "loss_tag": (
                loss_tag.detach() if isinstance(loss_tag, torch.Tensor) else loss_tag
            ),
            "tag_acc": tag_acc,
        }
        if loss_ctc is not None:
            stats["loss_ctc"] = loss_ctc.detach()
        if loss_disfluency is not None:
            stats["loss_disfluency"] = loss_disfluency.detach()
        if cer is not None:
            stats["cer"] = cer
        if wer is not None:
            stats["wer"] = wer

        weight = torch.tensor(batch_size, device=speech.device, dtype=torch.float)
        return total_loss, stats, weight

    # ------------------------------------------------------------------
    # CER/WER 計算
    # ------------------------------------------------------------------
    def _calc_cer_wer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.error_calculator is None:
            return None, None

        ys_hat = self.ctc.argmax(encoder_out).data  # (B, T)

        cer = None
        wer = None
        if self.report_cer:
            cer = self.error_calculator(ys_hat.cpu(), text.cpu(), is_ctc=True)
        if self.report_wer:
            _, wer = self.error_calculator(ys_hat.cpu(), text.cpu(), is_ctc=False)

        return cer, wer

    # ------------------------------------------------------------------
    # 推論（バッチ・早期確定付き）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference_tag_with_earliness(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sprt_upper: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """SPRT による早期確定タグ推論（バッチ処理・earliness 情報付き）。

        全音声を一括エンコードし、各 fire で LLR を累積する。
        いずれかのクラスの LLR が sprt_upper を超えた時点で確定する。
        全 fire でも超えなかった場合は最後の fire の argmax を返す。

        Args:
            speech       : (B, T) 音声波形
            speech_lengths: (B,)
            sprt_upper   : 停止境界（省略時は self.sprt_upper を使用）

        Returns:
            tag_pred      : (B,)      予測クラス（0=<no>, 1=<yes>, 2=<other>）
            tag_logits    : (B, C)    確定 fire での分類スコア
            cif_out       : (B, S, D) 各 fire のトークン表現
            decision_fire : (B,)      確定した fire インデックス（0 始まり）
            n_fires       : (B,)      各サンプルの総 fire 数
        """
        threshold = sprt_upper if sprt_upper is not None else self.sprt_upper

        encode_result = self.encode(speech, speech_lengths)
        encoder_out = encode_result[0]
        encoder_out_lens = encode_result[1]
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        B, T, D = encoder_out.shape
        alpha = self.alpha_predictor(encoder_out).squeeze(-1)
        pad_mask = (
            torch.arange(T, device=alpha.device).unsqueeze(0) < encoder_out_lens.unsqueeze(1)
        )
        alpha = alpha * pad_mask.float()

        cif_result = _cif_function_impl(
            inputs=encoder_out,
            alpha=alpha,
            target_lengths=None,
            beta=self.cif_threshold,
        )
        cif_out = cif_result["cif_out"][0]
        alpha_sum = cif_result["alpha_sum"][0]
        fire_counts = alpha_sum.round().long().clamp(min=1)

        device = cif_out.device
        tag_preds = []
        tag_logits_list = []
        decision_fires = []
        n_fires_list = []

        for b in range(B):
            n_fires = fire_counts[b].item()
            llr = torch.zeros(self.num_tag_classes, device=device)
            decision_fire = n_fires - 1
            decided = False

            for i in range(n_fires):
                logits_i = self.tag_classifier(
                    cif_out[b, i, :].unsqueeze(0)
                ).squeeze(0)                                  # (C,)
                probs_i = F.softmax(logits_i, dim=-1)         # (C,)
                llr = self._sprt_update(probs_i, llr)

                best_k = llr.argmax().item()
                if llr[best_k].item() >= threshold:
                    decision_fire = i
                    tag_preds.append(best_k)
                    tag_logits_list.append(logits_i)
                    decided = True
                    break

            if not decided:
                last_logits = self.tag_classifier(
                    cif_out[b, n_fires - 1, :].unsqueeze(0)
                ).squeeze(0)
                tag_preds.append(last_logits.argmax().item())
                tag_logits_list.append(last_logits)

            decision_fires.append(decision_fire)
            n_fires_list.append(n_fires)

        tag_pred = torch.tensor(tag_preds, device=device)
        tag_logits = torch.stack(tag_logits_list, dim=0)
        decision_fire_t = torch.tensor(decision_fires, device=device)
        n_fires_t = torch.tensor(n_fires_list, device=device)

        return tag_pred, tag_logits, cif_out, decision_fire_t, n_fires_t

    # ------------------------------------------------------------------
    # ストリーミング推論
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference_tag_streaming(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sim_chunk_feat_frames: int = 76,
        sprt_upper: Optional[float] = None,
    ) -> Tuple[int, int, int]:
        """特徴量レベルのストリーミング推論（SPRT による早期確定）。

        エンコーダを forward_infer でチャンク処理し（look_ahead=9 フレーム先のみ参照）、
        CIF を frame-by-frame で積算する。fire のたびに tag_classifier を実行し、
        LLR が sprt_upper を超えた時点でタグを確定して返す。

        Args:
            speech              : (1, T_samples) 音声波形。B=1 のみ対応。
            speech_lengths      : (1,)
            sim_chunk_feat_frames: エンコーダに一度に渡す特徴量フレーム数。
                                   CBT の hop_size × subsample ≈ 76 がデフォルト。
            sprt_upper          : 停止境界（省略時は self.sprt_upper を使用）

        Returns:
            tag_pred     : 予測クラス (0=<no>, 1=<yes>, 2=<other>)
            decision_fire: 確定した fire インデックス（0 始まり）
            n_fires      : 処理した総 fire 数
        """
        assert speech.shape[0] == 1, "streaming inference は B=1 のみ対応"
        threshold = sprt_upper if sprt_upper is not None else self.sprt_upper
        device = speech.device

        # ── 1. 特徴量抽出（全音声を一括処理） ──────────────────────────────
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        T_feat = feats.shape[1]
        enc_dim = self.encoder.output_size()

        # ── 2. ストリーミング状態の初期化 ───────────────────────────────────
        enc_states = None
        alpha_accum = 0.0
        cif_carry   = torch.zeros(enc_dim, device=device)

        # SPRT 状態
        llr           = torch.zeros(self.num_tag_classes, device=device)
        decided       = False
        decision_fire = 0
        fire_idx      = -1
        last_pred     = 0

        # ── 3. チャンクループ ────────────────────────────────────────────────
        pos = 0
        while pos < T_feat:
            end      = min(pos + sim_chunk_feat_frames, T_feat)
            is_final = (end >= T_feat)
            chunk    = feats[:, pos:end, :]
            chunk_lens = torch.tensor([chunk.shape[1]], device=device, dtype=torch.long)
            pos = end

            enc_out, _, enc_states = self.encoder(
                chunk, chunk_lens, enc_states,
                is_final=is_final, infer_mode=True,
            )

            if enc_out.shape[1] == 0:
                continue

            alpha = self.alpha_predictor(enc_out).squeeze(-1)  # (1, T_enc)

            # ── streaming CIF: frame-by-frame ────────────────────────────
            for t in range(enc_out.shape[1]):
                a_t = alpha[0, t].item()
                f_t = enc_out[0, t]

                while alpha_accum + a_t >= self.cif_threshold:
                    remainder   = self.cif_threshold - alpha_accum
                    fire_repr   = (cif_carry + remainder * f_t).unsqueeze(0)
                    a_t        -= remainder
                    alpha_accum = 0.0
                    cif_carry   = torch.zeros_like(f_t)
                    fire_idx   += 1

                    logits_i = self.tag_classifier(fire_repr).squeeze(0)  # (C,)
                    probs_i  = F.softmax(logits_i, dim=-1)
                    llr      = self._sprt_update(probs_i, llr)

                    best_k = llr.argmax().item()
                    last_pred = best_k

                    if llr[best_k].item() >= threshold:
                        decision_fire = fire_idx
                        decided = True
                        break

                if decided:
                    break

                alpha_accum += a_t
                cif_carry   += a_t * f_t

            if decided:
                break

        if not decided:
            decision_fire = max(fire_idx, 0)

        n_fires = fire_idx + 1
        return last_pred, decision_fire, n_fires

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        tag_label: Optional[torch.Tensor] = None,
        tag_label_lengths: Optional[torch.Tensor] = None,
        isdysfl: Optional[torch.Tensor] = None,
        isdysfl_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return super().collect_feats(
            speech=speech,
            speech_lengths=speech_lengths,
            text=text,
            text_lengths=text_lengths,
            isdysfl=isdysfl,
            isdysfl_lengths=isdysfl_lengths,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------
    def _resolve_tag_label(
        self,
        tag_label: Optional[torch.Tensor],
        text: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """タグクラスインデックス (B,) を返す。"""
        if tag_label is None:
            return None
        if tag_label.dim() == 2:
            return tag_label[:, 0].long()
        return tag_label.long()


# ------------------------------------------------------------------
# モジュール外ユーティリティ
# ------------------------------------------------------------------
def _get_last_token(cif_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """各バッチサンプルの最後の有効 fire 表現を返す。

    Args:
        cif_out : (B, S, D)
        lengths : (B,)  有効トークン数
    Returns:
        (B, D)
    """
    B, S, D = cif_out.shape
    idx = (lengths - 1).clamp(min=0, max=S - 1).long()
    idx = idx.view(B, 1, 1).expand(B, 1, D)
    return cif_out.gather(1, idx).squeeze(1)
