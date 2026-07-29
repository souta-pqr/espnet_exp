"""CIF + TEASER 早期確定モデル（Stage 2 学習 + 推論）。

使い方:
  Stage 1 : model: cif_tag (CIFTagTransformerModel) で学習済みチェックポイントを用意
  Stage 2 : model: cif_teaser (本クラス)、training_stage: 2 を指定して stop_head を学習
            init_param に Stage 1 チェックポイントを指定する

推論:
  inference_tag_teaser(speech, speech_lengths, stop_threshold, v)
  → tag_pred (B,), decision_fire (B,), n_fires (B,)
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.espnet_model_cif_tag import CIFTagTransformerModel

try:
    from torch_cif import cif_function as _cif_function_impl
    _CIF_AVAILABLE = True
except ImportError:
    _cif_function_impl = None
    _CIF_AVAILABLE = False


class CIFTeaserTransformerModel(CIFTagTransformerModel):
    """TEASER ベースの早期確定モデル。

    CIFTagTransformerModel（Stage 1）を継承し、stop_head を追加する。

    Stage 1: 親クラスの forward() をそのまま使う。
    Stage 2: training_stage=2 のとき forward() → forward_stage2() が呼ばれ、
             tag_classifier を凍結した上で stop_head の BCE 損失だけを返す。

    stop_head への入力特徴 φ_i:
        φ_i = [ソート済み確率(C), margin(1), entropy(1), 位置 i/(L-1)(1)] → dim = C+3
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
        # --- 親クラスパラメータ ---
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
        num_tag_classes: int = 3,
        cif_weight: float = 0.5,
        tag_weight: float = 1.0,
        cif_threshold: float = 1.0,
        sprt_upper: float = 2.0,
        # --- TEASER 固有パラメータ ---
        training_stage: int = 1,
        stop_head_hidden: int = 64,
        stop_threshold: float = 0.5,
        teaser_v: int = 2,
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
            num_tag_classes=num_tag_classes,
            cif_weight=cif_weight,
            tag_weight=tag_weight,
            cif_threshold=cif_threshold,
            sprt_upper=sprt_upper,
            **kwargs,
        )

        enc_dim = encoder.output_size()
        self.enc_dim = enc_dim

        # cif_out (D=256) を直接結合すると p_i 由来の5次元が埋もれるため、
        # cif_proj で16次元に削減してから結合する（cif_proj も Stage 2 で学習）。
        cif_proj_dim = 16
        self.cif_proj = nn.Sequential(
            nn.Linear(enc_dim, cif_proj_dim),
            nn.ReLU(),
        )

        # φ_i の次元: ソート済み確率(C) + margin(1) + entropy(1) + 位置(1) + cif_proj(16)
        #              = 3 + 1 + 1 + 1 + 16 = 22
        phi_dim = num_tag_classes + 3 + cif_proj_dim

        # master: stop_head（φ_i → stop_logit）
        self.stop_head = nn.Sequential(
            nn.Linear(phi_dim, stop_head_hidden),
            nn.ReLU(),
            nn.Linear(stop_head_hidden, 1),
            # Sigmoid は BCEWithLogitsLoss の内部で適用するため省略
        )

        self.training_stage = training_stage
        self.stop_threshold = stop_threshold
        self.teaser_v = teaser_v

        if training_stage == 2:
            self._freeze_except_stop_head()

        logging.info(
            f"CIFTeaserTransformerModel: training_stage={training_stage}, "
            f"stop_threshold={stop_threshold}, teaser_v={teaser_v}, "
            f"stop_head_hidden={stop_head_hidden}, phi_dim={phi_dim} "
            f"(C={num_tag_classes}+3+cif_proj={cif_proj_dim})"
        )

    # ------------------------------------------------------------------
    # freeze
    # ------------------------------------------------------------------
    def _freeze_except_stop_head(self):
        """stop_head と cif_proj 以外の全パラメータを凍結する。"""
        for name, param in self.named_parameters():
            if not name.startswith("stop_head.") and not name.startswith("cif_proj."):
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # forward（stage ルーティング）
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
        if self.training_stage == 2:
            return self.forward_stage2(
                speech, speech_lengths, text, text_lengths,
                tag_label, tag_label_lengths, **kwargs,
            )
        return super().forward(
            speech, speech_lengths, text, text_lengths,
            tag_label, tag_label_lengths, isdysfl, isdysfl_lengths,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Stage 2 forward
    # ------------------------------------------------------------------
    def forward_stage2(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        tag_label: Optional[torch.Tensor] = None,
        tag_label_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Stage 2: stop_head を BCE 損失で学習する。

        tag_classifier を含む全パラメータは凍結済みで、
        stop_head のみが更新される。
        """
        batch_size = speech.shape[0]

        resolved_tag_label = self._resolve_tag_label(tag_label, text)
        if resolved_tag_label is None:
            raise ValueError("Stage 2 forward には tag_label が必要です。")

        # ── encode + CIF（勾配不要）──────────────────────────────────────
        with torch.no_grad():
            encode_result = self.encode(speech, speech_lengths)
            encoder_out: torch.Tensor = encode_result[0]
            encoder_out_lens: torch.Tensor = encode_result[1]
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]

            alpha = self.alpha_predictor(encoder_out).squeeze(-1)   # (B, T)
            B, T = alpha.shape
            pad_mask = (
                torch.arange(T, device=alpha.device).unsqueeze(0)
                < encoder_out_lens.unsqueeze(1)
            )
            alpha = alpha * pad_mask.float()

            cif_result = _cif_function_impl(
                inputs=encoder_out,
                alpha=alpha,
                target_lengths=text_lengths.float(),
                beta=self.cif_threshold,
            )
            cif_out: torch.Tensor = cif_result["cif_out"][0]   # (B, T_cif, D)

            # tag_classifier（凍結済み）→ 確率
            tag_logits_all = self.tag_classifier(cif_out)           # (B, T_cif, C)
            tag_probs_all = F.softmax(tag_logits_all, dim=-1)       # (B, T_cif, C)

        B_out, T_cif, C = tag_probs_all.shape

        # 有効 fire マスク
        valid_fire = (
            torch.arange(T_cif, device=cif_out.device).unsqueeze(0)
            < text_lengths.unsqueeze(1)
        )   # (B, T_cif) bool

        # ── stop_target の生成 ───────────────────────────────────────────
        tag_preds_all = tag_logits_all.argmax(dim=-1)   # (B, T_cif)
        stop_target = self._make_stop_targets(
            tag_preds_all, resolved_tag_label, valid_fire, text_lengths, T_cif,
        )   # (B, T_cif) float {0, 1}

        # ── φ_i の構築 ───────────────────────────────────────────────────
        # ソート済み確率（降順）
        sorted_probs, _ = tag_probs_all.sort(dim=-1, descending=True)   # (B, T_cif, C)

        # margin = top1 - top2
        margin = (sorted_probs[:, :, 0] - sorted_probs[:, :, 1]).unsqueeze(-1)   # (B, T_cif, 1)

        # entropy = -Σ p log p
        entropy = -(tag_probs_all * (tag_probs_all + 1e-10).log()).sum(dim=-1).unsqueeze(-1)   # (B, T_cif, 1)

        # 位置 i / (L-1)
        pos_idx = (
            torch.arange(T_cif, device=cif_out.device)
            .float().unsqueeze(0).expand(B_out, T_cif)
        )   # (B, T_cif)
        denom = (text_lengths.float() - 1.0).unsqueeze(1).clamp(min=1.0)
        positions = (pos_idx / denom).unsqueeze(-1)   # (B, T_cif, 1)

        # cif_out を cif_proj で 256→16 次元に削減してから結合
        # → p_i 由来(5次元) と cif_proj(16次元) が均等に寄与
        cif_projected = self.cif_proj(cif_out)   # (B, T_cif, 16)

        phi = torch.cat(
            [sorted_probs, margin, entropy, positions, cif_projected],
            dim=-1,
        )   # (B, T_cif, C+3+16=22)

        # ── stop_head（ここだけ勾配あり）──────────────────────────────────
        stop_logits = self.stop_head(phi).squeeze(-1)   # (B, T_cif)

        # ── 重み付き BCE（0/1 不均衡を補正）─────────────────────────────
        valid_logits = stop_logits[valid_fire]   # (N_valid,)
        valid_targets = stop_target[valid_fire]  # (N_valid,)

        n_pos = valid_targets.sum().clamp(min=1.0)
        n_neg = (1.0 - valid_targets).sum().clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=10.0)

        loss_stop = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=pos_weight,
        )

        # モニタリング用精度
        with torch.no_grad():
            stop_pred = (torch.sigmoid(valid_logits) > 0.5).float()
            stop_acc = (stop_pred == valid_targets).float().mean().item()
            pos_ratio = valid_targets.mean().item()

        stats: Dict[str, torch.Tensor] = {
            "loss": loss_stop.detach(),
            "loss_stop": loss_stop.detach(),
            "stop_acc": stop_acc,
            "pos_ratio": pos_ratio,
        }
        weight = torch.tensor(batch_size, device=speech.device, dtype=torch.float)
        return loss_stop, stats, weight

    # ------------------------------------------------------------------
    # stop_target 生成（安定到達点ラベル）
    # ------------------------------------------------------------------
    @staticmethod
    def _make_stop_targets(
        tag_preds: torch.Tensor,      # (B, T_cif) argmax 予測
        tag_labels: torch.Tensor,     # (B,) 正解クラス
        valid_fire: torch.Tensor,     # (B, T_cif) bool
        text_lengths: torch.Tensor,   # (B,) 有効 fire 数
        T_cif: int,
    ) -> torch.Tensor:
        """安定到達点ラベルを生成する。

        τ* = min{ i : argmax(p_j) == y  for all j ∈ [i, L-1] }
        stop_target[i] = 1 if i >= τ*, else 0
        存在しない場合（全 fire 不正解）は最終 fire のみ 1。
        """
        # 各 fire で正解しているか（無効 fire は True にして suffix AND を壊さない）
        correct = (tag_preds == tag_labels.unsqueeze(1))   # (B, T_cif) bool
        correct_ext = correct.clone()
        correct_ext[~valid_fire] = True   # padding 位置は True で suffix AND に影響させない

        # suffix AND: flip → cumprod → flip
        suffix_and = (
            correct_ext.flip(dims=[1]).float().cumprod(dim=1).flip(dims=[1])
        )   # (B, T_cif) float {0, 1}

        # 有効 fire のみ
        stop_target = suffix_and * valid_fire.float()

        # 最終 fire は必ず 1（どこかで止まる必要がある）
        last_idx = (text_lengths - 1).clamp(min=0, max=T_cif - 1).long()
        stop_target.scatter_(1, last_idx.unsqueeze(1), 1.0)

        return stop_target

    # ------------------------------------------------------------------
    # バッチ推論（TEASER 停止ルール）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference_tag_teaser(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        stop_threshold: Optional[float] = None,
        v: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TEASER による早期確定タグ推論（バッチ処理）。

        停止条件: stop_prob > stop_threshold  かつ  同クラスが v 回連続

        Args:
            speech          : (B, T) 音声波形
            speech_lengths  : (B,)
            stop_threshold  : 省略時は self.stop_threshold を使用
            v               : 省略時は self.teaser_v を使用

        Returns:
            tag_pred      : (B,)  予測クラス（0=<no>, 1=<yes>, 2=<other>）
            decision_fire : (B,)  確定した fire インデックス（0 始まり）
            n_fires       : (B,)  各サンプルの総 fire 数
        """
        b_thresh = stop_threshold if stop_threshold is not None else self.stop_threshold
        min_v = v if v is not None else self.teaser_v

        encode_result = self.encode(speech, speech_lengths)
        encoder_out: torch.Tensor = encode_result[0]
        encoder_out_lens: torch.Tensor = encode_result[1]
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        B, T, D = encoder_out.shape
        alpha = self.alpha_predictor(encoder_out).squeeze(-1)
        pad_mask = (
            torch.arange(T, device=alpha.device).unsqueeze(0)
            < encoder_out_lens.unsqueeze(1)
        )
        alpha = alpha * pad_mask.float()

        cif_result = _cif_function_impl(
            inputs=encoder_out,
            alpha=alpha,
            target_lengths=None,
            beta=self.cif_threshold,
        )
        cif_out: torch.Tensor = cif_result["cif_out"][0]
        alpha_sum: torch.Tensor = cif_result["alpha_sum"][0]
        fire_counts = alpha_sum.round().long().clamp(min=1)

        device = cif_out.device
        tag_preds = []
        decision_fires = []
        n_fires_list = []

        for b in range(B):
            n_fires = fire_counts[b].item()
            L = n_fires

            stable_count = 0
            prev_class = -1
            decided = False

            for i in range(n_fires):
                cif_repr = cif_out[b, i]                        # (D,)
                logits_i = self.tag_classifier(
                    cif_repr.unsqueeze(0)
                ).squeeze(0)                                    # (C,)
                p_i = F.softmax(logits_i, dim=-1)              # (C,)
                pred_i = p_i.argmax().item()

                # φ_i を構築
                phi_i = self._build_phi(p_i, cif_repr, position=i / max(L - 1, 1))

                stop_logit = self.stop_head(phi_i.unsqueeze(0)).squeeze()
                stop_prob = torch.sigmoid(stop_logit).item()

                if stop_prob > b_thresh:
                    if pred_i == prev_class:
                        stable_count += 1
                        if stable_count >= min_v:
                            tag_preds.append(pred_i)
                            decision_fires.append(i)
                            decided = True
                            break
                    else:
                        prev_class = pred_i
                        stable_count = 1
                else:
                    # master が「まだ待て」→ 連続カウントをリセット
                    stable_count = 0

            if not decided:
                last_logits = self.tag_classifier(
                    cif_out[b, n_fires - 1].unsqueeze(0)
                ).squeeze(0)
                tag_preds.append(last_logits.argmax().item())
                decision_fires.append(n_fires - 1)

            n_fires_list.append(n_fires)

        tag_pred = torch.tensor(tag_preds, device=device)
        decision_fire_t = torch.tensor(decision_fires, device=device)
        n_fires_t = torch.tensor(n_fires_list, device=device)

        return tag_pred, decision_fire_t, n_fires_t

    # ------------------------------------------------------------------
    # φ 構築ユーティリティ
    # ------------------------------------------------------------------
    def _build_phi(
        self,
        p_i: torch.Tensor,
        cif_repr: torch.Tensor,
        position: float,
    ) -> torch.Tensor:
        """1 fire 分の特徴ベクトル φ を構築する。

        Args:
            p_i     : (C,) softmax 確率
            cif_repr: (D,) CIF の fire 表現（音響情報）
            position: i / (L-1)（発話内の相対位置）

        Returns:
            φ (C+3+16,=22): [ソート済み確率(C), margin(1), entropy(1), position(1), cif_proj(16)]
        """
        sorted_p, _ = p_i.sort(descending=True)
        margin = sorted_p[0] - sorted_p[1]
        entropy = -(p_i * (p_i + 1e-10).log()).sum()
        pos_t = torch.tensor([position], device=p_i.device, dtype=p_i.dtype)
        cif_projected = self.cif_proj(cif_repr.unsqueeze(0)).squeeze(0)   # (16,)
        return torch.cat([sorted_p, margin.unsqueeze(0), entropy.unsqueeze(0), pos_t, cif_projected])
