"""Multitask Transformer model implementation with Disfluency Detection.

This module implements a multitask Transformer model that combines:
1. Automatic Speech Recognition (ASR)
2. Disfluency Detection (filled pauses, repetitions, etc.)

Architecture:
- Encoder: Contextual Block Transformer (12 layers)
- Decoder: Disfluency Transformer Decoder (6 layers)
- CTC: Joint training with CTC (weight 0.3)
- Multitask: ASR + Disfluency detection

Created: 2026-01-15 for CEJC corpus multitask experiments
"""

import torch
from typing import Dict, List, Optional, Tuple, Union

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.decoder.transformer_decoder import DisfluencyTransformerDecoder
from espnet2.asr.ctc import CTC
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)


class MultitaskTransformerModel(ESPnetASRModel):
    """Multitask Transformer Model with Disfluency Detection.
    
    This model extends the standard Transformer architecture with
    disfluency detection capabilities using a multitask learning approach.
    
    Key features:
    - Contextual Block Transformer encoder
    - Disfluency-aware Transformer decoder
    - Joint training with CTC
    - Multitask loss: ASR + Disfluency detection
    """

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
        decoder: DisfluencyTransformerDecoder,
        ctc: CTC,
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
        **kwargs,
    ):
        """Initialize MultitaskTransformerModel.
        
        Args:
            vocab_size: Vocabulary size.
            token_list: Token list.
            frontend: Frontend module.
            specaug: SpecAugment module.
            normalize: Normalization module.
            preencoder: Pre-encoder module.
            encoder: Encoder module (Contextual Block Transformer).
            postencoder: Post-encoder module.
            decoder: Disfluency Transformer decoder.
            ctc: CTC module.
            use_disfluency_detection: Whether to use disfluency detection.
            disfluency_classes: Number of disfluency classes.
            disfluency_weight: Weight for disfluency loss.
            ctc_weight: Weight for CTC loss.
            ignore_id: Padding token ID.
            lsm_weight: Label smoothing weight.
            length_normalized_loss: Whether to normalize loss by length.
            report_cer: Whether to report CER.
            report_wer: Whether to report WER.
            sym_space: Space symbol.
            sym_blank: Blank symbol.
            extract_feats_in_collect_stats: Whether to extract features in collect_stats.
        """
        # Filter kwargs to avoid passing unexpected arguments
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['sym_sos', 'sym_eos', 'lang_token_id']
        }
        
        # Call parent constructor
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
            joint_network=None,  # Not used in attention-based model
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            **filtered_kwargs
        )
        
        # Multitask-specific configuration
        self.use_disfluency_detection = use_disfluency_detection
        self.disfluency_classes = disfluency_classes
        self.disfluency_weight = disfluency_weight
        self.ctc_weight = ctc_weight
        
        # Criterion for decoder (attention loss)
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        
        # Criterion for disfluency detection
        if self.use_disfluency_detection:
            self.criterion_disfluency = torch.nn.CrossEntropyLoss(
                ignore_index=ignore_id,
                reduction='mean',
            )
        
        print("=" * 60)
        print("MultitaskTransformerModel initialized successfully!")
        print(f"  - Encoder: {encoder.__class__.__name__}")
        print(f"  - Decoder: {decoder.__class__.__name__}")
        print(f"  - CTC weight: {ctc_weight}")
        print(f"  - Disfluency detection: {use_disfluency_detection}")
        print(f"  - Disfluency weight: {disfluency_weight}")
        print(f"  - Vocab size: {vocab_size}")
        print("=" * 60)
        
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        isdysfl: Optional[torch.Tensor] = None,
        isdysfl_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward computation for training.
        
        Args:
            speech: Speech signal (B, T_speech).
            speech_lengths: Speech lengths (B,).
            text: Text labels (B, T_text).
            text_lengths: Text lengths (B,).
            isdysfl: Disfluency labels (B, T_disfluency) or None.
            isdysfl_lengths: Disfluency lengths (B,) or None.
            
        Returns:
            Tuple of (loss, stats, weight).
        """
        batch_size = speech.shape[0]
        
        # 1. Extract features
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        
        # 2. Encode
        encoder_out, encoder_out_lens, _ = self.encode(feats, feats_lengths)
        
        # 3. CTC forward
        loss_ctc = None
        if self.ctc_weight > 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        
        # 4. Prepare decoder inputs (shift right)
        ys_in_pad, ys_out_pad = self._target_shift(text, self.ignore_id)
        ys_in_lens = text_lengths
        
        # 5. Decoder forward with disfluency detection
        if self.use_disfluency_detection and isdysfl is not None:
            # Build decoder input: prepend class-0 (fluent) and shift right
            # -1 padding is replaced with 0 only for the INPUT (embedding lookup)
            isdysfl_clean = isdysfl.clone()
            isdysfl_clean[isdysfl_clean == self.ignore_id] = 0
            sos_d = isdysfl_clean.new_zeros((isdysfl_clean.size(0), 1))
            disfluency_in_pad = torch.cat([sos_d, isdysfl_clean[:, :-1]], dim=1)
            # Keep original -1 padding in output target so criterion ignores pads
            disfluency_out_pad = isdysfl

            # Forward with disfluency
            (decoder_out, disfluency_logits), _ = self.decoder(
                hs_pad=encoder_out,
                hlens=encoder_out_lens,
                ys_in_pad=ys_in_pad,
                ys_in_lens=ys_in_lens,
                disfluency_in_pad=disfluency_in_pad,
                return_disfluency=True,
            )
        else:
            # Forward without disfluency
            decoder_out, _ = self.decoder(
                hs_pad=encoder_out,
                hlens=encoder_out_lens,
                ys_in_pad=ys_in_pad,
                ys_in_lens=ys_in_lens,
                return_disfluency=False,
            )
            disfluency_logits = None
        
        # 6. Calculate ASR loss (attention)
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        
        # 7. Calculate disfluency loss
        loss_disfluency = None
        if self.use_disfluency_detection and disfluency_logits is not None:
            # Flatten for cross-entropy
            disfluency_logits_flat = disfluency_logits.view(-1, self.disfluency_classes)
            disfluency_out_pad_flat = disfluency_out_pad.view(-1)
            loss_disfluency = self.criterion_disfluency(
                disfluency_logits_flat, disfluency_out_pad_flat
            )
        
        # 8. Combine losses
        loss = 0.0
        if self.ctc_weight > 0.0 and loss_ctc is not None:
            loss += self.ctc_weight * loss_ctc
        
        loss += (1 - self.ctc_weight) * loss_att
        
        if self.use_disfluency_detection and loss_disfluency is not None:
            loss += self.disfluency_weight * loss_disfluency
        
        # 9. Calculate CER/WER (during validation only)
        cer, wer = None, None
        if not self.training and (self.report_cer or self.report_wer):
            cer, wer = self._calc_cer_wer(
                encoder_out, encoder_out_lens, text, text_lengths
            )
        
        # 10. Create stats dictionary
        stats = {
            "loss": loss.detach() if isinstance(loss, torch.Tensor) else loss,
            "loss_att": loss_att.detach(),
        }
        
        if loss_ctc is not None:
            stats["loss_ctc"] = loss_ctc.detach()
        
        if loss_disfluency is not None:
            stats["loss_disfluency"] = loss_disfluency.detach()
        
        if cer is not None:
            stats["cer"] = cer
        if wer is not None:
            stats["wer"] = wer
        
        # 11. Weight is batch size for averaging
        weight = torch.tensor(batch_size, device=speech.device, dtype=torch.float)
        
        # Force gather stats for distributed training
        stats = {k: v for k, v in stats.items()}
        
        return loss, stats, weight

    def _target_shift(
        self, ys_pad: torch.Tensor, ignore_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift target sequence for teacher forcing.
        
        Args:
            ys_pad: Target sequence (B, L).
            ignore_id: Padding ID.
            
        Returns:
            Tuple of (ys_in_pad, ys_out_pad).
        """
        sos = ys_pad.new_full((ys_pad.size(0), 1), self.sos)
        ys_in_pad = torch.cat([sos, ys_pad[:, :-1]], dim=1)
        ys_out_pad = ys_pad
        return ys_in_pad, ys_out_pad

    def _calc_cer_wer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate CER and WER using beam search.
        
        Args:
            encoder_out: Encoder output (B, T, D).
            encoder_out_lens: Encoder output lengths (B,).
            text: Reference text (B, L).
            text_lengths: Reference text lengths (B,).
            
        Returns:
            Tuple of (cer, wer).
        """
        # Use error calculator if available
        if hasattr(self, 'error_calculator') and self.error_calculator is not None:
            cer, wer = self.error_calculator(encoder_out, text)
            return cer, wer
        
        return None, None

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        isdysfl: Optional[torch.Tensor] = None,
        isdysfl_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features for statistics calculation.
        
        Args:
            speech: Speech signal (B, T_speech).
            speech_lengths: Speech lengths (B,).
            text: Text labels (B, T_text).
            text_lengths: Text lengths (B,).
            isdysfl: Disfluency labels (B, T_disfluency) or None.
            isdysfl_lengths: Disfluency lengths (B,) or None.
            
        Returns:
            Dictionary of features.
        """
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Use raw speech
            feats, feats_lengths = speech, speech_lengths

        return {"feats": feats, "feats_lengths": feats_lengths}
