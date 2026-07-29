"""Multitask RNN-Transducer model implementation.

This module implements a multitask RNN-Transducer model that combines:
1. Automatic Speech Recognition (ASR)
2. Disfluency Detection (filled pauses, repetitions, etc.)

Architecture:
- Base: ESPnetASRModel
- Decoder: MultitaskTransducerDecoder with joint ASR+disfluency prediction
- Loss: Combined ASR transducer loss + disfluency classification loss
- Evaluation: CER/WER calculation with ErrorCalculatorTransducer

Usage:
- Configured via: model: multitask_rnnt in training config
- Paired with: decoder: multitask_transducer
- Supports: Joint training with weighted loss combination

Created: 2025-09 for CEJC corpus multitask experiments
Last Updated: 2025-09-30 (code cleanup and documentation)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.decoder.multitask_transducer import MultitaskTransducerDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer


class MultitaskRNNTModel(ESPnetASRModel):
    """Multitask RNN-Transducer Model with Disfluency Detection.
    
    This model extends the standard RNN-Transducer architecture with
    disfluency detection capabilities using a multitask learning approach.
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
        decoder: MultitaskTransducerDecoder,
        # CTC完全削除: ctcパラメータ削除
        joint_network: Optional[torch.nn.Module] = None,
        use_disfluency_detection: bool = True,
        disfluency_classes: int = 4,
        disfluency_weight: float = 1.0,
        report_cer: bool = True,
        report_wer: bool = True,
        extract_feats_in_collect_stats: bool = False,
        **kwargs,  # Accept additional arguments
    ):
        """Initialize MultitaskRNNTModel.
        
        Args:
            vocab_size: Vocabulary size.
            token_list: Token list.
            frontend: Frontend module.
            specaug: SpecAugment module.
            normalize: Normalization module.
            preencoder: Pre-encoder module.
            encoder: Encoder module.
            postencoder: Post-encoder module.
            decoder: Multitask transducer decoder.
            use_disfluency_detection: Whether to use disfluency detection.
            disfluency_classes: Number of disfluency classes.
            disfluency_weight: Weight for disfluency loss.
            report_cer: Whether to report CER.
            report_wer: Whether to report WER.
            extract_feats_in_collect_stats: Whether to extract features in collect_stats.
        """
        # Debug: Confirm MultitaskRNNTModel initialization
        print("=" * 50)
        print("MultitaskRNNTModel initialization successful!")
        print(f"CER reporting: {report_cer}, WER reporting: {report_wer}")
        print("=" * 50)
        
        # CTC完全削除: 型チェック回避のため最小限のダミーCTCを作成
        # 実際には使用しないが、親クラスの型チェックを通すために必要
        from espnet2.asr.ctc import CTC
        ctc = CTC(
            odim=vocab_size,
            encoder_output_size=encoder.output_size(),
            dropout_rate=0.0,
            ctc_type="builtin",
            reduce=True,
        )
        # CTCのパラメータを学習対象から除外
        for param in ctc.parameters():
            param.requires_grad = False
        
        # Remove unsupported kwargs
        filtered_kwargs = {}
        supported_kwargs = {
            'ignore_id', 'lsm_weight',
            'length_normalized_loss', 'sym_space', 'sym_blank',
            'transducer_multi_blank_durations', 'transducer_multi_blank_sigma',
            'sym_sos', 'sym_eos', 'lang_token_id'
        }
        # ctc_weight, interctc_weightは削除（使わない）
        for key, value in kwargs.items():
            if key in supported_kwargs:
                filtered_kwargs[key] = value
        
        # CTCを使わないようにweightを0に設定
        filtered_kwargs['ctc_weight'] = 0.0
        filtered_kwargs['interctc_weight'] = 0.0
        
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
            decoder=decoder,  # This will be the multitask decoder
            ctc=ctc,
            joint_network=joint_network,
            report_cer=report_cer,
            report_wer=report_wer,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            **filtered_kwargs
        )
        
        # Override with our multitask-specific configuration
        self.use_disfluency_detection = use_disfluency_detection
        self.disfluency_classes = disfluency_classes
        self.disfluency_weight = disfluency_weight
        
        # Expose joint_network for proper Transducer decoding
        # The decoder contains the joint network internally
        if hasattr(decoder, 'joint_network'):
            self.joint_network = decoder.joint_network
            # This ensures use_transducer_decoder = True in inference
            self.use_transducer_decoder = True
        
        # Add ErrorCalculatorTransducer for CER/WER calculation
        if report_cer or report_wer:
            self.error_calculator_trans = ErrorCalculatorTransducer(
                decoder,  # Pass the decoder directly 
                decoder.joint,  # Pass the joint wrapper (beam search compatible)
                token_list,
                filtered_kwargs.get('sym_space', '<space>'),
                filtered_kwargs.get('sym_blank', '<blank>'),
                report_cer=report_cer,
                report_wer=report_wer,
            )
        else:
            self.error_calculator_trans = None
        
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
        
        # Extract features
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        
        # Encode
        encoder_out, encoder_out_lens = self.encode(feats, feats_lengths)
        
        # Decoder forward pass
        # CTC不要！Transducerのjoint logitsからアライメントを取得
        
        # SAFETY: Replace padding (-1) with fluent (0) in disfluency labels
        if isdysfl is not None:
            isdysfl = torch.where(isdysfl == -1, torch.zeros_like(isdysfl), isdysfl)
            # Also clamp to valid range just in case
            isdysfl = torch.clamp(isdysfl, 0, self.disfluency_classes - 1)
        
        # Pass current_epoch for scheduled sampling (extract from kwargs if available)
        # current_epoch = kwargs.get('current_epoch', 0)
        
        loss, loss_dict = self.decoder(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            labels=text,
            label_lens=text_lengths,
            disfluency_labels=isdysfl,
            # current_epoch=current_epoch,
        )
        
        # Calculate CER/WER during validation/evaluation only (following ESPnet2 standard)
        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            try:
                # Use the ASR targets for CER/WER calculation
                cer_transducer, wer_transducer = self.error_calculator_trans(
                    encoder_out, text
                )
                print(f"[VALIDATION] CER: {cer_transducer:.4f}, WER: {wer_transducer:.4f}")
            except Exception as e:
                print(f"[ERROR] CER calculation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Create stats dictionary
        stats = {
            "loss": loss.detach(),
            "loss_asr": loss_dict.get("loss_asr", loss).detach(),
            "cer_transducer": cer_transducer,
            "wer_transducer": wer_transducer,
        }
        
        if self.use_disfluency_detection and "disfluency_loss" in loss_dict:
            stats["loss_disfluency"] = loss_dict["disfluency_loss"].detach()
        
        # Weight is batch size for averaging
        weight = torch.tensor(batch_size, device=speech.device, dtype=torch.float)
        
        return loss, stats, weight

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

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode speech features.
        
        Args:
            speech: Raw speech signal (B, T) or preprocessed features (B, T, D).
            speech_lengths: Speech lengths (B,).
            
        Returns:
            Tuple of (encoder_out, encoder_out_lens).
        """
        # Check if input is raw speech (2D) or features (3D)
        if speech.dim() == 2:
            # Raw speech input - extract features first
            speech, speech_lengths = self._extract_feats(speech, speech_lengths)
        elif speech.dim() == 3:
            # Already processed features
            pass
        else:
            raise ValueError(f"Unsupported speech tensor dimension: {speech.dim()}")
            
        # Pre-encoder
        if self.preencoder is not None:
            speech, speech_lengths = self.preencoder(speech, speech_lengths)
        
        # Encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        
        # Post-encoder
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )
            
        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from raw speech.
        
        Args:
            speech: Raw speech signal (B, T).
            speech_lengths: Speech lengths (B,).
            
        Returns:
            Tuple of (feats, feats_lengths).
        """
        feats, feats_lengths = speech, speech_lengths
        
        # Frontend
        if self.frontend is not None:
            feats, feats_lengths = self.frontend(feats, feats_lengths)
            
        # SpecAugment
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)
            
        # Normalization
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)
            
        return feats, feats_lengths
    
    def predict_disfluency(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        token_ids: List[int],
    ) -> List[int]:
        """Predict disfluency labels for decoded tokens with token-dependency mechanism.
        
        Token-Dependency Mechanism:
            p(di|X, y≤i) = softmax(W[E(yi); si] + b)
            - E(yi): embedding of the current predicted token
            - si: joint network hidden state (encoder + decoder)
            - Naturally implemented by joint_network.forward()
        
        Args:
            encoder_out: Encoder output (1, T, D)
            encoder_out_lens: Encoder output lengths (1,)
            token_ids: Decoded token IDs (without blank)
        
        Returns:
            disfluency_predictions: List of disfluency class IDs 
                                   (0=fluent, 1=filler, 2=reparandum, 3=interjection)
        """
        if not self.use_disfluency_detection:
            return None
        
        # Get decoder and joint network
        decoder_module = self.decoder
        joint_network = decoder_module.joint_network
        
        # Initialize decoder state
        dec_state = decoder_module.init_state(1)
        
        disfluency_preds = []
        
        # For each decoded token, get disfluency prediction
        yseq = [decoder_module.blank_id] + token_ids
        
        for i, token_id in enumerate(token_ids):
            # Prepare label (previous token for decoder input)
            label = torch.full(
                (1, 1), yseq[i], dtype=torch.long, device=encoder_out.device
            )
            
            # Decoder forward
            dec_emb = decoder_module.embed(label)
            dec_out, dec_state = decoder_module.rnn_forward(dec_emb, dec_state)
            
            # Get disfluency prediction from joint network
            # Align with encoder timesteps - use proportional mapping
            T = encoder_out.shape[1]
            t = min(int(i / len(token_ids) * T), T - 1)
            
            enc_t = encoder_out[:, t:t+1, :]  # (1, 1, D_enc)
            dec_t = dec_out  # (1, 1, D_dec)
            
            # Prepare predicted token for token-dependency mechanism
            # current_token is token_ids[i] (the token we just predicted)
            current_token = torch.tensor([[token_id]], dtype=torch.long, device=encoder_out.device)  # (1, 1)
            
            # Call joint network with token-dependency mechanism
            # predicted_tokens shape: (1, 1, 1) for single step inference
            outputs = joint_network.forward(
                enc_t.squeeze(1),  # (1, D_enc)
                dec_t.squeeze(1),  # (1, D_dec)
                return_disfluency=True,
                predicted_tokens=current_token.unsqueeze(1)  # (1, 1, 1)
            )
            
            if "disfluency_scores" in outputs:
                dysfl_scores = outputs["disfluency_scores"]  # (1, 1, 1, num_classes)
                dysfl_pred = torch.argmax(dysfl_scores, dim=-1).squeeze().item()
                disfluency_preds.append(dysfl_pred)
            else:
                disfluency_preds.append(0)  # default: fluent
        
        return disfluency_preds
