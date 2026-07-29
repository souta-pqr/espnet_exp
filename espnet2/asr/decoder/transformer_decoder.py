# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
import logging
from typing import Any, List, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import (
    BatchScorerInterface,
    MaskParallelScorerInterface,
)


class BaseTransformerDecoder(
    AbsDecoder, BatchScorerInterface, MaskParallelScorerInterface
):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        gradient_checkpoint_layers: List[int] = [],
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self._output_size_bf_softmax = attention_dim
        # Must set by the inheritance
        self.decoders = None
        self.batch_ids = None

        # For gradient checkpointing, start from 1 (not 0)
        self.gradient_checkpoint_layers = gradient_checkpoint_layers
        logging.info(f"Gradient checkpoint layers: {self.gradient_checkpoint_layers}")

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        return_hs: bool = False,
        return_all_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            return_hs: (bool) whether to return the last hidden output
                                  before output layer
            return_all_hs: (bool) whether to return all the hidden intermediates
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        x = self.embed(tgt)
        intermediate_outs = []
        for layer_idx, decoder_layer in enumerate(self.decoders):
            if layer_idx + 1 in self.gradient_checkpoint_layers:
                x, tgt_mask, memory, memory_mask = torch.utils.checkpoint.checkpoint(
                    decoder_layer, x, tgt_mask, memory, memory_mask, use_reentrant=False
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder_layer(
                    x, tgt_mask, memory, memory_mask
                )
            if return_all_hs:
                intermediate_outs.append(x)
        if self.normalize_before:
            x = self.after_norm(x)
        if return_hs:
            hidden = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        if return_hs:
            return (x, hidden), olens
        elif return_all_hs:
            return (x, intermediate_outs), olens
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if return_hs:
            hidden = y
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return (y, hidden), new_cache
        return y, new_cache

    def score(self, ys, state, x, return_hs=False):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if return_hs:
            (logp, hs), state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), hs, state
        else:
            logp, state = self.forward_one_step(
                ys.unsqueeze(0),
                ys_mask,
                x.unsqueeze(0),
                cache=state,
                return_hs=return_hs,
            )
            return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).


        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        if return_hs:
            (logp, hs), states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=return_hs
            )
        else:
            logp, states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=return_hs
            )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        if return_hs:
            return (logp, hs), state_list
        return logp, state_list

    def forward_partially_AR(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_lengths: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (n_mask * n_beam, maxlen_out)
            tgt_mask: input token mask,  (n_mask * n_beam, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            tgt_lengths: (n_mask * n_beam, )
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)  # (n_mask * n_beam, maxlen_out, D)
        new_cache = []
        if cache is None:
            cache = [None] * len(self.decoders)

        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, tgt_lengths, memory, memory_mask = (
                decoder.forward_partially_AR(
                    x, tgt_mask, tgt_lengths, memory, None, cache=c
                )
            )
            new_cache.append(x)

        if self.batch_ids is None or len(self.batch_ids) < x.size(0):
            self.batch_ids = torch.arange(x.size(0), device=x.device)

        if self.normalize_before:
            y = self.after_norm(
                x[self.batch_ids[: x.size(0)], tgt_lengths.unsqueeze(0) - 1].squeeze(0)
            )
        else:
            y = x[self.batch_ids, tgt_lengths.unsqueeze(0) - 1].squeeze(0)

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, torch.stack(new_cache)

    def batch_score_partially_AR(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        yseq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Any]]:
        # merge states
        if states[0] is None:
            batch_state = None
        else:
            # reshape state of [mask * batch, layer, 1, D]
            # into [layer, mask * batch, 1, D]
            batch_state = states.transpose(0, 1)

        # batch decoding
        tgt_mask = (~make_pad_mask(yseq_lengths)[:, None, :]).to(xs.device)
        m = subsequent_mask(tgt_mask.size(-1), device=xs.device).unsqueeze(0)
        tgt_mask = tgt_mask & m

        logp, states = self.forward_partially_AR(
            ys, tgt_mask, yseq_lengths, xs, cache=batch_state
        )

        # states is torch.Tensor, where shape is (layer, n_mask * n_beam, yseq_len, D)
        # reshape state to [n_mask * n_beam, layer, yseq_len, D]
        state_list = states.transpose(0, 1)
        return logp, state_list


class TransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
        gradient_checkpoint_layers: List[int] = [],
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            gradient_checkpoint_layers=gradient_checkpoint_layers,
        )

        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn  # noqa
            except Exception:
                use_flash_attn = False

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    True,
                    False,
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    True,
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )


class LightweightConvolutionTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class LightweightConvolution2DTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                LightweightConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolutionTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class DynamicConvolution2DTransformerDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        conv_wshare: int = 4,
        conv_kernel_length: Sequence[int] = (11, 11, 11, 11, 11, 11),
        conv_usebias: int = False,
    ):
        if len(conv_kernel_length) != num_blocks:
            raise ValueError(
                "conv_kernel_length must have equal number of values to num_blocks: "
                f"{len(conv_kernel_length)} != {num_blocks}"
            )
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                DynamicConvolution2D(
                    wshare=conv_wshare,
                    n_feat=attention_dim,
                    dropout_rate=self_attention_dropout_rate,
                    kernel_size=conv_kernel_length[lnum],
                    use_kernel_mask=True,
                    use_bias=conv_usebias,
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )


class TransformerMDDecoder(BaseTransformerDecoder):
    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        use_speech_attn: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )

        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
                (
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    )
                    if use_speech_attn
                    else None
                ),
            ),
        )

        self.use_speech_attn = use_speech_attn

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        speech: torch.Tensor = None,
        speech_lens: torch.Tensor = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )

        if speech is not None:
            speech_mask = (~make_pad_mask(speech_lens, maxlen=speech.size(1)))[
                :, None, :
            ].to(speech.device)
        else:
            speech_mask = None

        x = self.embed(tgt)
        if self.use_speech_attn:
            x, tgt_mask, memory, memory_mask, _, speech, speech_mask = self.decoders(
                x, tgt_mask, memory, memory_mask, None, speech, speech_mask
            )
        else:
            x, tgt_mask, memory, memory_mask = self.decoders(
                x, tgt_mask, memory, memory_mask
            )
        if self.normalize_before:
            x = self.after_norm(x)
            if return_hs:
                hs_asr = x
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)

        if return_hs:
            return x, olens, hs_asr

        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        speech: torch.Tensor = None,
        speech_mask: torch.Tensor = None,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            speech: encoded speech, float32  (batch, maxlen_in, feat)
            speech_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list of (batch, max_time_out-1, size)
            return_hs: dec hidden state corresponding to ys,
                used for searchable hidden ints
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            if self.use_speech_attn:
                x, tgt_mask, memory, memory_mask, _, speech, speech_mask = decoder(
                    x,
                    tgt_mask,
                    memory,
                    memory_mask,
                    cache=c,
                    pre_memory=speech,
                    pre_memory_mask=speech_mask,
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder(
                    x, tgt_mask, memory, memory_mask, cache=c
                )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if return_hs:
            h_asr = y

        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        if return_hs:
            return y, h_asr, new_cache
        return y, new_cache

    def score(self, ys, state, x, speech=None):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0),
            ys_mask,
            x.unsqueeze(0),
            speech=speech.unsqueeze(0) if speech is not None else None,
            cache=state,
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        speech: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(
            ys, ys_mask, xs, speech=speech, cache=batch_state
        )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


class DisfluencyTransformerDecoder(BaseTransformerDecoder):
    """Transformer decoder with disfluency detection capability.
    
    This decoder jointly performs ASR token prediction and disfluency detection
    with a token-dependency mechanism as described in:
    "Streaming Joint Speech Recognition and Disfluency Detection"
    
    Args:
        vocab_size: output dim for ASR tokens
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        positional_dropout_rate: positional encoding dropout rate
        self_attention_dropout_rate: dropout rate for self attention
        src_attention_dropout_rate: dropout rate for src attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
        layer_drop_rate: layer drop rate
        qk_norm: whether to use qk normalization
        use_flash_attn: whether to use flash attention
        gradient_checkpoint_layers: layers to use gradient checkpointing
        disfluency_classes: number of disfluency classes (default: 4)
            0: fluent, 1: filler, 2: repetition, 3: interjection
        disfluency_weight: weight for disfluency loss (default: 1.0)
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
        gradient_checkpoint_layers: List[int] = [],
        disfluency_classes: int = 4,
        disfluency_weight: float = 1.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            gradient_checkpoint_layers=gradient_checkpoint_layers,
        )

        if use_flash_attn:
            try:
                from espnet2.torch_utils.get_flash_attn_compatability import (
                    is_flash_attn_supported,
                )

                use_flash_attn = is_flash_attn_supported()
                import flash_attn  # noqa
            except Exception:
                use_flash_attn = False

        attention_dim = encoder_output_size
        self.disfluency_classes = disfluency_classes
        self.disfluency_weight = disfluency_weight
        
        # Standard transformer decoder layers
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    True,
                    False,
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    qk_norm,
                    use_flash_attn,
                    False,
                    True,
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
            layer_drop_rate,
        )
        
        # =========================================================================
        # Paper Implementation: Complete Multi-task Model with Token-Dependency
        # =========================================================================
        
        # 1. Disfluency Embedding (NEW - from paper Section 2.2)
        #    "we input d<i along with y<i. To feed these input representations
        #     to the network, token and disfluency input embeddings are summed,
        #     similar to token and segment embeddings in BERT."
        self.disfluency_embedding = torch.nn.Embedding(disfluency_classes, attention_dim)
        
        # 2. Token embedding layer for token-dependency mechanism
        #    Used in: p(di|X, y≤i, d<i) = softmax(W[E(yi); si] + b)
        self.token_embedding = torch.nn.Embedding(vocab_size, attention_dim)
        
        # 3. Disfluency detection output layer with token-dependency mechanism
        #    Concatenate token embedding E(yi) and hidden state si
        self.disfluency_output_layer = torch.nn.Linear(
            attention_dim + attention_dim,  # [E(yi); si]
            disfluency_classes
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        disfluency_in_pad: torch.Tensor = None,
        return_hs: bool = False,
        return_all_hs: bool = False,
        return_disfluency: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder with disfluency detection.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input token ids, int64 (batch, maxlen_out)
            ys_in_lens: (batch)
            disfluency_in_pad: input disfluency labels, int64 (batch, maxlen_out)
                              For training: ground-truth d<i
                              For inference: predicted d<i
                              If None, use zeros (all fluent)
            return_hs: whether to return the last hidden output before output layer
            return_all_hs: whether to return all hidden intermediates
            return_disfluency: whether to return disfluency predictions

        Returns:
            (tuple): tuple containing:
            asr_out: decoded token score before softmax (batch, maxlen_out, vocab_size)
            disfluency_out: disfluency scores (batch, maxlen_out, disfluency_classes)
            olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens, maxlen=memory.size(1)))[:, None, :].to(
            memory.device
        )
        # Padding for Longformer
        if memory_mask.shape[-1] != memory.shape[1]:
            padlen = memory.shape[1] - memory_mask.shape[-1]
            memory_mask = torch.nn.functional.pad(
                memory_mask, (0, padlen), "constant", False
            )

        # =========================================================================
        # Paper Implementation: Input d<i along with y<i (Section 2.2)
        # "To model the dependency between di and previous disfluency predictions
        #  d<i, we input d<i along with y<i. To feed these input representations
        #  to the network, token and disfluency input embeddings are summed,
        #  similar to token and segment embeddings in BERT."
        # =========================================================================
        x = self.embed(tgt)  # Token embedding
        
        # Add disfluency embedding (BERT-style: token + segment embeddings)
        if disfluency_in_pad is not None:
            # Use provided disfluency labels (training or autoregressive inference)
            dysfl_emb = self.disfluency_embedding(disfluency_in_pad)
        else:
            # Default: all fluent (class 0)
            dysfl_pad = torch.zeros_like(tgt, dtype=torch.long, device=tgt.device)
            dysfl_emb = self.disfluency_embedding(dysfl_pad)
        
        # Sum token and disfluency embeddings (like BERT)
        x = x + dysfl_emb
        intermediate_outs = []
        for layer_idx, decoder_layer in enumerate(self.decoders):
            if layer_idx + 1 in self.gradient_checkpoint_layers:
                x, tgt_mask, memory, memory_mask = torch.utils.checkpoint.checkpoint(
                    decoder_layer, x, tgt_mask, memory, memory_mask, use_reentrant=False
                )
            else:
                x, tgt_mask, memory, memory_mask = decoder_layer(
                    x, tgt_mask, memory, memory_mask
                )
            if return_all_hs:
                intermediate_outs.append(x)
                
        if self.normalize_before:
            x = self.after_norm(x)
            
        hidden_states = x  # Store hidden states for disfluency prediction
        
        if return_hs:
            hidden = x
            
        # ASR token prediction
        asr_out = None
        if self.output_layer is not None:
            asr_out = self.output_layer(x)
        
        # Disfluency prediction with token-dependency mechanism
        disfluency_out = None
        if return_disfluency and asr_out is not None:
            # Get predicted tokens (for teacher forcing during training)
            # During training, use target tokens; during inference, use predicted tokens
            if self.training:
                # Use target tokens (teacher forcing)
                target_tokens = ys_in_pad
            else:
                # Use predicted tokens
                target_tokens = asr_out.argmax(dim=-1)
            
            # Token-dependency mechanism: p(di|X, y≤i, d<i) = softmax(W[E(yi); si] + b)
            token_embeddings = self.token_embedding(target_tokens)  # (B, L, D)
            
            # Concatenate token embeddings and hidden states
            disfluency_input = torch.cat([token_embeddings, hidden_states], dim=-1)  # (B, L, 2*D)
            
            # Predict disfluency scores
            disfluency_out = self.disfluency_output_layer(disfluency_input)  # (B, L, disfluency_classes)

        # tgt_mask is (B, L, L); take the last row of each sample to get (B,)
        olens = tgt_mask[:, -1, :].sum(-1)

        if return_disfluency and disfluency_out is not None:
            if return_hs:
                return (asr_out, disfluency_out, hidden), olens
            elif return_all_hs:
                return (asr_out, disfluency_out, intermediate_outs), olens
            else:
                return (asr_out, disfluency_out), olens
        else:
            if return_hs:
                return (asr_out, hidden), olens
            elif return_all_hs:
                return (asr_out, intermediate_outs), olens
            return asr_out, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor = None,
        *,
        cache: List[torch.Tensor] = None,
        return_hs: bool = False,
        return_disfluency: bool = True,
        disfluency_in_pad: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step with disfluency detection.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask (batch, 1, maxlen_in)
            cache: cached output list
            return_hs: whether to return hidden states
            return_disfluency: whether to return disfluency predictions
            disfluency_in_pad: previous disfluency labels d<i, int64 (batch, maxlen_out)
                              Used as input following paper Section 2.2

        Returns:
            y: NN output value
            disfluency_y: disfluency predictions (if return_disfluency=True)
            cache: cache per self.decoders
        """
        # Token embedding
        x = self.embed(tgt)
        
        # NEW: Add disfluency embedding (d<i input from paper Section 2.2)
        if disfluency_in_pad is not None:
            dysfl_emb = self.disfluency_embedding(disfluency_in_pad)
        else:
            # Default: all fluent (label=0) if not provided
            dysfl_pad = torch.zeros_like(tgt, dtype=torch.long, device=tgt.device)
            dysfl_emb = self.disfluency_embedding(dysfl_pad)
        
        # Sum token and disfluency embeddings (BERT-style: token + segment)
        x = x + dysfl_emb
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
            
        hidden_state = y  # Store for disfluency prediction
        
        if return_hs:
            hidden = y
            
        # ASR token prediction
        asr_y = None
        if self.output_layer is not None:
            asr_y = torch.log_softmax(self.output_layer(y), dim=-1)
        
        # Disfluency prediction
        disfluency_y = None
        if return_disfluency and asr_y is not None:
            # Get the predicted token for the current step
            predicted_token = asr_y.argmax(dim=-1)  # (batch,)
            
            # Token embedding for current predicted token
            token_emb = self.token_embedding(predicted_token)  # (batch, D)
            
            # Token-dependency mechanism
            disfluency_input = torch.cat([token_emb, hidden_state], dim=-1)  # (batch, 2*D)
            disfluency_y = torch.log_softmax(self.disfluency_output_layer(disfluency_input), dim=-1)

        if return_disfluency and disfluency_y is not None:
            if return_hs:
                return (asr_y, disfluency_y, hidden), new_cache
            else:
                return (asr_y, disfluency_y), new_cache
        else:
            if return_hs:
                return (asr_y, hidden), new_cache
            return asr_y, new_cache

    def score(self, ys, state, x, return_hs=False, return_disfluency=True, disfluency_in_pad=None):
        """Score with disfluency detection.
        
        Args:
            ys: token sequence
            state: decoder state cache
            x: encoder output
            return_hs: whether to return hidden states
            return_disfluency: whether to return disfluency predictions
            disfluency_in_pad: previous disfluency labels d<i (optional)
        """
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        
        if return_disfluency:
            if return_hs:
                (logp, disfluency_logp, hs), state = self.forward_one_step(
                    ys.unsqueeze(0),
                    ys_mask,
                    x.unsqueeze(0),
                    cache=state,
                    return_hs=return_hs,
                    return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad.unsqueeze(0) if disfluency_in_pad is not None else None,
                )
                return (logp.squeeze(0), disfluency_logp.squeeze(0)), hs, state
            else:
                (logp, disfluency_logp), state = self.forward_one_step(
                    ys.unsqueeze(0),
                    ys_mask,
                    x.unsqueeze(0),
                    cache=state,
                    return_hs=return_hs,
                    return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad.unsqueeze(0) if disfluency_in_pad is not None else None,
                )
                return (logp.squeeze(0), disfluency_logp.squeeze(0)), state
        else:
            if return_hs:
                (logp, hs), state = self.forward_one_step(
                    ys.unsqueeze(0),
                    ys_mask,
                    x.unsqueeze(0),
                    cache=state,
                    return_hs=return_hs,
                    return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad.unsqueeze(0) if disfluency_in_pad is not None else None,
                )
                return logp.squeeze(0), hs, state
            else:
                logp, state = self.forward_one_step(
                    ys.unsqueeze(0),
                    ys_mask,
                    x.unsqueeze(0),
                    cache=state,
                    return_hs=return_hs,
                    return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad.unsqueeze(0) if disfluency_in_pad is not None else None,
                )
                return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        return_hs: bool = False,
        return_disfluency: bool = True,
        disfluency_in_pad: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch with disfluency detection.
        
        Args:
            ys: token sequences (batch, maxlen_out)
            states: decoder state cache per batch
            xs: encoder outputs (batch, maxlen_in, feat)
            return_hs: whether to return hidden states
            return_disfluency: whether to return disfluency predictions
            disfluency_in_pad: previous disfluency labels d<i (batch, maxlen_out)
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        
        if return_disfluency:
            if return_hs:
                (logp, disfluency_logp, hs), states = self.forward_one_step(
                    ys, ys_mask, xs, cache=batch_state, 
                    return_hs=return_hs, return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad
                )
            else:
                (logp, disfluency_logp), states = self.forward_one_step(
                    ys, ys_mask, xs, cache=batch_state, 
                    return_hs=return_hs, return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad
                )
        else:
            if return_hs:
                (logp, hs), states = self.forward_one_step(
                    ys, ys_mask, xs, cache=batch_state, 
                    return_hs=return_hs, return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad
                )
            else:
                logp, states = self.forward_one_step(
                    ys, ys_mask, xs, cache=batch_state, 
                    return_hs=return_hs, return_disfluency=return_disfluency,
                    disfluency_in_pad=disfluency_in_pad
                )

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        
        if return_disfluency:
            if return_hs:
                return (logp, disfluency_logp, hs), state_list
            else:
                return (logp, disfluency_logp), state_list
        else:
            if return_hs:
                return (logp, hs), state_list
            return logp, state_list
        
    def recognize_disfluency(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ):
        """推論時にdisfluency予測ラベル列を返すユーティリティ"""
        self.eval()
        with torch.no_grad():
            # ASRトークン予測
            (asr_out, disfluency_out), olens = self.forward(
                hs_pad, hlens, ys_in_pad, ys_in_lens, return_disfluency=True
            )
            # asr_out: (B, L, vocab_size), disfluency_out: (B, L, disfluency_classes)
            # トークンごとに最大値のインデックスを取得
            disfluency_preds = disfluency_out.argmax(dim=-1)  # (B, L)
            return disfluency_preds
        