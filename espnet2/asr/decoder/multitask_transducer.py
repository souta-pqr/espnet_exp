import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Any, Union, List
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.layers.multitask_joint_network import MultitaskJointNetwork, MultitaskTransducerLoss


class BeamSearchJointWrapper(nn.Module):
    """Wrapper for MultitaskJointNetwork to provide beam search compatibility
    
    This wrapper makes the multitask joint network compatible with standard 
    RNN-T beam search by extracting only the ASR scores when needed.
    """
    
    def __init__(self, multitask_joint_network: MultitaskJointNetwork):
        super().__init__()
        self.multitask_joint_network = multitask_joint_network
        
    def forward(
        self, 
        encoder_out: torch.Tensor, 
        decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for beam search compatibility
        
        Args:
            encoder_out: Encoder output - can be various shapes depending on beam search stage
            decoder_out: Decoder output - can be various shapes depending on beam search stage
            
        Returns:
            ASR scores tensor compatible with beam search (vocab_size,) for single predictions
        """
        # Store original shapes for debugging
        original_enc_shape = encoder_out.shape
        original_dec_shape = decoder_out.shape
        
        # Handle various encoder output formats
        if encoder_out.dim() == 1:
            # Single time step: (encoder_dim,) -> (1, 1, encoder_dim)
            encoder_out = encoder_out.unsqueeze(0).unsqueeze(0)
        elif encoder_out.dim() == 2:
            if encoder_out.size(0) == 1:
                # Already has batch: (1, encoder_dim) -> (1, 1, encoder_dim)
                encoder_out = encoder_out.unsqueeze(1)
            else:
                # Multiple time steps: (T, encoder_dim) -> (1, T, encoder_dim)
                encoder_out = encoder_out.unsqueeze(0)
        
        # Handle various decoder output formats  
        if decoder_out.dim() == 1:
            # Single prediction: (decoder_dim,) -> (1, 1, decoder_dim)
            decoder_out = decoder_out.unsqueeze(0).unsqueeze(0)
        elif decoder_out.dim() == 2:
            if decoder_out.size(0) == 1:
                # Batch of 1: (1, decoder_dim) -> (1, 1, decoder_dim)
                decoder_out = decoder_out.unsqueeze(1)
            else:
                # Multiple predictions: (U, decoder_dim) -> (1, U, decoder_dim)
                decoder_out = decoder_out.unsqueeze(0)
        
        # Call the multitask joint network
        outputs = self.multitask_joint_network.forward(encoder_out, decoder_out, return_disfluency=False)
        
        # Extract ASR scores
        asr_scores = outputs["asr_scores"]
        
        # For beam search compatibility, return (vocab_size,)
        # The joint network returns (B, T, U, vocab_size), we want (vocab_size,)
        if asr_scores.dim() == 4:
            # (B, T, U, vocab_size) -> (vocab_size,)
            asr_scores = asr_scores.squeeze(0).squeeze(0).squeeze(0)
        elif asr_scores.dim() == 3:
            # (T, U, vocab_size) -> (vocab_size,)
            asr_scores = asr_scores.squeeze(0).squeeze(0)
        elif asr_scores.dim() == 2:
            # (U, vocab_size) -> (vocab_size,)
            asr_scores = asr_scores.squeeze(0)
        
        # Ensure 1D output for beam search
        if asr_scores.dim() != 1:
            asr_scores = asr_scores.view(-1)
        
        return asr_scores


class MultitaskBeamSearchJoint(nn.Module):
    """Multitask joint network for beam search that returns both ASR and disfluency scores
    
    This is for future implementation of multitask beam search.
    """
    
    def __init__(self, multitask_joint_network: MultitaskJointNetwork):
        super().__init__()
        self.multitask_joint_network = multitask_joint_network
        
    def forward(
        self, 
        encoder_out: torch.Tensor, 
        decoder_out: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multitask beam search
        
        Args:
            encoder_out: Encoder output 
            decoder_out: Decoder output
            
        Returns:
            Dictionary with both ASR and disfluency scores
        """
        # Call explicit forward to ensure we receive a dict containing both
        # ASR and disfluency scores. Calling the module directly may invoke
        # a compatibility __call__ that returns only ASR tensor.
        outputs = self.multitask_joint_network.forward(
            encoder_out, decoder_out, return_disfluency=True
        )
        
        # Adjust shapes for beam search if needed
        if encoder_out.dim() == 1 and decoder_out.dim() == 1:
            # Squeeze extra dimensions for beam search
            if "asr_scores" in outputs:
                outputs["asr_scores"] = outputs["asr_scores"].squeeze(0).squeeze(0).squeeze(0)
            if "disfluency_scores" in outputs:
                outputs["disfluency_scores"] = outputs["disfluency_scores"].squeeze(0).squeeze(0).squeeze(0)
        
        return outputs


class MultitaskTransducerDecoder(TransducerDecoder):
    """Multitask Transducer Decoder extending the existing TransducerDecoder
    
    既存のTransducerDecoderを継承し、disfluency detection機能を追加
    """
    
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        # RNN decoder parameters (from parent class)
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 256,
        dropout: float = 0.1,
        dropout_embed: float = 0.1,
        embed_size: int = 256,
        embed_pad: int = 0,  # blank id
        # Joint network parameters
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
        # Disfluency detection parameters
        use_disfluency_detection: bool = True,
        disfluency_classes: int = 4,
        disfluency_projection_size: Optional[int] = None,
        disfluency_weight: float = 1.0,
        use_viterbi_alignment: bool = False,  # Enable Viterbi alignment for disfluency
        # Loss parameters
        blank_id: int = 0,
        trans_type: str = "warp_rnnt",
    ):
        """Initialize Multitask Transducer Decoder
        
        Args:
            vocab_size: Vocabulary size
            encoder_output_size: Encoder output dimension
            rnn_type: Type of RNN (lstm, gru, rnn)
            num_layers: Number of RNN layers
            hidden_size: Hidden size of RNN
            dropout: Dropout rate for RNN
            dropout_embed: Dropout rate for embedding
            embed_size: Embedding size
            embed_pad: Padding index for embedding
            joint_space_size: Joint network hidden size
            joint_activation_type: Activation type for joint network
            use_disfluency_detection: Enable disfluency detection
            disfluency_classes: Number of disfluency classes
            disfluency_projection_size: Optional projection size for disfluency
            disfluency_weight: Weight for disfluency loss
            use_viterbi_alignment: Use Viterbi alignment for frame-level labels
            blank_id: Blank token ID
            trans_type: Transducer loss type
        """
        # Initialize parent TransducerDecoder (RNN prediction network)
        super().__init__(
            vocab_size=vocab_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            dropout_embed=dropout_embed,
        )
        
        # Store additional parameters
        self.vocab_size = vocab_size
        self.encoder_output_size = encoder_output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size  # Store for potential future use
        self.embed_pad = embed_pad    # Store for potential future use
        self.blank_id = blank_id
        self.use_disfluency_detection = use_disfluency_detection
        self.disfluency_classes = disfluency_classes
        self.disfluency_weight = disfluency_weight
        self.use_viterbi_alignment = use_viterbi_alignment
        
        # =========================================================================
        # Paper Implementation: Disfluency Embedding (NEW - from paper Section 2.2)
        # =========================================================================
        # "we input d<i along with y<i. To feed these input representations
        #  to the network, token and disfluency input embeddings are summed,
        #  similar to token and segment embeddings in BERT."
        if self.use_disfluency_detection:
            self.disfluency_embedding = nn.Embedding(disfluency_classes, hidden_size)
        
        # Create multitask joint network (replaces simple linear joint)
        self.joint_network = MultitaskJointNetwork(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            decoder_output_size=hidden_size,
            joint_space_size=joint_space_size,
            joint_activation_type=joint_activation_type,
            use_disfluency_detection=use_disfluency_detection,
            disfluency_classes=disfluency_classes,
            disfluency_projection_size=disfluency_projection_size,
            dropout_rate=dropout,
        )
        
        # Create multitask loss function
        self.criterion = MultitaskTransducerLoss(
            blank_id=blank_id,
            use_disfluency_detection=use_disfluency_detection,
            disfluency_weight=disfluency_weight,
            trans_type=trans_type,
            use_viterbi_alignment=use_viterbi_alignment,
        )
        
        # Create joint network wrapper for beam search compatibility
        # This wrapper makes the joint network compatible with standard RNN-T beam search
        self._joint_network_wrapper = BeamSearchJointWrapper(self.joint_network)
        
        # For beam search compatibility: provide 'joint' attribute
        # Standard beam search expects decoder.joint to be callable
        self.joint = self._joint_network_wrapper
    
    def compute_joint(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        predicted_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Joint computation for encoder and decoder outputs with token-dependency mechanism
        
        Args:
            encoder_out: Encoder output (B, T, encoder_dim)
            decoder_out: Decoder output (B, U, decoder_dim)
            predicted_tokens: Ground-truth tokens for training (B, T, U)
                            or None for inference (will use argmax)
            
        Returns:
            Dictionary with ASR scores and optionally disfluency scores
            
        Token-Dependency Mechanism:
            For disfluency prediction, we use the token-dependency mechanism:
            p(di|X, y≤i) = softmax(W[E(yi); si] + b)
            During training: use ground-truth tokens (teacher forcing)
            During inference: use predicted tokens (argmax)
        """
        # Use the explicit forward() to get a dictionary containing both
        # ASR and disfluency scores with token-dependency mechanism
        return self.joint_network.forward(
            encoder_out, 
            decoder_out, 
            return_disfluency=True,
            predicted_tokens=predicted_tokens  # Pass tokens for token-dependency
        )
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        disfluency_labels: Optional[torch.Tensor] = None,
        # current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with loss calculation and Scheduled Sampling for token-dependency
        
        Args:
            encoder_out: Encoder output (B, T, encoder_dim)
            encoder_out_lens: Encoder output lengths (B,)
            labels: Target labels (B, L)
            label_lens: Target label lengths (B,)
            disfluency_labels: Disfluency labels (B, L) character-level
            current_epoch: Current training epoch for scheduled sampling (default: 0)
            
        Returns:
            Tuple of (loss, loss_dict)
            
        Scheduled Sampling Strategy:
            - Epoch 1-5: Pure teacher forcing (sampling_ratio=0.0)
            - Epoch 6-30: Gradually increase predicted tokens (0.0 → 0.5)
            - Epoch 31+: Use 50% predicted tokens (sampling_ratio=0.5)
            
        Paper Implementation:
            1. Token-Dependency Mechanism (Joint Network):
               - NEW: Mix predicted_tokens and ground-truth labels (scheduled sampling)
               - OLD: predicted_tokens = labels (pure teacher forcing)
               - p(di|X, yi) uses mixed tokens for robustness
            2. Disfluency Embedding (Decoder):
               - Input d<i along with y<i (Section 2.2)
               - BERT-style: token embedding + disfluency embedding
        """
        # Get decoder outputs for all label positions with disfluency embedding (paper Section 2.2)
        # During training: use ground-truth disfluency labels (teacher forcing)
        decoder_out = self.predict(labels, label_lens, disfluency_in_pad=disfluency_labels)
        
        # =========================================================================
        # IMPLEMENTATION SWITCH: Token-dependency mechanism vs Simple multitask
        # =========================================================================
        # CURRENT: Simple multitask (like exp 1029) - No token-dependency
        # ALTERNATIVE: Token-dependency with Scheduled Sampling (commented below)
        # =========================================================================
        
        # Token-dependency mechanism with PURE Teacher Forcing
        # We always provide ground-truth tokens to the joint network so that
        # E(yi) is learned from correct tokens only (no scheduled sampling).
        # Prepare tokens for token-dependency mechanism
        B, T = encoder_out.shape[0], encoder_out.shape[1]
        U = decoder_out.shape[1]

        # Create predicted_tokens tensor: (B, T, U)
        predicted_tokens = torch.zeros(B, T, U, dtype=torch.long, device=encoder_out.device)

        # Fill with ground-truth tokens (teacher forcing)
        for b in range(B):
            actual_len = min(label_lens[b].item(), labels.shape[1])
            for u in range(1, U):
                label_idx = u - 1
                if label_idx < actual_len:
                    token_value = labels[b, label_idx].item()
                    if token_value >= 0:
                        predicted_tokens[b, :, u] = token_value

        # Joint computation with token-dependency (using ground-truth tokens)
        joint_out = self.compute_joint(encoder_out, decoder_out, predicted_tokens=predicted_tokens)
        asr_scores = joint_out["asr_scores"]
        disfluency_scores = joint_out.get("disfluency_scores", None)
        
        # Calculate loss
        # Convert labels to int32 for RNN-Transducer loss
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        label_lens = label_lens.to(torch.int32)
        labels = labels.to(torch.int32)
        
        # Safety check: ensure labels are within valid range [0, vocab_size-1]
        # This prevents CUDA index out of bounds errors
        vocab_size = asr_scores.shape[-1]
        if torch.any(labels >= vocab_size) or torch.any(labels < 0):
            # Clamp invalid labels to valid range
            labels = labels.clamp(0, vocab_size - 1)
        
        # CTC不要！asr_scoresからTransducerのアライメントを抽出
        loss, loss_dict = self.criterion(
            asr_scores=asr_scores,  # Transducerのjoint logitsを使用
            targets=labels,
            encoder_lens=encoder_out_lens,
            target_lens=label_lens,
            disfluency_scores=disfluency_scores,
            disfluency_targets=disfluency_labels,
        )
        
        return loss, loss_dict
    
    def predict(
        self,
        labels: torch.Tensor,
        label_lens: torch.Tensor,
        disfluency_in_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict decoder outputs for given labels with disfluency embedding
        
        Args:
            labels: Target labels (B, L)
            label_lens: Label lengths (B,)
            disfluency_in_pad: Previous disfluency labels d<i (B, L) - optional
            
        Returns:
            Decoder outputs (B, U, decoder_dim)
            
        Paper Implementation (Section 2.2):
            "we input d<i along with y<i. To feed these input representations
             to the network, token and disfluency input embeddings are summed,
             similar to token and segment embeddings in BERT."
        """
        # Add blank at the beginning for transducer
        batch_size, max_len = labels.shape
        device = labels.device
        
        # Fix invalid label indices and ensure proper data type
        if torch.any(labels >= self.vocab_size) or torch.any(labels < 0):
            # Log warning about invalid labels
            invalid_count = torch.sum((labels >= self.vocab_size) | (labels < 0))
            if invalid_count > 0:
                # Replace invalid indices: -1 (ignore_id) → 0 (blank), others clamped to valid range  
                labels = torch.where(labels == -1, torch.zeros_like(labels), labels)
                labels = torch.clamp(labels, 0, self.vocab_size - 1)
        
        # Ensure labels are int32 for RNN-Transducer
        labels = labels.to(torch.int32)
        
        # Create input with blank prepended
        ys_in = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.int32, device=device),
            labels
        ], dim=1)
        
        # Get decoder outputs with proper device handling
        batch_size = ys_in.size(0)
        device = ys_in.device
        
        # Initialize state with correct device
        if hasattr(self, "device"):
            self.device = device
        
        # =========================================================================
        # Paper Implementation: BERT-style Embedding Summation
        # =========================================================================
        # Token embedding
        dec_embed = self.embed(ys_in)  # (B, U, hidden_size)
        
        # Add disfluency embedding if using disfluency detection (paper Section 2.2)
        if self.use_disfluency_detection:
            if disfluency_in_pad is not None:
                # SAFETY: Clamp disfluency labels to valid range [0, disfluency_classes-1]
                if (disfluency_in_pad < 0).any() or (disfluency_in_pad >= self.disfluency_classes).any():
                    print(f"[WARNING] Invalid disfluency labels detected!")
                    print(f"  Min: {disfluency_in_pad.min().item()}, Max: {disfluency_in_pad.max().item()}")
                    print(f"  Valid range: [0, {self.disfluency_classes-1}]")
                    disfluency_in_pad = torch.clamp(disfluency_in_pad, 0, self.disfluency_classes - 1)
                # Use provided disfluency labels (training: ground-truth d<i)
                # Prepend fluent (0) at the beginning to match ys_in format
                dysfl_in = torch.cat([
                    torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                    disfluency_in_pad
                ], dim=1)
                dysfl_emb = self.disfluency_embedding(dysfl_in)  # (B, U, hidden_size)
            else:
                # Default: all fluent (class 0)
                dysfl_in = torch.zeros_like(ys_in, dtype=torch.long, device=device)
                dysfl_emb = self.disfluency_embedding(dysfl_in)  # (B, U, hidden_size)
            
            # Sum token and disfluency embeddings (BERT-style: token + segment)
            dec_embed = dec_embed + dysfl_emb
        
        # Apply dropout
        dec_embed = self.dropout_embed(dec_embed)
        
        # Initialize states on the same device as input
        if self.dtype == "lstm":
            h_n = torch.zeros(self.dlayers, batch_size, self.dunits, device=device)
            c_n = torch.zeros(self.dlayers, batch_size, self.dunits, device=device)
            init_state = (h_n, c_n)
        else:
            h_n = torch.zeros(self.dlayers, batch_size, self.dunits, device=device)
            init_state = (h_n, None)
        
        decoder_out, _ = self.rnn_forward(dec_embed, init_state)
        
        return decoder_out
    
    # NOTE: We don't implement batch_score for RNN-Transducer
    # because it uses non-batch beam search by default

    # NOTE: We don't implement create_batch_states and select_state for RNN-Transducer
    # because it uses non-batch beam search by default

    def score(
        self, hyp: Any, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """One-step forward hypothesis (for BeamSearchTransducer).

        Args:
            hyp: Hypothesis with yseq and dec_state
            cache: Cache for decoder outputs

        Returns:
            dec_out: Decoder output (D_dec,)
            new_state: New decoder state
            label: Label ID for LM (1,)
        """
        label = torch.full(
            (1, 1), hyp.yseq[-1], dtype=torch.long, device=next(self.parameters()).device
        )
        
        str_labels = "_".join(list(map(str, hyp.yseq)))
        
        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            # =========================================================================
            # Paper Implementation: BERT-style Embedding (inference mode)
            # =========================================================================
            # Token embedding
            dec_emb = self.embed(label)
            
            # Add disfluency embedding if using disfluency detection
            # Inference mode: use default fluent (0)
            if self.use_disfluency_detection:
                device = label.device
                dysfl_label = torch.zeros_like(label, dtype=torch.long, device=device)
                dysfl_emb = self.disfluency_embedding(dysfl_label)
                dec_emb = dec_emb + dysfl_emb  # BERT-style summation
            
            dec_out, dec_state = self.rnn_forward(dec_emb, hyp.dec_state)
            cache[str_labels] = (dec_out, dec_state)
        
        return dec_out[0][0], dec_state, label[0]
    
    def score_with_encoder(
        self, yseq: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """One-step forward for beam search (compatible with standard BeamSearch).

        For non-Transducer beam search, this method must return vocabulary scores.
        We call the joint network internally to get ASR logits.

        Args:
            yseq: Label sequence (1D tensor of token IDs)
            state: Previous decoder state (tuple of h, c for LSTM)
            x: Encoder output (batch, time, encoder_output_size) - needed for joint network

        Returns:
            Tuple of (logits, new_state):
                - logits: ASR vocabulary logits (vocab_size,)
                - new_state: New decoder state (tuple of h, c)
        """
        # Get the last label from sequence
        device = next(self.parameters()).device
        
        # Handle different yseq formats
        if isinstance(yseq, list):
            last_label = yseq[-1]
        elif isinstance(yseq, torch.Tensor):
            last_label = yseq[-1].item() if yseq.dim() > 0 else yseq.item()
        else:
            last_label = int(yseq)
        
        # Create input tensor: (1, 1) for batch and time dimensions
        label = torch.full((1, 1), last_label, dtype=torch.long, device=device)
        
        # =========================================================================
        # Paper Implementation: BERT-style Embedding (inference mode)
        # =========================================================================
        # Token embedding
        dec_emb = self.embed(label)  # (1, 1, embed_size)
        
        # Add disfluency embedding if using disfluency detection
        # Inference mode: use default fluent (0) since we don't have d<i yet
        if self.use_disfluency_detection:
            dysfl_label = torch.zeros_like(label, dtype=torch.long, device=device)
            dysfl_emb = self.disfluency_embedding(dysfl_label)
            dec_emb = dec_emb + dysfl_emb  # BERT-style summation
        
        # Ensure state is on the correct device
        if state is not None:
            if isinstance(state, tuple):
                state = tuple(s.to(dec_emb.device) if s is not None else None for s in state)
            else:
                state = state.to(dec_emb.device)
        else:
            # Initialize state on the correct device
            state = self.init_state(1)
            if isinstance(state, tuple):
                state = tuple(s.to(dec_emb.device) if s is not None else None for s in state)
            else:
                state = state.to(dec_emb.device)
        
        # RNN forward
        dec_out, new_state = self.rnn_forward(dec_emb, state)
        
        # For non-Transducer beam search, we need to return vocabulary scores
        # Call joint network to get ASR logits
        # 
        # IMPORTANT: This is a simplified implementation for compatibility with
        # standard BeamSearch. For proper Transducer decoding, use BeamSearchTransducer.
        # 
        # Strategy: Use a representative encoder frame instead of all frames
        # Options:
        #   1. Mean pooling: Average of all frames (most robust)
        #   2. First frame: Beginning of utterance
        #   3. Last frame: End of utterance
        # We use mean pooling for best coverage
        
        # Ensure encoder output has proper dimensions
        if x.dim() == 2:
            # (time, encoder_dim) -> (1, time, encoder_dim)
            enc_out = x.unsqueeze(0)
        elif x.dim() == 3:
            # (batch, time, encoder_dim) - already correct
            enc_out = x
        else:
            # x is 1D: (encoder_dim,) -> (1, 1, encoder_dim)
            enc_out = x.view(1, 1, -1)
        
        # dec_out has shape (1, 1, decoder_dim)
        
        # Use mean-pooled encoder output as representative frame
        # This is much faster than processing all frames
        enc_mean = enc_out.mean(dim=1, keepdim=True)  # (1, 1, encoder_dim)
        
        # Call joint network with mean encoder output
        if hasattr(self, '_joint_network_wrapper'):
            # Use wrapper
            asr_logits = self._joint_network_wrapper(enc_mean, dec_out)  # (vocab_size,)
        else:
            # Call joint network directly
            outputs = self.joint_network.forward(enc_mean, dec_out, return_disfluency=False)
            # outputs["asr_scores"]: (1, 1, 1, vocab_size)
            asr_logits = outputs["asr_scores"].squeeze(0).squeeze(0).squeeze(0)  # (vocab_size,)
        
        # Return logits and new state
        return asr_logits, new_state
    
    def batch_score(
        self, 
        hyps: List[Any], 
        states: List[Any], 
        xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Batch forward for beam search (for Transducer).
        
        NOTE: This method is called by BeamSearchTransducer, not by regular BeamSearch.
        It processes multiple hypotheses in parallel.
        
        Args:
            hyps: List of hypotheses (each containing yseq)
            states: List of decoder states for each hypothesis
            xs: Encoder output (not used in prediction network, but required for interface)
        
        Returns:
            Tuple of (dec_outs, new_states):
                - dec_outs: Decoder outputs (batch, decoder_output_size)
                - new_states: New decoder states (list of tuples)
        """
        device = next(self.parameters()).device
        batch_size = len(hyps)
        
        # Get last labels from all hypotheses
        labels = []
        for hyp in hyps:
            if hasattr(hyp, 'yseq'):
                last_label = hyp.yseq[-1]
            elif isinstance(hyp, list):
                last_label = hyp[-1]
            elif isinstance(hyp, torch.Tensor):
                last_label = hyp[-1].item() if hyp.dim() > 0 else hyp.item()
            else:
                last_label = int(hyp)
            labels.append(last_label)
        
        # Create batch input
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(1)  # (batch, 1)
        
        # =========================================================================
        # Paper Implementation: BERT-style Embedding (inference mode)
        # =========================================================================
        # Token embedding
        dec_emb = self.embed(labels_tensor)  # (batch, 1, embed_size)
        
        # Add disfluency embedding if using disfluency detection
        # Inference mode: use default fluent (0)
        if self.use_disfluency_detection:
            dysfl_labels = torch.zeros_like(labels_tensor, dtype=torch.long, device=device)
            dysfl_emb = self.disfluency_embedding(dysfl_labels)
            dec_emb = dec_emb + dysfl_emb  # BERT-style summation
        
        # Process states
        new_states = []
        dec_outs = []
        
        for i in range(batch_size):
            state = states[i] if i < len(states) else None
            
            # Ensure state is on the correct device
            if state is not None:
                if isinstance(state, tuple):
                    state = tuple(s.to(dec_emb.device) if s is not None else None for s in state)
                else:
                    state = state.to(dec_emb.device)
            else:
                # Initialize state
                state = self.init_state(1)
                if isinstance(state, tuple):
                    state = tuple(s.to(dec_emb.device) if s is not None else None for s in state)
                else:
                    state = state.to(dec_emb.device)
            
            # RNN forward for this hypothesis
            dec_out, new_state = self.rnn_forward(dec_emb[i:i+1], state)
            dec_outs.append(dec_out[0, 0])  # Remove batch and time dims
            new_states.append(new_state)
        
        # Stack outputs
        dec_outs_tensor = torch.stack(dec_outs, dim=0)  # (batch, decoder_output_size)
        
        return dec_outs_tensor, new_states
            
    def init_state(
        self, batch_size: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Initialize decoder states for beam search.

        Args:
            batch_size: Batch size (usually 1 for beam search) or encoder output tensor.
                       When called from beam search, this is the encoder output tensor.

        Returns:
            Initial decoder hidden states (tuple of h, c for LSTM).
        """
        # For beam search compatibility:
        # - If batch_size is a tensor (encoder output), always use batch_size=1
        # - If batch_size is an int, use it directly
        if isinstance(batch_size, torch.Tensor):
            # This is encoder output from beam search: always use batch_size=1
            actual_batch_size = 1
            device = batch_size.device
        elif isinstance(batch_size, int):
            actual_batch_size = batch_size
            device = next(self.parameters()).device
        else:
            actual_batch_size = 1
            device = next(self.parameters()).device
        
        h_n = torch.zeros(
            self.dlayers,
            actual_batch_size,
            self.dunits,
            device=device,
        )

        if self.dtype == "lstm":
            c_n = torch.zeros(
                self.dlayers,
                actual_batch_size,
                self.dunits,
                device=device,
            )
            return (h_n, c_n)
        else:
            return (h_n, None)
            
    def inference(
        self,
        encoder_out: torch.Tensor,
        beam_size: int = 10,
        max_sym_exp: int = 2,
        u_max: int = 50,
        return_disfluency: bool = False,
    ) -> Dict[str, Any]:
        """Inference with beam search
        
        Args:
            encoder_out: Encoder output (B, T, encoder_dim)
            beam_size: Beam size
            max_sym_exp: Maximum symbols per expansion
            u_max: Maximum output length
            return_disfluency: Whether to return disfluency predictions
            
        Returns:
            Dictionary containing:
                - "best_hyp": Best hypothesis with token IDs
                - "disfluency_scores": Disfluency scores (if return_disfluency=True)
        """
        # For now, implement a simple greedy search
        # This can be extended to full beam search later
        
        batch_size = encoder_out.size(0)
        max_time = encoder_out.size(1)
        device = encoder_out.device
        
        results = []
        
        for b in range(batch_size):
            # Initialize decoder state
            ys = torch.zeros(1, 1, dtype=torch.long, device=device)  # Start with blank
            
            # Greedy search
            output_tokens = []
            disfluency_preds = [] if return_disfluency else None
            
            for t in range(max_time):
                # Get encoder output at time t
                enc_t = encoder_out[b:b+1, t:t+1, :]  # (1, 1, encoder_dim)
                
                # Get decoder output
                dec_out = self.predict(ys, torch.tensor([ys.size(1)], device=device))
                dec_t = dec_out[:, -1:, :]  # (1, 1, decoder_dim)
                
                # Joint computation
                # Ensure we call forward(...) to get a dict with both heads
                joint_out = self.joint_network.forward(enc_t, dec_t, return_disfluency=True)
                asr_scores = joint_out["asr_scores"]  # (1, 1, 1, vocab_size)
                
                # Get best token
                best_token = torch.argmax(asr_scores, dim=-1).item()
                
                if best_token != self.blank_id:  # Not blank
                    output_tokens.append(best_token)
                    
                    # Get disfluency prediction if requested
                    if return_disfluency and "disfluency_scores" in joint_out:
                        disf_scores = joint_out["disfluency_scores"]  # (1, 1, 1, disf_classes)
                        best_disf = torch.argmax(disf_scores, dim=-1).item()
                        disfluency_preds.append(best_disf)
                    
                    # Update decoder input
                    new_token = torch.tensor([[best_token]], device=device)
                    ys = torch.cat([ys, new_token], dim=1)
            
            result = {"yseq": output_tokens}
            if return_disfluency and disfluency_preds is not None:
                result["disfluency"] = disfluency_preds
                
            results.append(result)
        
        # Return the first result for single batch
        if batch_size == 1:
            return results[0]
        else:
            return {"batch_results": results}
    
    def create_beam_search_transducer(
        self,
        beam_size: int = 10,
        lm: Optional[nn.Module] = None,
        lm_weight: float = 0.0,
        output_disfluency: bool = True,
        score_norm: bool = True,
        nbest: int = 1,
    ):
        """Create BeamSearchTransducer instance for this decoder.
        
        Args:
            beam_size: Beam width
            lm: Language model (optional)
            lm_weight: Language model weight
            output_disfluency: Whether to output disfluency predictions
            score_norm: Whether to normalize scores by length
            nbest: Number of best hypotheses to return
        
        Returns:
            BeamSearchTransducer instance
        """
        from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
        
        return BeamSearchTransducer(
            decoder=self,
            joint_network=self.joint_network,
            beam_size=beam_size,
            lm=lm,
            lm_weight=lm_weight,
            search_type="default",
            score_norm=score_norm,
            nbest=nbest,
        )
    