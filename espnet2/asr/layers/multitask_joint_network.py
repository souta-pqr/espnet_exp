import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


def transducer_viterbi_alignment(
    joint_logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """Compute Viterbi alignment from Transducer joint network logits.
    
    This extracts the most likely alignment path from RNN-Transducer,
    NOT from CTC. This respects the Transducer architecture.
    
    Args:
        joint_logits: Joint network output (B, T, U, vocab_size)
                     where U = max_target_length + 1
        targets: Target token sequences (B, L)
        input_lengths: Input sequence lengths (B,)
        target_lengths: Target sequence lengths (B,)
        blank_id: ID of the blank token (default: 0)
    
    Returns:
        alignments: Frame-to-token alignment (B, T)
                   Each element indicates which target token (0 to L-1)
                   the frame is aligned to, or -1 for blank/padding
    
    Algorithm:
        Uses Viterbi algorithm on Transducer lattice:
        - State (t, u): at time t, having emitted u tokens
        - Blank transition: (t, u) -> (t+1, u)
        - Label transition: (t, u) -> (t, u+1) emitting targets[u]
    """
    batch_size, max_time, max_u, vocab_size = joint_logits.shape
    device = joint_logits.device
    
    # Output alignments: (B, T), initialized to -1 (blank/padding)
    alignments = torch.full((batch_size, max_time), -1, dtype=torch.long, device=device)
    
    # Convert to log probabilities
    log_probs = F.log_softmax(joint_logits, dim=-1)
    
    for b in range(batch_size):
        T = input_lengths[b].item()
        U = target_lengths[b].item() + 1  # +1 for initial blank state
        
        if target_lengths[b].item() == 0:
            continue
        
        # Get target sequence for this batch item
        target_seq = targets[b, :target_lengths[b]]  # (L,)
        
        # Viterbi forward pass on Transducer lattice
        # State (t, u): at acoustic frame t, having predicted u tokens (0 to L)
        # viterbi[t, u] = max log probability of reaching state (t, u)
        viterbi = torch.full((T, U), float('-inf'), device=device)
        backpointer = torch.zeros((T, U, 2), dtype=torch.long, device=device)  # Store (prev_t, prev_u)
        
        # Initialize: start at (0, 0) with probability 1 (log prob = 0)
        viterbi[0, 0] = 0.0
        
        # Fill first column (t=0): can only emit labels at t=0
        for u in range(1, min(U, T + 1)):
            if u <= U - 1:
                # Emit target_seq[u-1] to go from (0, u-1) to (0, u)
                label_id = target_seq[u - 1].item()
                viterbi[0, u] = viterbi[0, u - 1] + log_probs[b, 0, u - 1, label_id]
                backpointer[0, u] = torch.tensor([0, u - 1], device=device)
        
        # Forward pass
        for t in range(1, T):
            for u in range(U):
                candidates = []
                pointers = []
                
                # Option 1: Blank transition from (t-1, u) to (t, u)
                # Stay at same label position, advance time
                blank_score = viterbi[t - 1, u] + log_probs[b, t - 1, u, blank_id]
                candidates.append(blank_score)
                pointers.append([t - 1, u])
                
                # Option 2: Label transition from (t, u-1) to (t, u)
                # Emit target_seq[u-1], stay at same time
                if u > 0:
                    label_id = target_seq[u - 1].item()
                    label_score = viterbi[t, u - 1] + log_probs[b, t, u - 1, label_id]
                    candidates.append(label_score)
                    pointers.append([t, u - 1])
                
                # Find best candidate
                if candidates:
                    best_idx = torch.tensor(candidates).argmax().item()
                    viterbi[t, u] = candidates[best_idx]
                    backpointer[t, u] = torch.tensor(pointers[best_idx], device=device)
        
        # Backtracking: Start from (T-1, U-1) which is the final state
        # We need to reach having emitted all U-1 labels
        current_t = T - 1
        current_u = U - 1
        
        # Find the actual final state (might not be exactly at T-1, U-1)
        # due to trailing blanks
        best_final_score = float('-inf')
        best_final_t = T - 1
        best_final_u = U - 1
        
        # Check last few time steps for final state
        for t in range(max(0, T - 5), T):
            if viterbi[t, U - 1] > best_final_score:
                best_final_score = viterbi[t, U - 1]
                best_final_t = t
                best_final_u = U - 1
        
        current_t = best_final_t
        current_u = best_final_u
        
        # Backtrack to get path
        path = [(current_t, current_u)]
        while current_t > 0 or current_u > 0:
            prev_t, prev_u = backpointer[current_t, current_u].tolist()
            if prev_t == current_t and prev_u == current_u:
                break  # Avoid infinite loop
            path.append((prev_t, prev_u))
            current_t, prev_u = prev_t, prev_u
            current_u = prev_u
        
        path.reverse()
        
        # Convert path to frame-level alignment
        # For each time step, record which token (u-1) it corresponds to
        for i in range(len(path)):
            t, u = path[i]
            if u > 0:  # u=0 is the initial blank state
                # Frame t corresponds to token (u-1)
                alignments[b, t] = u - 1
            # else: remains -1 (blank)
        
        # Fill in gaps: if a time step is not in the path, inherit from previous
        current_token = -1
        for t in range(T):
            if alignments[b, t] >= 0:
                current_token = alignments[b, t].item()
            elif current_token >= 0 and t < T:
                # Assign to current token (blank frames during token emission)
                alignments[b, t] = current_token
    
    return alignments


def align_disfluency_labels_with_transducer(
    joint_logits: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    isdysfl: torch.Tensor,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align disfluency labels to frames using Transducer's own alignment.
    
    This uses the Transducer joint network outputs to determine alignment,
    NOT CTC. This respects the Transducer architecture.
    
    Args:
        joint_logits: Joint network output (B, T, U, vocab_size)
        encoder_out_lens: Encoder lengths (B,)
        text: Target text tokens (B, L)
        text_lengths: Text lengths (B,)
        isdysfl: Disfluency labels at character level (B, L)
        blank_id: Blank token ID
    
    Returns:
        frame_level_labels: Disfluency labels at frame level (B, T)
                           -100 for blank/padding positions
        frame_to_char_idx: Character index for each frame (B, T)
                          -1 for blank/padding positions
    """
    B, T, U, V = joint_logits.shape
    device = joint_logits.device
    
    # Get Viterbi alignment from Transducer: (B, T)
    # alignments[b, t] = which token (0 to L-1) frame t is aligned to, or -1 for blank
    alignments = transducer_viterbi_alignment(
        joint_logits=joint_logits,
        targets=text,
        input_lengths=encoder_out_lens,
        target_lengths=text_lengths,
        blank_id=blank_id,
    )
    
    # Create frame-level disfluency labels and character indices
    frame_level_labels = torch.full((B, T), -100, dtype=torch.long, device=device)
    frame_to_char_idx = torch.full((B, T), -1, dtype=torch.long, device=device)
    
    for b in range(B):
        T_b = encoder_out_lens[b].item()
        L_b = text_lengths[b].item()
        
        for t in range(T_b):
            token_idx = alignments[b, t].item()
            
            # CRITICAL: Check token_idx bounds BEFORE accessing isdysfl
            if token_idx >= L_b:
                # token_idx out of bounds - this should not happen!
                print(f"ERROR: token_idx={token_idx} >= L_b={L_b} at batch {b}, frame {t}")
                frame_level_labels[b, t] = -100
                frame_to_char_idx[b, t] = -1
                continue
                
            if token_idx >= 0 and token_idx < L_b:
                # This frame is aligned to a token
                # Get disfluency label value
                dysfl_label = isdysfl[b, token_idx].item()
                
                # Handle padding value (-1) and invalid values
                # -1 is used as padding in isdysfl data
                if dysfl_label == -1:
                    # Padding token - mark as ignore_index
                    frame_level_labels[b, t] = -100
                    frame_to_char_idx[b, t] = -1
                elif dysfl_label < 0 or dysfl_label >= 4:
                    # Invalid value - clamp to valid range and warn
                    print(f"WARNING: Invalid disfluency label {dysfl_label} at batch {b}, token {token_idx}. Clamping to [0, 3].")
                    dysfl_label = max(0, min(3, dysfl_label))
                    frame_level_labels[b, t] = dysfl_label
                    frame_to_char_idx[b, t] = token_idx
                else:
                    # Valid label [0, 3]
                    frame_level_labels[b, t] = dysfl_label
                    frame_to_char_idx[b, t] = token_idx
            # else: remains -100 (blank or padding)
    
    return frame_level_labels, frame_to_char_idx


class MultitaskJointNetwork(nn.Module):
    """Multitask Joint Network for RNN-Transducer with Token-Dependency Mechanism
    
    This implements the token-dependency mechanism for disfluency detection:
        p(di|X, y≤i) = softmax(W[E(yi); si] + b)
    
    Where:
        - E(yi): embedding of the predicted token yi
        - si: joint network hidden state (combination of encoder and decoder)
        - [E(yi); si]: concatenation of token embedding and hidden state
    
    Reference: "Streaming Joint Speech Recognition and Disfluency Detection"
    """
    
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
        # Disfluency specific parameters
        use_disfluency_detection: bool = True,
        disfluency_classes: int = 4,
        disfluency_projection_size: Optional[int] = None,
        dropout_rate: float = 0.1,
    ):
        """Initialize Multitask Joint Network with Token-Dependency Mechanism
        
        Args:
            vocab_size: Size of vocabulary
            encoder_output_size: Output dimension from encoder
            decoder_output_size: Output dimension from decoder
            joint_space_size: Dimension of joint space
            joint_activation_type: Activation function type
            use_disfluency_detection: Whether to use disfluency detection
            disfluency_classes: Number of disfluency classes (default: 4)
                0: fluent, 1: filler, 2: reparandum, 3: interjection
            disfluency_projection_size: Optional separate projection for disfluency
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.use_disfluency_detection = use_disfluency_detection
        self.disfluency_classes = disfluency_classes if use_disfluency_detection else 0
        self.joint_space_size = joint_space_size
        self.encoder_size = encoder_output_size  # Save for disfluency projections
        self.decoder_size = decoder_output_size  # Save for disfluency projections
        
        # Activation function selection
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        self.activation = activations.get(joint_activation_type, nn.Tanh())
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # ============================================================
        # ASR Joint Network (Standard RNN-T)
        # ============================================================
        self.encoder_proj = nn.Linear(encoder_output_size, joint_space_size)
        self.decoder_proj = nn.Linear(decoder_output_size, joint_space_size)
        self.asr_output_layer = nn.Linear(joint_space_size, vocab_size)
        
        # ============================================================
        # Token-Dependency Mechanism for Disfluency Detection
        # ============================================================
        if self.use_disfluency_detection:
            # =====================================================================
            # IMPLEMENTATION: Simple multitask (exp 1029 style)
            # Separate projections for encoder and decoder, no token embedding
            # =====================================================================
            self.token_embedding = None  # Disabled for simple multitask
            
            # Separate projections for disfluency (exp 1029 style)
            proj_size = disfluency_projection_size if disfluency_projection_size else joint_space_size
            self.disfluency_acoustic_proj = nn.Linear(self.encoder_size, proj_size)
            self.disfluency_decoder_proj = nn.Linear(self.decoder_size, proj_size)
            
            # Disfluency output layer
            self.disfluency_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(proj_size, disfluency_classes)
            )
            
            # =====================================================================
            # COMMENTED: Token-dependency mechanism implementation
            # =====================================================================
            # Uncomment to enable token-dependency mechanism
            """
            # Token embedding layer: E(yi)
            # This embeds the predicted token for disfluency prediction
            self.token_embedding = nn.Embedding(vocab_size, joint_space_size)
            
            # Disfluency prediction layer: W[E(yi); si] + b
            # Input: concatenation of token embedding and joint hidden state
            # Output: disfluency class scores
            disfluency_input_size = joint_space_size + joint_space_size  # E(yi) + si
            
            if disfluency_projection_size:
                # Two-layer projection for disfluency with token-dependency
                self.disfluency_output_layer = nn.Sequential(
                    nn.Linear(disfluency_input_size, disfluency_projection_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(disfluency_projection_size, disfluency_classes)
                )
            else:
                # Direct projection to disfluency classes
                self.disfluency_output_layer = nn.Linear(
                    disfluency_input_size, disfluency_classes
                )
            """
        else:
            self.token_embedding = None
            self.disfluency_output_layer = None
            self.disfluency_acoustic_proj = None
            self.disfluency_decoder_proj = None
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        return_disfluency: bool = True,
        predicted_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of joint network with token-dependency mechanism
        
        Args:
            encoder_out: Encoder output (B, T, encoder_dim) or (encoder_dim,) for beam search
            decoder_out: Decoder output (B, U, decoder_dim) or (decoder_dim,) for beam search
            return_disfluency: Whether to return disfluency predictions
            predicted_tokens: Predicted token IDs for token-dependency mechanism
                            (B, T, U) for training or None for inference (will use argmax)
            
        Returns:
            Dictionary containing:
                - "asr_scores": ASR logits (B, T, U, vocab_size)
                - "disfluency_scores": Disfluency logits (B, T, U, disfluency_classes)
                  [with token-dependency mechanism: uses predicted tokens]
        
        Token-Dependency Mechanism:
            p(di|X, y≤i) = softmax(W[E(yi); si] + b)
            - E(yi): embedding of predicted token
            - si: joint hidden state (encoder + decoder fusion)
            - [E(yi); si]: concatenation
        """
        # Handle different input shapes (training vs beam search)
        if encoder_out.dim() == 3:
            # Training mode: (B, T, encoder_dim)
            B, T, encoder_dim = encoder_out.shape
            _, U, decoder_dim = decoder_out.shape
            
            # ============================================================
            # ASR Joint Network (Standard RNN-T)
            # ============================================================
            # encoder_out: (B, T, 1, joint_space_size)
            encoder_proj = self.encoder_proj(encoder_out).unsqueeze(2)
            # decoder_out: (B, 1, U, joint_space_size)
            decoder_proj = self.decoder_proj(decoder_out).unsqueeze(1)
            
        elif encoder_out.dim() == 1:
            # Beam search mode: (encoder_dim,) and (decoder_dim,)
            encoder_dim = encoder_out.shape[0]
            decoder_dim = decoder_out.shape[0]
            B, T, U = 1, 1, 1
            
            # Add batch dimensions for processing
            # encoder_out: (1, 1, 1, joint_space_size)
            encoder_proj = self.encoder_proj(encoder_out.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            # decoder_out: (1, 1, 1, joint_space_size)
            decoder_proj = self.decoder_proj(decoder_out.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            
        elif encoder_out.dim() == 2:
            # Inference mode with batch dimension: (B, encoder_dim) and (B, decoder_dim)
            # This occurs in predict_disfluency when called with squeezed encoder output
            B = encoder_out.shape[0]
            encoder_dim = encoder_out.shape[1]
            decoder_dim = decoder_out.shape[1]
            T, U = 1, 1
            
            # Add time/label dimensions for processing
            # encoder_out: (B, 1, 1, joint_space_size)
            encoder_proj = self.encoder_proj(encoder_out).unsqueeze(1).unsqueeze(1)
            # decoder_out: (B, 1, 1, joint_space_size)
            decoder_proj = self.decoder_proj(decoder_out).unsqueeze(1).unsqueeze(1)
            
        else:
            raise ValueError(f"Unsupported encoder_out shape: {encoder_out.shape}")
        
        # ASR Joint representation
        asr_joint = encoder_proj + decoder_proj  # (B, T, U, joint_space_size)
        asr_joint = self.activation(asr_joint)
        asr_joint = self.dropout(asr_joint)
        
        # ASR output: (B, T, U, vocab_size)
        asr_scores = self.asr_output_layer(asr_joint)
        
        outputs = {"asr_scores": asr_scores}
        
        # ============================================================
        # Disfluency Detection
        # ============================================================
        if self.use_disfluency_detection and return_disfluency:
            # =====================================================================
            # IMPLEMENTATION: Simple multitask (exp 1029 style)
            # Disfluency: p(di|X, y<i) using encoder + decoder states only
            # =====================================================================
            
            # Project encoder and decoder outputs separately for disfluency
            # Reuse encoder_proj and decoder_proj shapes but with separate projections
            if encoder_out.dim() == 3:
                # Training mode
                disfluency_acoustic = self.disfluency_acoustic_proj(encoder_out).unsqueeze(2)  # (B, T, 1, proj)
                disfluency_decoder = self.disfluency_decoder_proj(decoder_out).unsqueeze(1)    # (B, 1, U, proj)
            elif encoder_out.dim() == 1:
                # Beam search mode
                disfluency_acoustic = self.disfluency_acoustic_proj(encoder_out.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                disfluency_decoder = self.disfluency_decoder_proj(decoder_out.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
            else:  # dim == 2
                # Inference mode
                disfluency_acoustic = self.disfluency_acoustic_proj(encoder_out).unsqueeze(1).unsqueeze(1)
                disfluency_decoder = self.disfluency_decoder_proj(decoder_out).unsqueeze(1).unsqueeze(1)
            
            disfluency_joint = disfluency_acoustic + disfluency_decoder
            
            # Classify disfluency
            disfluency_scores = self.disfluency_output_layer(disfluency_joint)  # (B, T, U, disfluency_classes)
            outputs["disfluency_scores"] = disfluency_scores
            
            # =====================================================================
            # COMMENTED: Token-dependency mechanism
            # =====================================================================
            # Uncomment to enable token-dependency mechanism
            """
            # Step 1: Get predicted tokens
            if predicted_tokens is None:
                # Inference mode: use argmax of ASR scores
                predicted_tokens = asr_scores.argmax(dim=-1)  # (B, T, U)
            
            # Safety check: clamp token IDs to valid range [0, vocab_size-1]
            predicted_tokens = predicted_tokens.clamp(0, self.vocab_size - 1)
            
            # Step 2: Get token embeddings E(yi)
            token_embeddings = self.token_embedding(predicted_tokens)  # (B, T, U, joint_space_size)
            
            # Step 3: Token-dependency mechanism: [E(yi); si]
            # si = asr_joint (the joint hidden state)
            # Concatenate token embedding and joint hidden state
            disfluency_input = torch.cat([
                token_embeddings,  # E(yi): (B, T, U, joint_space_size)
                asr_joint,         # si: (B, T, U, joint_space_size)
            ], dim=-1)  # (B, T, U, 2*joint_space_size)
            
            # Step 4: Predict disfluency: W[E(yi); si] + b
            disfluency_scores = self.disfluency_output_layer(disfluency_input)  # (B, T, U, disfluency_classes)
            
            outputs["disfluency_scores"] = disfluency_scores
            """
        
        return outputs
    
    def __call__(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        """Call method for BeamSearchTransducer compatibility.
        
        This method returns only ASR scores as a tensor (not dict) to maintain
        compatibility with the existing BeamSearchTransducer implementation.
        
        Args:
            encoder_out: Encoder output
            decoder_out: Decoder output
        
        Returns:
            asr_scores: ASR scores tensor (B, T, U, vocab_size) or (vocab_size,) for beam search
        """
        outputs = self.forward(encoder_out, decoder_out, return_disfluency=False)
        asr_scores = outputs["asr_scores"]
        
        # For beam search (1D inputs), return squeezed output
        if encoder_out.dim() == 1:
            # (1, 1, 1, vocab_size) -> (vocab_size,)
            return asr_scores.squeeze(0).squeeze(0).squeeze(0)
        else:
            return asr_scores


class MultitaskTransducerLoss(nn.Module):

    """Multitask loss function for Transducer with disfluency detection
    
    既存のRNNT lossにdisfluency lossを追加
    """
    
    def __init__(
        self,
        blank_id: int = 0,
        use_disfluency_detection: bool = True,
        disfluency_weight: float = 1.0,
        disfluency_smoothing: float = 0.1,
        reduction: str = "mean",
        trans_type: str = "warp_rnnt",  # or "torchaudio"
        use_viterbi_alignment: bool = False,  # NEW: Enable Viterbi alignment
    ):
        """Initialize multitask loss
        
        Args:
            blank_id: Blank token ID for transducer
            use_disfluency_detection: Whether to use disfluency detection
            disfluency_weight: Weight for disfluency loss
            disfluency_smoothing: Label smoothing for disfluency
            reduction: Loss reduction method
            trans_type: Transducer loss implementation type
            use_viterbi_alignment: Use Viterbi alignment for disfluency labels
        """
        super().__init__()
        self.blank_id = blank_id
        self.use_disfluency_detection = use_disfluency_detection
        self.disfluency_weight = disfluency_weight
        self.reduction = reduction
        self.use_viterbi_alignment = use_viterbi_alignment
        
        # Setup RNN-T loss
        if trans_type == "warp_rnnt":
            try:
                from warp_rnnt import RNNTLoss
                self.rnnt_loss = RNNTLoss(blank=blank_id, reduction=reduction)
                self.trans_type = "warp_rnnt"
            except ImportError:
                print("warp_rnnt not found, falling back to torchaudio")
                trans_type = "torchaudio"
        
        if trans_type == "torchaudio":
            try:
                from torchaudio.functional import rnnt_loss
                self.rnnt_loss = rnnt_loss
                self.trans_type = "torchaudio"
            except ImportError:
                raise ImportError(
                    "Please install warp-rnnt or torchaudio for RNN-T loss"
                )
        
        # Disfluency detection loss
        if self.use_disfluency_detection:
            self.disfluency_criterion = nn.CrossEntropyLoss(
                reduction=reduction,
                label_smoothing=disfluency_smoothing,
                ignore_index=-100,  # Ignore padding/blank positions
            )
    
    def forward(
        self,
        asr_scores: torch.Tensor,
        targets: torch.Tensor,
        encoder_lens: torch.Tensor,
        target_lens: torch.Tensor,
        disfluency_scores: Optional[torch.Tensor] = None,
        disfluency_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate multitask loss
        
        Args:
            asr_scores: ASR output from joint network (B, T, U, V)
                       This is used for BOTH ASR loss AND alignment for disfluency
            targets: ASR target sequences (B, L)
            encoder_lens: Encoder output lengths (B,)
            target_lens: Target sequence lengths (B,)
            disfluency_scores: Disfluency output (B, T, U, C)
            disfluency_targets: Disfluency labels (B, L) - character level
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Calculate RNN-T loss for ASR
        if self.trans_type == "warp_rnnt":
            asr_loss = self.rnnt_loss(
                asr_scores, targets, encoder_lens, target_lens
            )
        else:  # torchaudio
            # Convert to log probabilities
            asr_scores_log = F.log_softmax(asr_scores, dim=-1)
            asr_loss = self.rnnt_loss(
                asr_scores_log, targets, encoder_lens, target_lens,
                blank=self.blank_id, reduction=self.reduction
            )
        
        loss_dict = {"loss_asr": asr_loss.detach()}
        total_loss = asr_loss
        
        # Calculate disfluency detection loss if enabled
        if self.use_disfluency_detection and disfluency_scores is not None:
            B, T, U, C = disfluency_scores.shape
            
            # ===== Transducer Alignment-Aware Selection =====
            if self.use_viterbi_alignment and disfluency_targets is not None:
                # Check if disfluency_targets is at character level (B, L) where L != T
                if disfluency_targets.dim() == 2 and disfluency_targets.size(1) != T:
                    # PATH A: Viterbi alignment with character-level targets
                    print(f"[PATH A] Using Viterbi alignment: B={B}, T={T}, L={disfluency_targets.size(1)}", flush=True)
                    
                    # Use Transducer's own alignment (NOT CTC!)
                    # Extract alignment from asr_scores (joint network output)
                    frame_level_labels, frame_to_char_idx = align_disfluency_labels_with_transducer(
                        joint_logits=asr_scores,  # Use Transducer joint logits for alignment
                        encoder_out_lens=encoder_lens,
                        text=targets,
                        text_lengths=target_lens,
                        isdysfl=disfluency_targets,
                        blank_id=self.blank_id,
                    )
                    # frame_level_labels: (B, T) with -100 for blank/padding
                    # frame_to_char_idx: (B, T) with -1 for blank/padding
                    
                    # Select predictions at aligned character positions
                    # Create valid mask (non-blank, non-padding)
                    # CRITICAL: Must filter both frame_to_char_idx >= 0 AND frame_level_labels >= 0
                    valid_mask = (frame_to_char_idx >= 0) & (frame_level_labels >= 0) & (frame_level_labels < C)  # (B, T)
                    
                    if valid_mask.any():
                        # Prepare indices for gathering
                        batch_idx = torch.arange(B, device=disfluency_scores.device)[:, None].expand(B, T)[valid_mask]
                        frame_idx = torch.arange(T, device=disfluency_scores.device)[None, :].expand(B, T)[valid_mask]
                        char_idx = frame_to_char_idx[valid_mask]
                        
                        # Select corresponding predictions: (N_valid, C)
                        # disfluency_scores[b, t, u, :] where u = frame_to_char_idx[b, t]
                        selected_scores = disfluency_scores[batch_idx, frame_idx, char_idx, :]
                        selected_labels = frame_level_labels[valid_mask]
                        
                        # DEBUG: Check for invalid labels before loss calculation
                        import sys
                        if selected_labels.numel() > 0:
                            min_label = selected_labels.min().item()
                            max_label = selected_labels.max().item()
                            if min_label < 0 or max_label >= C:
                                print(f"\n{'='*60}", flush=True)
                                print(f"ERROR: Invalid labels detected!", flush=True)
                                print(f"  Label range: [{min_label}, {max_label}], Expected: [0, {C-1}]", flush=True)
                                print(f"  Num valid positions: {valid_mask.sum().item()}", flush=True)
                                print(f"  Num selected labels: {selected_labels.numel()}", flush=True)
                                print(f"  frame_to_char_idx range: [{frame_to_char_idx.min().item()}, {frame_to_char_idx.max().item()}]", flush=True)
                                print(f"  frame_level_labels unique: {frame_level_labels.unique().tolist()}", flush=True)
                                print(f"  selected_labels unique: {selected_labels.unique().tolist()}", flush=True)
                                
                                # Show specific invalid values
                                invalid_mask = (selected_labels < 0) | (selected_labels >= C)
                                if invalid_mask.any():
                                    invalid_labels = selected_labels[invalid_mask]
                                    print(f"  Invalid label values (first 20): {invalid_labels[:20].tolist()}", flush=True)
                                    print(f"  Count by value:", flush=True)
                                    for val in invalid_labels.unique():
                                        count = (invalid_labels == val).sum().item()
                                        print(f"    {val}: {count} occurrences", flush=True)
                                print(f"{'='*60}\n", flush=True)
                                sys.stdout.flush()
                                
                                # Emergency fix: clamp to valid range
                                selected_labels = torch.clamp(selected_labels, 0, C-1)
                        
                        # FORCE clamp ALWAYS to be absolutely sure
                        selected_labels = torch.clamp(selected_labels, 0, C-1)
                        
                        # Calculate loss only on valid (non-blank) positions
                        # CrossEntropyLoss with ignore_index=-100 will automatically skip -100 labels
                        disfluency_loss = self.disfluency_criterion(selected_scores, selected_labels)
                    else:
                        # No valid positions (shouldn't happen in practice)
                        disfluency_loss = torch.tensor(0.0, device=disfluency_scores.device, requires_grad=True)
                else:
                    # PATH B: Fall back to old method if dimensions don't match expectations
                    print(f"[PATH B] Fallback method 1: disfluency_targets.dim()={disfluency_targets.dim()}, size={disfluency_targets.shape}", flush=True)
                    # This handles edge cases where disfluency_targets is already frame-level
                    if disfluency_targets.dim() == 2:  # (B, T)
                        # Average pool over character dimension
                        frame_scores = disfluency_scores.mean(dim=2)  # (B, T, C)
                        disfluency_scores_flat = frame_scores.reshape(-1, C)
                        disfluency_targets_flat = disfluency_targets.reshape(-1)
                    else:
                        # (B, T, U) format
                        disfluency_scores_flat = disfluency_scores.reshape(-1, C)
                        disfluency_targets_flat = disfluency_targets.reshape(-1)
                    
                    # CRITICAL: Replace -1 padding with -100 (ignore_index)
                    disfluency_targets_flat = torch.where(
                        disfluency_targets_flat == -1,
                        torch.tensor(-100, dtype=disfluency_targets_flat.dtype, device=disfluency_targets_flat.device),
                        disfluency_targets_flat
                    )
                    # CRITICAL: Clamp all values to valid range [0, C-1] or -100
                    invalid_mask = (disfluency_targets_flat >= 0) & (disfluency_targets_flat >= C)
                    if invalid_mask.any():
                        print(f"WARNING [fallback path 1]: Found {invalid_mask.sum().item()} invalid labels, clamping...", flush=True)
                        disfluency_targets_flat = torch.clamp(disfluency_targets_flat, -100, C-1)
                    
                    disfluency_loss = self.disfluency_criterion(
                        disfluency_scores_flat, disfluency_targets_flat
                    )
            else:
                # PATH C: No Viterbi alignment: fall back to averaging over U dimension
                print(f"[PATH C] No Viterbi alignment: use_viterbi={self.use_viterbi_alignment}, targets is None={disfluency_targets is None}", flush=True)
                # Average pool over character dimension
                frame_scores = disfluency_scores.mean(dim=2)  # (B, T, C)
                
                # Prepare targets
                if disfluency_targets.dim() == 2 and disfluency_targets.size(1) == T:
                    # Already frame-level (B, T)
                    pass
                elif disfluency_targets.dim() == 2:
                    # Character-level but no Viterbi - pad/truncate to T
                    target_T = disfluency_targets.size(1)
                    if target_T < T:
                        pad_size = T - target_T
                        padding = torch.full((B, pad_size), -100, dtype=disfluency_targets.dtype, device=disfluency_targets.device)
                        disfluency_targets = torch.cat([disfluency_targets, padding], dim=1)
                    else:
                        disfluency_targets = disfluency_targets[:, :T]
                
                disfluency_scores_flat = frame_scores.reshape(-1, C)
                disfluency_targets_flat = disfluency_targets.reshape(-1)
                
                # CRITICAL: Replace -1 padding with -100 (ignore_index)
                disfluency_targets_flat = torch.where(
                    disfluency_targets_flat == -1,
                    torch.tensor(-100, dtype=disfluency_targets_flat.dtype, device=disfluency_targets_flat.device),
                    disfluency_targets_flat
                )
                # CRITICAL: Clamp all values to valid range [0, C-1] or -100
                invalid_mask = (disfluency_targets_flat >= 0) & (disfluency_targets_flat >= C)
                if invalid_mask.any():
                    print(f"WARNING [fallback path 2]: Found {invalid_mask.sum().item()} invalid labels, clamping...", flush=True)
                    disfluency_targets_flat = torch.clamp(disfluency_targets_flat, -100, C-1)
                
                disfluency_loss = self.disfluency_criterion(
                    disfluency_scores_flat, disfluency_targets_flat
                )
            # ===== End Transducer Alignment-Aware Selection =====
            
            loss_dict["disfluency_loss"] = disfluency_loss.detach()
            total_loss = total_loss + self.disfluency_weight * disfluency_loss
        
        loss_dict["loss"] = total_loss.detach()
        
        return total_loss, loss_dict
    