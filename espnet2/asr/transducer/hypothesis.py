"""Hypothesis class for Transducer beam search."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TransducerHypothesis:
    """Hypothesis for Transducer beam search.
    
    This class represents a hypothesis during Transducer beam search.
    It tracks the label sequence, decoder state, current time step,
    and optionally disfluency information.
    """
    
    yseq: List[int]  # Label sequence (token IDs)
    score: float  # Cumulative log probability
    dec_state: Any  # Decoder state (e.g., LSTM (h, c))
    t: int  # Current encoder time step
    
    # Disfluency-specific fields (optional)
    dysfl_seq: List[int] = field(default_factory=list)  # Disfluency label sequence
    dysfl_scores: List[float] = field(default_factory=list)  # Disfluency confidence scores
    
    # Language model state (optional)
    lm_state: Optional[Any] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not isinstance(self.yseq, list):
            raise TypeError(f"yseq must be a list, got {type(self.yseq)}")
        if not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be numeric, got {type(self.score)}")
        if not isinstance(self.t, int):
            raise TypeError(f"t must be an integer, got {type(self.t)}")
    
    def __len__(self) -> int:
        """Return the length of the label sequence."""
        return len(self.yseq)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TransducerHypothesis("
            f"yseq={self.yseq}, "
            f"score={self.score:.3f}, "
            f"t={self.t}, "
            f"len={len(self.yseq)})"
        )
    
    def copy(self) -> "TransducerHypothesis":
        """Create a shallow copy of the hypothesis.
        
        Note: Lists are copied, but decoder state is shared.
        For decoder state updates, use the returned new state from decoder.
        """
        return TransducerHypothesis(
            yseq=self.yseq.copy(),
            score=self.score,
            dec_state=self.dec_state,  # Shared reference
            t=self.t,
            dysfl_seq=self.dysfl_seq.copy(),
            dysfl_scores=self.dysfl_scores.copy(),
            lm_state=self.lm_state,
        )
    
    def normalized_score(self, length_penalty: float = 0.0) -> float:
        """Compute length-normalized score.
        
        Args:
            length_penalty: Length penalty coefficient (0 = no penalty)
        
        Returns:
            Normalized score
        """
        length = len(self.yseq)
        if length == 0:
            return self.score
        
        if length_penalty == 0.0:
            return self.score / length
        else:
            # Google NMT style: ((5 + len) / 6) ^ penalty
            return self.score / ((5 + length) / 6.0) ** length_penalty
