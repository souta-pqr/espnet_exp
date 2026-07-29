"""Transducer-specific modules."""

from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.hypothesis import TransducerHypothesis

__all__ = ["BeamSearchTransducer", "TransducerHypothesis"]
