#!/usr/bin/env python3
"""Disfluency ASR training."""

from espnet2.tasks.asr import DisfluencyASRTask


def get_parser():
    """Get argument parser for disfluency ASR training."""
    parser = DisfluencyASRTask.get_parser()
    return parser


def main(cmd=None):
    """Train disfluency ASR model.

    Example:
        % python disfluency_asr_train.py --print_config --optim adam \
                > conf/train_disfluency_asr.yaml
        % python disfluency_asr_train.py --config conf/train_disfluency_asr.yaml
    """
    DisfluencyASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()