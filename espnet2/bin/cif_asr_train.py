#!/usr/bin/env python3
"""CIF タグ検出付き ASR モデルの学習スクリプト。"""

from espnet2.tasks.asr import CIFASRTask


def get_parser():
    return CIFASRTask.get_parser()


def main(cmd=None):
    """CIF タグ検出付き ASR モデルの学習。

    Example:
        % python cif_asr_train.py --print_config --optim adam > conf/train_cif.yaml
        % python cif_asr_train.py --config conf/train_cif.yaml
    """
    CIFASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
