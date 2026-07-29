#!/usr/bin/env python3
"""turn-taking（発話区間末予測）マルチタスク付き ASR モデルの学習スクリプト。"""

from espnet2.tasks.asr import TurnTakingASRTask


def get_parser():
    return TurnTakingASRTask.get_parser()


def main(cmd=None):
    """turn-taking マルチタスク ASR の学習。

    Example:
        % python turntaking_asr_train.py --print_config --optim adam \
                > conf/train_turntaking.yaml
        % python turntaking_asr_train.py --config conf/train_turntaking.yaml
    """
    TurnTakingASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
