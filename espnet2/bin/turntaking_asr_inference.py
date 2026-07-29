#!/usr/bin/env python3
"""turn-taking マルチタスク ASR モデルの推論（デコード）スクリプト。

ASR デコードは標準の asr_inference と同一でよい（turntaking モデルは ASRTask の
model_choices に登録済みで、ASRTask.build_model_from_file で transducer ASR として
組める。発話区間末ヘッド/ctx は ASR デコードでは未使用）。
asr.sh stage 12 が `espnet2.bin.turntaking_asr_inference` を呼ぶための薄いラッパー。
"""
from espnet2.bin.asr_inference import get_parser, main  # noqa: F401

if __name__ == "__main__":
    main()
