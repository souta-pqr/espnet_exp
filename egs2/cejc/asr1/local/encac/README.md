# encac：音響だけの判定器（提案手法）と Easy Turn の検証

silver13 の `~/2026/work/easy_turn/cejc/` で使ったもの（2026-11）．

## 提案手法の学習

- `enc_plug.py`：エンコーダ＋フレームの平均＋線形層．エンコーダは `ENC=espnet:...`（CEJC の音声認識で学習した contextual block conformer），`whisper:`・`w2v:`・`mimi:` を差し替えられる．
- `train_ac.py`：学習．`--train train_gxac` で前の区間とつないだ音声も足す．環境変数 `CTCW` が 0 より大きいと，音声認識を補助タスクとして CTC 損失を足す（マルチタスク学習）．`PRE` で読み出し位置を区間末より前にも広げる．
- `train_et_data.py`：学習データの読み込み（`train_et.py` と共通）．
- 提案手法（espmk1）：`MAXSEC=12 CTCW=1.0 python train_ac.py <出力> --lr 5e-4 --train train_gxac --bs 8 --acc 8`
- `run_enc*.sh`・`run_dump*.sh`：エンコーダを取り替える比較の学習とスコアの書き出し．

## Easy Turn の検証

- `et_model.py`：Easy Turn（ASLP-lab，Apache-2.0）を読み込み，状態のタグの確率を出す．重みと Easy Turn のコードは含まない．
- `train_et.py`：CEJC での fine-tuning．`--no_text` で書き起こしを出さない版．
- `ac_model.py`・`run_et_ac.sh`：音響だけ（エンコーダ＋平均＋線形層）のアブレーション．
- `et_dump.py`：スコアの書き出し．`et_speed.py`・`try_et.py`：動作と速さの確認．
