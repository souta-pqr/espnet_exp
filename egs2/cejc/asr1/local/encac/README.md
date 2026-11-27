# encac：音響だけの判定器（提案手法）と Easy Turn の検証

手順は [05 提案手法](../../docs/experiments/05_proposed/README.md) と [06 比較手法](../../docs/experiments/06_baselines/README.md)．

silver13 の `~/2026/work/easy_turn/cejc/` で使ったもの（2026-11）．11/26 以降は複製の `~/2026/work/encac_pool/` で使ったもの．

## 提案手法の学習

- `enc_plug.py`：エンコーダ＋フレームの平均＋線形層．エンコーダは `ENC=espnet:...`（CEJC の音声認識で学習した contextual block conformer），`whisper:`・`w2v:`・`mimi:` を差し替えられる．環境変数 `POOL` で，全フレームの平均の代わりに最後の `POOL` フレームの平均で判定する（11/26．効かなかった）．
- `train_ac.py`：学習．`--train train_gxac` で前の区間とつないだ音声も足す．環境変数 `CTCW` が 0 より大きいと，音声認識を補助タスクとして CTC 損失を足す（マルチタスク学習）．`PRE` で読み出し位置を区間末より前にも広げる．`CTCW` のとき補助タスクの出力層も `best_aux.pt` に残す．`--ctc_only` で 3 クラスの損失を使わず音声認識（CTC）だけで学習する．500 回の更新ごとに `last.pt` を残し，同じ命令で途中から再開する（`--ckpt_every`）．
- `train_et_data.py`：学習データの読み込み（`train_et.py` と共通）．環境変数 `LAST` で入力を区間の終わりの `LAST` 秒に縮める（11/20．効かなかった）．
- 提案手法（espmk1gs）：`MAXSEC=12 CTCW=1.0 python train_ac.py <出力> --lr 5e-4 --train train_gs --bs 2 --acc 32`（`run_gs10.sh`．11/19 から．それまでは espmk1：`--train train_gxac --bs 8 --acc 8`）
- `run_gxacgs9.sh`：train_gxac と train_gs を合わせて学習する版（効かなかった）．
- `export_encac.py`：学習した重みを，逐次処理のプログラム（exp-realtime-epd）が読む形に書き出す．
- `run_enc*.sh`・`run_dump*.sh`：エンコーダを取り替える比較の学習とスコアの書き出し．

## 音声認識の性能（CER）と判定の時刻（11/26〜）

- `cer_aux.py`：補助タスクの出力層（CTC）で評価データを認識し，CER を測る．`cer_breakdown.py`：置換・脱落・挿入，区間の長さ別の内訳．
- `kobori_chain31.sh`・`kobori_chain32.sh`：CER を測るジョブと，エンコーダを固定して出力層だけ学習する基準．
- `run_ctc9.sh`（silver9）：音声認識だけでエンコーダを学習する（`--ctc_only`）．`run_r10.sh`（silver10）：提案手法を出力層も残す形で学び直す．
- `acc_by_off.py`・`kobori_accoff.sh`：区間末から何秒後に判定するかごとの正解率．

## Easy Turn の検証

- `et_model.py`：Easy Turn（ASLP-lab，Apache-2.0）を読み込み，状態のタグの確率を出す．重みと Easy Turn のコードは含まない．
- `train_et.py`：CEJC での fine-tuning．`--no_text` で書き起こしを出さない版．
- `ac_model.py`・`run_et_ac.sh`：音響だけ（エンコーダ＋平均＋線形層）のアブレーション．
- `et_dump.py`：スコアの書き出し．`et_speed.py`・`try_et.py`：動作と速さの確認．

## データ

- `make_train_gxac.py`：前の区間とつないだ学習データ `data/train_gxac` を作る．
- `make_train_gs.py`：提案手法の学習データ `data/train_gs`（silero-vad で決めた音声区間で切ったもの）を作る．
