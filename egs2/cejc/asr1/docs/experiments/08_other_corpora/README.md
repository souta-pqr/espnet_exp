# 08 別の対話データ

CEJC の正解は LLM が付けたラベルなので，人が付けたラベルや実際の会話の流れで確かめる．
どちらのデータも学習にも，応答の決め方を選ぶのにも使わない．決め方は CEJC の開発データで選んだものをそのまま使う．

[← 全体の README](../../../../../../README.md)

スクリプトは `local/realtime_eval/`．採点は `rt_rule.py` と同じ場所で動かす．データは研究室の外に出さない（出力は `/autofs/diamond5/share/users/kobori/` の下）．

## waseda_soma（今の報告で使う）

客の役とオペレータの役の 2 人（どちらも人）が，電話でレストランを探す対話．人がターンと対話行為の注釈を付けている．

### 1. CEJC と同じ形にする

```bash
python soma_prep.py          # → waseda_soma/data/{segments,tag,wav.scp}・wav/（話者ごとの音声）
python soma_cejc_tag.py      # → waseda_soma/data_cejc/（CEJC の決め方では完了にならない区間を数えない tag）
```

- `soma_prep.py`：区間は注釈の音声区間．相槌＝対話行為が相槌（bc）か了解（ack）だけ，継続＝同じ話者の次の区間が前のターンの続き，完了＝次の区間がターンの始まりか，後に区間が無い．相手の声の漏れ込みが大きいので，相手が話している区間は無音にする．
- `soma_cejc_tag.py`：完了のうち，0.5 秒未満の応答だけ（583）・言い終わりの合図だけ（570）・フィラーだけ（832）の区間を除く．CEJC ではこれらを相槌か継続にしているため．

### 2. スコアを書き出して採点する

```bash
bash kobori_chain21.sh espmk1          # 話者ごとの録音 180 本 → dump_soma_espmk1/
python rt_soma.py /autofs/diamond5/share/users/kobori/waseda_soma/data_cejc dump_soma_espmk1 \
    上限8=0.4,0.84,2.0,0.4,3.0,0.1 上限6=0.5,0.92,2.0,0.4,3.0,0.1 sil=1.0 sil=1.5 sil=2.0
```

- `名前=g1,th1,g2,th2,S,phi` で決め方を並べる．`sil=1.0` は「無音が 1.0 秒続いたら応答」（silero-vad のみ）．
- silero-vad の確率は書き出しの中のものを使う．
- 応答しなかった区間の内訳は `rt_miss_soma.py <data> <silero> <書き出し> <決め方> [出力.tsv]`．

## Hazumi（参考）

人が WoZ のエージェントに話す雑談．正解は LLM のラベルではなく，実際の会話の流れ（操作者が次の応答を出した時刻まで）．

```bash
python hazumi_extract.py 1911 1712 1902    # 動画からユーザの音声だけを取り出す（システムの発話の時間は無音にする）
bash kobori_chain13.sh espmk1              # スコアの書き出し → dump_hazumi1911_espmk1/
python rt_hazumi.py /autofs/diamond5/share/users/kobori/hazumi_audio/Hazumi1911_exchanges.tsv dump_hazumi1911_espmk1 \
    提案=0.4,0.84,2.0,0.4,3.0,0.1 sil=1.0
python rt_hazumi_pause.py ...              # ユーザの発話の途中の間ごとに，その間で応答してしまうか
```
