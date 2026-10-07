import sys, collections
import numpy as np
sys.path.insert(0, 'local/turntaking')
from frame_fusion import prepare
from frame_rules import utt_lengths
from nbest_fusion import mix_frames
D = 'exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave'
grid = np.arange(0.05, 3.0 + 1e-9, 0.05)
G = {l.split('\t')[0]: l.split('\t')[1] for l in list(open('tt_analysis/gold1000_uttid.tsv'))[1:]}
rows = []
for dset, txt in [('eval', f'{D}/eval/text'), ('train_dev', f'{D}/train_dev_dec/text')]:
    P, Pt, lab, utts = prepare('frame_attn-frame2-blk42_same_n5', 'valid.loss.ave', dset, 'exp/text_bert_asr', txt, grid)
    lens = utt_lengths(utts, dset)
    cap = np.clip(np.searchsorted(grid, lens + 0.25, side='right') - 1, 0, len(grid) - 1)
    ix = np.arange(len(utts))
    pa = P[cap, ix].argmax(-1); pf = mix_frames(P, Pt, grid, lens, 0.55)[cap, ix].argmax(-1)
    for i, u in enumerate(utts):
        rows.append((u, dset, str(lab[i]), str(pa[i]), str(pf[i]), G.get(u)))
# 継続/完了の区間（LLM ラベル 0/1）全体で、モデル vs LLM ラベル
cc = [r for r in rows if r[2] in '01']
for k, name in [(3, '音声のみ'), (4, '融合')]:
    print(f'{name}: LLM ラベルとの一致（継続/完了 {len(cc)} 区間）= {np.mean([r[2] == r[k] for r in cc]):.3f}')
g = [r for r in cc if r[5] is not None]
print(f'\n人手正解のある区間 n={len(g)}', collections.Counter(r[1] for r in g))
for k, name in [(2, 'LLM ラベル'), (3, '音声のみ'), (4, '融合')]:
    print(f'  {name:8s} 人手正解との一致 {np.mean([r[5] == r[k] for r in g]):.3f}')
print('  音声のみ: LLM ラベルが正しい区間での一致', np.mean([r[3] == r[5] for r in g if r[2] == r[5]]).round(3), sum(r[2] == r[5] for r in g),
      '/ LLM ラベルが誤りの区間で人手に一致', np.mean([r[3] == r[5] for r in g if r[2] != r[5]]).round(3), sum(r[2] != r[5] for r in g))
print('  融合    : 同上', np.mean([r[4] == r[5] for r in g if r[2] == r[5]]).round(3), np.mean([r[4] == r[5] for r in g if r[2] != r[5]]).round(3))

# ---- train の人手正解区間（学習に使われた＝LLM ラベル寄りに偏る）----
from frame_rules import load_frame
from frame_fusion import _text_probs_maybe_ens
P, lab, utts = load_frame('tt_preds/frame_attn-frame2-blk42_same_n5_goldtrain_valid.loss.ave.tsv', grid)
lens = utt_lengths(utts, 'train_nodup')
cap = np.clip(np.searchsorted(grid, lens + 0.25, side='right') - 1, 0, len(grid) - 1)
ix = np.arange(len(utts)); pa = P[cap, ix].argmax(-1)
T = _text_probs_maybe_ens('exp/text_bert_asr', 'train_nodup', f'{D}/train_nodup_dec/text')
Pt = np.array([T[u] for u in utts])
pf = mix_frames(P, Pt, grid, lens, 0.55)[cap, ix].argmax(-1)
tr = [(u, 'train', str(lab[i]), str(pa[i]), str(pf[i]), G[u]) for i, u in enumerate(utts) if str(lab[i]) in '01']
print(f'\ntrain の人手正解区間 n={len(tr)}')
for k, name in [(2, 'LLM ラベル'), (3, '音声のみ'), (4, '融合')]:
    print(f'  {name:8s} 人手正解との一致 {np.mean([r[5] == r[k] for r in tr]):.3f}')
for name, k in [('音声のみ', 3), ('融合', 4)]:
    ok = [r for r in tr if r[2] == r[5]]; ng = [r for r in tr if r[2] != r[5]]
    print(f'  {name}: LLM 正の区間で人手一致 {np.mean([r[k] == r[5] for r in ok]):.3f} (n={len(ok)}) / LLM 誤の区間で人手一致 {np.mean([r[k] == r[5] for r in ng]):.3f} (n={len(ng)})')
import pickle; pickle.dump(g + tr, open('tt_analysis/gold1000_rows.pkl', 'wb'))
