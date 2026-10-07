import sys, re, collections
sys.path.insert(0, 'section-detection/exp17')
from make_text_kanji import parse_chronological, make_utt_id
blocks = parse_chronological('section-detection/exp17/chronological')
ur = re.compile(r'^開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)(?:\s+<\w+>)?\s*$')
key2conv = collections.defaultdict(set)
for cid, utts in blocks:
    for u in utts:
        s, e, spk, _ = ur.match(u).groups()
        key2conv[(spk, s, e)].add(cid)
LR = re.compile(r'^\[LINE (\d+)\] 会話=(\d+) 発話位置=(\d+) 話者=(\S+)')
UR = re.compile(r'^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$')
items = {}; cur = None; sec = None
for raw in open('/autofs/diamond5/share/users/kobori/result_oneshot.txt', encoding='utf-8'):
    if raw.startswith('[LINE'):
        m = LR.match(raw); cur = [int(m[1]), int(m[2]), None]; sec = None; continue
    if raw.startswith('[TARGET]'): sec = 't'; continue
    if raw.startswith('[POST_HISTORY]') or raw.startswith('[HISTORY]'): sec = None; continue
    if sec == 't' and raw.startswith('- '):
        m = UR.match(raw.rstrip('\n')); cur[2] = m.groups(); sec = None; continue
    if raw.startswith('[RESULT]') and cur and cur[2]:
        items[cur[0]] = (cur[1], cur[2], int(raw.split()[1]))
print('items', len(items))
# 会話番号 → 会話 ID（多数決）
vote = collections.defaultdict(collections.Counter)
for ln, (c, (s, e, spk, t), r) in items.items():
    for cid in key2conv.get((spk, s, e), ()): vote[c][cid] += 1
c2id = {c: v.most_common(1)[0][0] for c, v in vote.items()}
print('会話 対応', len(c2id), '/', len({v[0] for v in items.values()}))
with open('tt_analysis/qwen3_fc_os_uttid.tsv', 'w') as f:
    f.write('line_no\tutt_id\tresult\n')
    for ln, (c, (s, e, spk, t), r) in sorted(items.items()):
        if c in c2id:
            f.write(f'{ln}\t{make_utt_id(c2id[c], spk, float(s), float(e))}\t{r}\n')
