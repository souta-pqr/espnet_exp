"""train_ac.py・train_et_data.py に，文字を当てる追加の課題（CTC）を足す（2026-11-16）．環境変数 CTCW（重み）が 0 より大きいときだけ働く．"""
import shutil
def rep(path, pairs):
    s = open(path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a); s = s.replace(a, b)
    open(path, "w").write(s)

for f in ("train_ac.py", "train_et_data.py"):
    shutil.copy(f, f + ".bak_20261116b")
rep("train_et_data.py", [
    ('NEXT = os.environ.get("NEXT") == "1" ',
     'CTCW = float(os.environ.get("CTCW", 0))   # 11/16：0 より大きいと，区間の長さ（秒．入力の頭が切られて文字と合わないときは -1）も返す（train_ac.py が文字を当てる追加の課題に使う）\nNEXT = os.environ.get("NEXT") == "1" '),
    ('        if HARDW:\n',
     '        if CTCW:\n            return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt, float(e - b) if s0 == int(round(b * SR)) else -1.0\n        if HARDW:\n'),
])
rep("train_ac.py", [
    ('def loss_batch(feats, tags, rem=None):\n',
     '''CTCW = train_et_data.CTCW        # 11/16：追加の課題（区間の文字を CTC で当てる．文字の手がかりを contextual block conformer エンコーダに残す）．推論では使わないので重みには残さない
if CTCW:
    import yaml, enc_plug
    TOK = {t: i for i, t in enumerate(yaml.safe_load(open(enc_plug.GTCONF))["token_list"])}
    aux = torch.nn.Linear(m.head.in_features, len(TOK)).cuda(); enc_out = {}
    m.encoder.register_forward_hook(lambda mod, i, o: enc_out.__setitem__("x", o))


def ctc_loss(txts, segs):
    """区間の終わりまでのフレームで，その区間の文字を当てる（区間の後の無音や次の声は入れない）"""
    h, hl = enc_out["x"][0], enc_out["x"][1]
    use = [i for i, (t, s) in enumerate(zip(txts, segs)) if s > 0 and 0 < len(t) <= int(s * 16000 / 528)]
    if not use:
        return h.sum() * 0.0
    n = torch.tensor([min(int(hl[i]), int(segs[i] * 16000 / 528) + 1) for i in use])
    y = [torch.tensor([TOK.get(c, 1) for c in txts[i]]) for i in use]
    lp = aux(h[use]).log_softmax(-1).transpose(0, 1)
    return torch.nn.functional.ctc_loss(lp, torch.cat(y), n, torch.tensor([len(v) for v in y]), blank=0, zero_infinity=True)


def loss_batch(feats, tags, rem=None, txts=None):
'''),
    ('    elif rem is not None:\n',
     '    elif rem is not None and CTCW:\n        l = l + CTCW * ctc_loss(txts, rem)\n    elif rem is not None:\n'),
    ('(list(aux.parameters()) if NEXT else [])', '(list(aux.parameters()) if NEXT or CTCW else [])'),
    ('    for k, (feats, _, tags, _, *rem) in enumerate(dl):\n        loss, _ = loss_batch(feats, tags, *rem)',
     '    for k, (feats, txts, tags, _, *rem) in enumerate(dl):\n        loss, _ = loss_batch(feats, tags, *rem, txts=txts)'),
])
print("ok")
