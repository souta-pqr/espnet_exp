import os
d = os.path.expanduser("~/2026/work/easy_turn/cejc/")
def sub(f, a, b):
    s = open(d + f).read(); assert s.count(a) == 1, (f, a); open(d + f, "w").write(s.replace(a, b))
f = "train_et_data.py"
sub(f, 'PRE = float(os.environ.get("PRE", 0))', 'NEXT = os.environ.get("NEXT") == "1"     # 11/15：読み出し位置から同じ話者の次の声までの時間（秒．最大 2 秒）も返す（train_ac.py が追加の課題に使う）\nPRE = float(os.environ.get("PRE", 0))')
sub(f, "        if CLIP:                                     #", "        if CLIP or NEXT:                             #")
sub(f, "        if self.gap is not None:\n            off = min(off, self.gap[i])", "        if CLIP:\n            off = min(off, self.gap[i])")
sub(f, "        return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt", "        if NEXT:\n            return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt, float(np.clip(self.gap[i] - off, 0.0, 2.0))\n        return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt")
f = "train_ac.py"
sub(f, "def loss_batch(feats, tags):\n    y = torch.tensor([CLS[t] for t in tags], device=\"cuda\")\n    lg = m(feats)\n    return torch.nn.functional.cross_entropy(lg, y), (lg.argmax(1) == y).sum().item()",
'''NEXT = train_et_data.NEXT        # 11/15：追加の課題（同じ話者の次の声までの時間を当てる．Next-Turn, arXiv 2606.18094 にならう）．推論では使わないので重みには残さない
if NEXT:
    aux = torch.nn.Linear(1024, 1).cuda(); pooled = {}
    m.head.register_forward_hook(lambda mod, i, o: pooled.__setitem__("x", i[0]))


def loss_batch(feats, tags, rem=None):
    y = torch.tensor([CLS[t] for t in tags], device="cuda")
    lg = m(feats)
    l = torch.nn.functional.cross_entropy(lg, y)
    if rem is not None:
        l = l + torch.nn.functional.mse_loss(2.0 * torch.sigmoid(aux(pooled["x"]).squeeze(1)), torch.tensor(rem, device="cuda", dtype=torch.float32))
    return l, (lg.argmax(1) == y).sum().item()''')
sub(f, "    for feats, _, tags, _ in dl:\n        l, k = loss_batch(feats, tags); ls", "    for feats, _, tags, *_ in dl:\n        l, k = loss_batch(feats, tags); ls")
sub(f, '{"params": m.head.parameters(), "lr": 1e-3}]', '{"params": list(m.head.parameters()) + (list(aux.parameters()) if NEXT else []), "lr": 1e-3}]')
sub(f, "    for k, (feats, _, tags, _) in enumerate(dl):\n        loss, _ = loss_batch(feats, tags)", "    for k, (feats, _, tags, _, *rem) in enumerate(dl):\n        loss, _ = loss_batch(feats, tags, *rem)")
