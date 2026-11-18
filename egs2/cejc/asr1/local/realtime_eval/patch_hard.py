import os
d = os.path.expanduser("~/2026/work/easy_turn/cejc/")
def sub(f, a, b):
    s = open(d + f).read(); assert s.count(a) == 1, (f, a); open(d + f, "w").write(s.replace(a, b))
f = "train_et_data.py"
sub(f, 'NEXT = os.environ.get("NEXT") == "1"', 'HARDW = float(os.environ.get("HARDW", 0))   # 11/16：0 より大きいと，区間末から同じ話者の次の声までの間（秒）も返す（train_ac.py が，間が 1 秒以上ある「まだ続く」区間の損失を HARDW 倍にする）\nNEXT = os.environ.get("NEXT") == "1"')
sub(f, "        if CLIP or NEXT:  ", "        if CLIP or NEXT or HARDW:")
sub(f, "        if NEXT:\n            return", "        if HARDW:\n            return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt, float(self.gap[i])\n        if NEXT:\n            return")
f = "train_ac.py"
sub(f, "    l = torch.nn.functional.cross_entropy(lg, y)\n    if rem is not None:", "    l = torch.nn.functional.cross_entropy(lg, y)\n    if rem is not None and HARDW:    # 間が 1 秒以上ある「まだ続く」区間を重くする（実時間で割り込みになるのはこの区間）\n        w = torch.where((y == 0) & (torch.tensor(rem, device=\"cuda\") >= 1.0), HARDW, 1.0)\n        l = (torch.nn.functional.cross_entropy(lg, y, reduction=\"none\") * w).sum() / w.sum()\n    elif rem is not None:")
sub(f, "NEXT = train_et_data.NEXT ", "HARDW = train_et_data.HARDW\nNEXT = train_et_data.NEXT ")
