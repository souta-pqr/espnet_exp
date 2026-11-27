"""train_et.py のデータの読み込み（train_ac.py から使うために同じものを置く．2026-11-03）"""
import os, re, wave, random
import numpy as np, torch
import et_model as E

ASR1 = os.environ.get("ASR1", "/home/kobori/2025/espnet/egs2/cejc/asr1")   # silver12 では ~/2026/espnet_tt/…
SR = 16000
POST = float(os.environ.get("POST", 2.0))   # 区間末の後，何秒までを読むか（11/15：環境変数で変えられるようにした．既定は 2 秒のまま）
MAXSEC = float(os.environ.get("MAXSEC", 0))
LAST = float(os.environ.get("LAST", 0))     # 11/20：0 より大きいと，入力を区間の終わりの LAST 秒＋区間末の後に縮める（長い入力で言い終わりが薄まるかを確かめる）
CLIP = os.environ.get("CLIP") == "1"       # 11/15：区間末の後の読み出しを，同じ録音の次の区間（声）が始まる前までにする（実時間ではスコアを無音の間だけ出すため）
HARDW = float(os.environ.get("HARDW", 0))   # 11/16：0 より大きいと，区間末から同じ話者の次の声までの間（秒）も返す（train_ac.py が，間が 1 秒以上ある「まだ続く」区間の損失を HARDW 倍にする）
CTCW = float(os.environ.get("CTCW", 0))   # 11/16：0 より大きいと，区間の長さ（秒．入力の頭が切られて文字と合わないときは -1）も返す（train_ac.py が文字を当てる追加の課題に使う）
NEXT = os.environ.get("NEXT") == "1"     # 11/15：読み出し位置から同じ話者の次の声までの時間（秒．最大 2 秒）も返す（train_ac.py が追加の課題に使う）
PRE = float(os.environ.get("PRE", 0))   # 11/12：学習の読み出し位置を区間末の PRE 秒前から選ぶ（アブレーション．既定は区間末の後だけ）
FEAT = E.logmel                  # 11/04：train_ac.py が ac_model.feat に置き換える（エンコーダの取り替え）
TAG = {"0": "<incomplete>", "1": "<complete>", "2": "<backchannel>"}


def norm(t):
    t = re.sub(r"<(mask|unclear)>", "＊", t)
    t = re.sub(r"\([A-Z?]+ ?", "", t).replace(")", "")
    return t.replace(" ", "")


def load(d):
    wav = dict(l.split() for l in open(f"{ASR1}/data/{d}/wav.scp"))
    tag = dict(l.split() for l in open(f"{ASR1}/data/{d}/tag"))
    txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(f"{ASR1}/data/{d}/text"))
    return [(wav[rec], float(b), float(e), norm(txt[u]), TAG[tag[u]]) for u, rec, b, e in (l.split() for l in open(f"{ASR1}/data/{d}/segments"))]


class DS(torch.utils.data.Dataset):
    def __init__(self, items, fixed):
        self.items = items
        self.off = np.random.default_rng(0).uniform(0, POST, len(items)) if fixed else None
        self.gap = None
        if CLIP or NEXT or HARDW:                           # 各区間の末から，同じ録音の次の区間の頭までの長さ（秒）
            bs = {}
            for path, b, e, _, _ in items:
                bs.setdefault(path, set()).add(b)
            bs = {k: np.array(sorted(v)) for k, v in bs.items()}
            self.gap = np.empty(len(items))
            for i, (path, b, e, _, _) in enumerate(items):
                j = np.searchsorted(bs[path], e - 1e-3)
                self.gap[i] = max(0.0, bs[path][j] - e) if j < len(bs[path]) else np.inf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, b, e, txt, tag = self.items[i]
        off = self.off[i] if self.off is not None else np.random.uniform(-PRE, POST)
        if CLIP:
            off = min(off, self.gap[i])
        w = wave.open(path)
        s0 = int(round(b * SR)); s1 = min(int(round((e + off) * SR)), w.getnframes())
        if MAXSEC:                                   # 11/11：つないだ音声は長いので，最後の MAXSEC 秒だけ入れる（元の区間は 10 秒＋2 秒まで）
            s0 = max(s0, s1 - int(MAXSEC * SR))
        if LAST:                                     # 11/20：区間の終わりの LAST 秒＋その後だけ入れる（実時間は ctx="p0:LAST"）
            s0 = max(s0, int(round((e - LAST) * SR)))
        w.setpos(s0)
        x = np.frombuffer(w.readframes(max(s1 - s0, 400)), dtype=np.int16).astype(np.float32) / 32768
        prompt = random.choice(E.PROMPTS[E.TASK]) if self.off is None else E.PROMPT
        if CTCW:
            return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt, float(e - b) if s0 == int(round(b * SR)) else -1.0
        if HARDW:
            return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt, float(self.gap[i])
        if NEXT:
            return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt, float(np.clip(self.gap[i] - off, 0.0, 2.0))
        return FEAT(torch.from_numpy(x.copy())), txt, tag, prompt


def collate(b):
    return list(zip(*b))
