"""Easy Turn のアブレーション 2：音響だけ（2026-11-03，silver13）。
Easy Turn の音声エンコーダ（Whisper medium 相当，公開の重みから）だけを使い，フレームの平均を線形層で 3 クラス
（0=継続・1=完了・2=相槌．data/*/tag と同じ番号）に分ける。論文の「Whisper + Linear」に当たる。
"""
import importlib.util, os, sys, types, yaml, torch
import et_model as E
# エンコーダだけを使うので，wenet/__init__.py（torchaudio などを読む）を通さずに読み，whisper が無ければ
# wenet/utils/common.py が読む whisper.tokenizer.LANGUAGES だけを空で置く（silver12 の espnet の環境で動かすため．11/03）
if "wenet" not in sys.modules:
    _p = types.ModuleType("wenet"); _p.__path__ = [os.path.join(E.ET, "wenet")]; sys.modules["wenet"] = _p
if importlib.util.find_spec("whisper") is None:
    _w, _t = types.ModuleType("whisper"), types.ModuleType("whisper.tokenizer")
    _t.LANGUAGES = {}; _w.tokenizer = _t
    sys.modules["whisper"], sys.modules["whisper.tokenizer"] = _w, _t
from wenet.transformer.encoder import TransformerEncoder


class AC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = yaml.safe_load(open(E.CONF))
        self.encoder = TransformerEncoder(80, **cfg["encoder_conf"])
        self.head = torch.nn.Linear(1024, 3)

    def forward(self, feats):
        dev = self.head.weight.device
        lens = torch.tensor([len(f) for f in feats], device=dev)
        x = torch.nn.utils.rnn.pad_sequence(list(feats), batch_first=True).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, mask = self.encoder(x, lens)
        k = mask.squeeze(1).float()
        return self.head((h.float() * k[..., None]).sum(1) / k.sum(1, keepdim=True))


def load_pretrained(ckpt=os.environ.get("ET_ENC", "/home/kobori/2026/work/easy_turn/ckpt/checkpoint.pt")):
    """公開の重みのエンコーダの部分を読む．ET_ENC にエンコーダだけを抜き出したファイル（silver12 用）も渡せる"""
    m = AC()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    if any(k.startswith("encoder.") for k in sd):
        sd = {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}
    m.encoder.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return m


def load(path, device="cuda"):
    m = AC()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def probs(m, feats):
    """(B, 4)：et_model.tag_probs と同じ並び（<complete>・<incomplete>・<backchannel>・<wait>）"""
    p = m(feats).softmax(-1).cpu()
    return torch.stack([p[:, 1], p[:, 0], p[:, 2], torch.zeros(len(p))], 1)


# 11/04：エンコーダを取り替える（環境変数 ENC．既定は Easy Turn のエンコーダ．enc_plug.py）
feat = E.logmel
if os.environ.get("ENC", "et") != "et":
    import enc_plug as _P
    AC, load_pretrained, feat = _P.AC, _P.load_pretrained, _P.feat
