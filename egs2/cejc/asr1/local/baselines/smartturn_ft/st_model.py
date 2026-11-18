"""Smart Turn v3 のモデル（pipecat-ai/smart-turn の train.py と同じ構成．BSD-2-Clause）と，公開 ONNX からの重みの読み戻し（2026-10-28）。

公開されているのは ONNX だけなので，ONNX の重みを PyTorch のモデルに移す。
名前のある重み（inner.*）はそのまま，MatMul 用に転置して保存されている重み（val_*）はグラフをたどって行き先を決める。
"""
import numpy as np, torch, torch.nn as nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder


def make_config():
    return WhisperConfig(d_model=384, encoder_layers=4, encoder_attention_heads=6, encoder_ffn_dim=1536,
                         num_mel_bins=80, max_source_positions=400, dropout=0.0, attention_dropout=0.0,
                         activation_dropout=0.0)


class SmartTurnV3(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = make_config()
        self.encoder = WhisperEncoder(cfg)
        d = cfg.d_model
        self.pool_attention = nn.Sequential(nn.Linear(d, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(nn.Linear(d, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                                        nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, input_features):              # (B, 80, 800) → ロジット (B,)
        h = self.encoder(input_features=input_features).last_hidden_state
        w = torch.softmax(self.pool_attention(h), dim=1)
        return self.classifier((w * h).sum(dim=1)).squeeze(-1)


def load_from_onnx(path):
    import onnx
    from onnx import numpy_helper
    g = onnx.load(path).graph
    W = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    sd = {k[len("inner."):]: torch.from_numpy(v.copy()) for k, v in W.items() if k.startswith("inner.")}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    # MatMul の重み（val_*）→ 出力の先の Add のバイアス名で行き先を決める。k_proj はバイアスなし
    lay_mm = {}
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        wn = [i for i in n.input if i in W and not i.startswith("inner.")]
        if not wn:
            continue
        w = wn[0]
        bias = None
        for c in cons.get(n.output[0], []):
            if c.op_type == "Add":
                bias = [i for i in c.input if i.startswith("inner.")]
                bias = bias[0] if bias else None
        lay_mm[w] = bias
    m = SmartTurnV3()
    ks = []
    for w, b in lay_mm.items():
        if b is None:
            ks.append(w)
            continue
        tgt = b[len("inner."):].replace(".bias", ".weight")
        sd[tgt] = torch.from_numpy(W[w].T.copy())
    # バイアスのない MatMul：各層の k_proj（4 つ）と pool_attention.2（[256,1]）
    kk = [w for w in ks if W[w].shape == (384, 384)]
    assert len(kk) == 4, ks
    order = [n.input for n in g.node]
    kk.sort(key=lambda w: next(i for i, ins in enumerate(order) if w in ins))
    for li, w in enumerate(kk):
        sd[f"encoder.layers.{li}.self_attn.k_proj.weight"] = torch.from_numpy(W[w].T.copy())
    for w in ks:
        if W[w].shape == (256, 1):
            sd["pool_attention.2.weight"] = torch.from_numpy(W[w].T.copy())
        elif W[w].shape == (384, 256):
            sd["pool_attention.0.weight"] = torch.from_numpy(W[w].T.copy())
    missing, unexpected = m.load_state_dict(sd, strict=False)
    return m, missing, unexpected
