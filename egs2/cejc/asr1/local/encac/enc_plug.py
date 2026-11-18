"""音響だけのモデル（ac_model.py）のエンコーダを取り替える（2026-11-04，silver13）。環境変数 ENC で選ぶ。
  espnet:<pth>                 提案手法の contextual block conformer（gt と同じ構成．frontend・正規化つき．入力は波形）
  whisper:<repo>[:chunk=N]     Whisper のエンコーダ（入力は Easy Turn と同じ log-mel）．chunk=N なら N フレーム（1 フレーム 20 ms）
                               ごとの塊で，塊の終わりより先を見ない attention にする（ストリーミング）
  w2v:<repo>                   wav2vec 2.0 のエンコーダ（入力は波形を平均 0・分散 1 にしたもの）
  mimi:<repo>                  Mimi（Kyutai）のエンコーダ（因果的な畳み込み＋因果的な transformer → 12.5 Hz．量子化の前）．
                               入力は 24 kHz に直した波形
whisper・w2v・mimi は transformers 4.46（PYTHONPATH に pylib_tf446）で動かす．どれも入力は最後の 30 秒まで．
読み出しは ac_model.py と同じ（フレームの平均 → 線形層で 3 クラス）．学習・評価も train_ac.py・et_dump.py をそのまま使う。
"""
import os, torch
ENC = os.environ["ENC"]
ASR1 = "/home/kobori/2025/espnet/egs2/cejc/asr1"
GTCONF = f"{ASR1}/exp/asr_20261030-turntaking-xfmr-pool_attn-frame2-blk42xh2-gts31-same_n5_p30/config.yaml"


KIND = ENC.split(":", 1)[0]
WSUM = os.environ.get("WSUM") == "1"   # espnet だけ：全部の層の出力の重み付き和を読む（11/06）
TAIL = int(os.environ.get("TAIL", 0))   # espnet だけ：フレームの平均に，最後の TAIL フレーム（1 フレーム約 33 ms）の平均をつないで読む（11/13．長い入力で言い終わりが薄まるため）
MAXS = 30 * 16000


def feat(wav):
    """エンコーダごとの入力"""
    if KIND == "espnet":
        return wav                                 # 波形のまま（特徴量は frontend で作る）．長さも切らない
    wav = wav[-MAXS:]
    if KIND == "whisper":
        import et_model as E
        if ":chunk=" not in ENC:
            return E.logmel(wav)
        # ストリーミング：Whisper の log-mel は「発話全体の最大値 − 8」で下を切るので先を見てしまう．そこまでの最大値で切る
        stft = torch.stft(wav, 400, 160, window=torch.hann_window(400), return_complex=True)
        ls = torch.clamp(E._filters() @ (stft[..., :-1].abs() ** 2), min=1e-10).log10()
        ls = torch.maximum(ls, ls.max(0).values.cummax(0).values[None] - 8.0)
        return ((ls + 4.0) / 4.0).transpose(0, 1)
    if KIND == "w2v":
        return (wav - wav.mean()) / (wav.std() + 1e-7)
    if KIND == "mimi":
        import torchaudio
        return torchaudio.functional.resample(wav, 16000, 24000)
    raise ValueError(KIND)


class ESP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import yaml
        from espnet2.asr.encoder.contextual_block_conformer_encoder import ContextualBlockConformerEncoder
        from espnet2.asr.frontend.default import DefaultFrontend
        from espnet2.layers.global_mvn import GlobalMVN
        cfg = yaml.safe_load(open(GTCONF))
        self.frontend = DefaultFrontend(**cfg["frontend_conf"])
        self.normalize = GlobalMVN(f"{ASR1}/{cfg['normalize_conf']['stats_file']}")
        self.encoder = ContextualBlockConformerEncoder(self.frontend.output_size(), **cfg["encoder_conf"])
        self.head = torch.nn.Linear(cfg["encoder_conf"]["output_size"] * (2 if TAIL else 1), 3)
        if WSUM:   # 各層の出力の重み付き和（SUPERB と同じ．重みは head に置き，線形層と同じ学習率で学ぶ）
            self.head.register_parameter("lw", torch.nn.Parameter(torch.zeros(len(self.encoder.encoders))))
            self._hs = []
            for layer in self.encoder.encoders:
                layer.register_forward_hook(lambda mod, inp, out: self._hs.append(out[0]))

    def _unchunk(self, y, T):
        """contextual block conformer の塊ごとの出力 (B, 塊, block+2, D) を (B, T, D) に戻す（forward_train の最後と同じ）"""
        e = self.encoder
        if y.dim() == 3:
            return y
        out = y.new_zeros(y.size(0), T, y.size(-1))
        offset = e.block_size - e.look_ahead - e.hop_size + 1
        cur = e.block_size - e.look_ahead
        out[:, :cur] = y[:, 0, 1:cur + 1]
        left, bi = e.hop_size, 1
        while left + e.block_size < T and bi < y.size(1):
            out[:, cur:cur + e.hop_size] = y[:, bi, offset:offset + e.hop_size]
            cur += e.hop_size; left += e.hop_size; bi += 1
        out[:, cur:T] = y[:, bi, offset:T - left + 1]
        return out

    def forward(self, feats):
        dev = self.head.weight.device
        lens = torch.tensor([len(f) for f in feats], device=dev)
        x = torch.nn.utils.rnn.pad_sequence(list(feats), batch_first=True).to(dev)
        f, fl = self.frontend(x, lens)
        f, fl = self.normalize(f, fl)
        if WSUM:
            self._hs = []
        h, hl, _ = self.encoder(f, fl)
        if WSUM:
            hs = torch.stack([torch.nn.functional.layer_norm(self._unchunk(y, h.size(1)), y.shape[-1:]) for y in self._hs])
            self._hs = []
            h = (torch.softmax(self.head.lw, 0)[:, None, None, None] * hs).sum(0)
        i = torch.arange(h.size(1), device=dev)[None]
        k = (i < hl[:, None]).float()
        v = (h * k[..., None]).sum(1) / k.sum(1, keepdim=True)
        if TAIL:
            k2 = k * (i >= hl[:, None] - TAIL).float()
            v = torch.cat([v, (h * k2[..., None]).sum(1) / k2.sum(1, keepdim=True)], 1)
        return self.head(v)


class HF(torch.nn.Module):
    """transformers のエンコーダ（whisper・w2v・mimi）＋平均＋線形層"""
    def __init__(self, pretrained=False):
        super().__init__()
        f = ENC.split(":")
        repo = f[1]
        self.chunk = int(f[2].split("=")[1]) if len(f) > 2 else 0
        import transformers as T
        if KIND == "whisper":
            cls, cfg = T.WhisperModel, T.WhisperConfig.from_pretrained(repo)
        elif KIND == "w2v":
            cls, cfg = T.Wav2Vec2Model, T.Wav2Vec2Config.from_pretrained(repo)
        else:
            cls, cfg = T.MimiModel, T.MimiConfig.from_pretrained(repo)
        full = cls.from_pretrained(repo) if pretrained else cls(cfg)
        if KIND == "whisper":
            self.enc, dim = full.encoder, cfg.d_model
        elif KIND == "w2v":
            full.config.mask_time_prob = 0.0                  # SpecAugment 風のマスクは使わない（他のエンコーダとそろえる）
            self.enc, dim = full, cfg.hidden_size
            self.enc.gradient_checkpointing_enable()
        else:
            self.enc = torch.nn.ModuleDict({"encoder": full.encoder, "encoder_transformer": full.encoder_transformer,
                                            "downsample": full.downsample})
            dim = cfg.hidden_size
        del full
        self.head = torch.nn.Linear(dim, 3)

    @property
    def encoder(self):   # train_ac.py が m.encoder で学習率を分ける
        return self.enc

    def _whisper(self, x, lens):
        e = self.enc
        x = torch.nn.functional.gelu(e.conv1(x.transpose(1, 2)))
        x = torch.nn.functional.gelu(e.conv2(x)).transpose(1, 2)
        T_ = x.size(1)
        x = x + e.embed_positions.weight[:T_]
        n = (lens + 1) // 2
        ok = torch.arange(T_, device=x.device)[None, None, :] < n[:, None, None]          # (B, 1, T) 鍵がある所
        if self.chunk:
            q = torch.arange(T_, device=x.device)
            ok = ok & (q[None, :] < ((q[:, None] // self.chunk + 1) * self.chunk))[None]   # (B, T, T)
        mask = torch.zeros(ok.shape, device=x.device, dtype=x.dtype).masked_fill(~ok, torch.finfo(x.dtype).min)[:, None]
        for layer in e.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, None, use_reentrant=False)[0]
            else:
                x = layer(x, mask, None)[0]
        return e.layer_norm(x), n

    def forward(self, feats):
        dev = self.head.weight.device
        lens = torch.tensor([len(f) for f in feats], device=dev)
        x = torch.nn.utils.rnn.pad_sequence(list(feats), batch_first=True).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if KIND == "whisper":
                h, n = self._whisper(x, lens)
            elif KIND == "w2v":
                am = (torch.arange(x.size(1), device=dev)[None] < lens[:, None]).long()
                h = self.enc(x, attention_mask=am).last_hidden_state
                n = self.enc._get_feat_extract_output_lengths(lens)
            else:
                e = self.enc["encoder"](x[:, None])
                if self.training:
                    e = torch.utils.checkpoint.checkpoint(lambda z: self.enc["encoder_transformer"](z)[0], e.transpose(1, 2),
                                                          use_reentrant=False)
                else:
                    e = self.enc["encoder_transformer"](e.transpose(1, 2))[0]
                h = self.enc["downsample"](e.transpose(1, 2)).transpose(1, 2)
                n = torch.clamp((lens + 1919) // 1920, max=h.size(1))       # 1 フレーム 1920 サンプル（24 kHz）．因果的なので後ろの詰め物は前に効かない
        k = (torch.arange(h.size(1), device=dev)[None] < n[:, None]).float()
        return self.head((h.float() * k[..., None]).sum(1) / k.sum(1, keepdim=True))


AC = ESP if KIND == "espnet" else HF


def load_pretrained():
    """事前学習の重みを読む（線形層は新しく作る）"""
    if KIND != "espnet":
        return HF(pretrained=True)
    m = ESP()
    sd = torch.load(ENC.split(":", 1)[1], map_location="cpu")
    sd = {k: v for k, v in sd.items() if k.startswith(("frontend.", "normalize.", "encoder."))}
    r = m.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys and all(k.startswith("head.") for k in r.missing_keys), r
    return m
