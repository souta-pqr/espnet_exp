"""Easy Turn（ASLP-lab．Apache-2.0）を読み込み，「書き起こし＋状態のタグ」の確率を出す（2026-11-01，silver13）。

構成：Whisper medium 相当のエンコーダ（24 層・1024 次元）→ 4 倍に間引き → 4 層の transformer → Qwen2.5-0.5B-Instruct。
出力は「書き起こし<complete>」のような文字列（タグは <complete>・<incomplete>・<backchannel>・<wait>）。
  load(ckpt) → モデル
  logmel(wav) → (T, 80)．公式（wenet/dataset/processor.py の compute_log_mel_spectrogram）と同じ計算
  generate(m, feats) → 書き起こしとタグ（貪欲．公式の generate と同じ設定）
  tag_probs(m, feats_list, texts) → 書き起こしを与えたときに各タグが続く確率 (B, 4)
"""
import sys, yaml, torch
ET = "/home/kobori/2026/work/easy_turn/Easy_Turn"
sys.path.insert(0, ET)
QWEN = "/home/kobori/2026/work/easy_turn/ckpt/Qwen2.5-0.5B-Instruct"
CONF = f"{ET}/examples/wenetspeech/whisper/conf/train.yaml"
PROMPTS = yaml.safe_load(open(f"{ET}/examples/wenetspeech/whisper/conf/prompt.yaml"))
TASK = "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"
PROMPT = PROMPTS[TASK][0]                    # 推論ではいつも 1 つ目を使う（公式は毎回ランダム）
TAGS = ["<complete>", "<incomplete>", "<backchannel>", "<wait>"]


def load(ckpt, device="cuda", lora=False):
    from wenet.transformer.encoder import TransformerEncoder      # 音響だけ（ac_model.py）では読まない（silver12 の環境に無い部品を使う．11/03）
    from wenet.llm_asr.llmasr_model import LLMASR_Model
    cfg = yaml.safe_load(open(CONF))
    enc = TransformerEncoder(80, **cfg["encoder_conf"])
    m = LLMASR_Model(encoder=enc, encoder_output_dim=1024, llm_path=QWEN, lora=lora, is_inference=True,
                     downsample_rate=cfg["downsample_rate"], adapter_type=cfg["adapter_type"],
                     speech_token_num=cfg["speech_token_num"])
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    r = m.load_state_dict(sd, strict=False)
    return m.to(device).eval(), r


_F = None


def _filters():
    global _F
    if _F is None:
        import librosa
        _F = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=80))
    return _F


def logmel(wav):
    """wav：float の 1 次元 tensor（16 kHz）→ (T, 80)"""
    stft = torch.stft(wav, 400, 160, window=torch.hann_window(400), return_complex=True)
    mel = _filters() @ (stft[..., :-1].abs() ** 2)
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).transpose(0, 1)


def _prefix(m, feats_list):
    """プロンプト・<bos>・音声・<eos> の埋め込み（行ごとに長さが違う）のリスト"""
    dev = next(m.parameters()).device
    lens = torch.tensor([len(f) for f in feats_list], device=dev)
    x = torch.nn.utils.rnn.pad_sequence(feats_list, batch_first=True).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sp, mask = m.get_embedding_from_wav(x, lens)
    sp = sp.to(m.embed_tokens.weight.dtype)
    sp, mask, _ = m._add_bos_eos(m.speech_token_num, 1 + m.speech_token_num, sp, mask, None)
    pr = m.tokenizer([PROMPT], return_tensors="pt")["input_ids"].to(dev)
    pe = m.embed_tokens(pr)[0]
    sp = sp.to(pe.dtype)                     # <bos>・<eos> の埋め込みは float32 なので LLM（bf16）にそろえる
    return [torch.cat([pe, sp[i][mask[i]]], 0) for i in range(len(feats_list))]


@torch.no_grad()
def generate(m, feats):
    e = _prefix(m, [feats])[0][None].to(torch.bfloat16)
    att = torch.ones(e.shape[:2], dtype=torch.long, device=e.device)
    o = m.llama_model.generate(inputs_embeds=e, attention_mask=att, max_new_tokens=m.max_length, num_beams=1,
                               do_sample=False, repetition_penalty=m.repetition_penalty, eos_token_id=151643,
                               pad_token_id=151643, use_cache=True)
    return m.tokenizer.batch_decode(o, skip_special_tokens=True)[0]


@torch.no_grad()
def tag_probs(m, feats_list, texts):
    """書き起こし texts[i] の後に各タグが続く確率（4 つで正規化）→ (B, 4)．タグのトークン列の対数確率の和で比べる"""
    tok = m.tokenizer
    tag_ids = [tok(t, add_special_tokens=False)["input_ids"] for t in TAGS]
    pre = _prefix(m, feats_list)
    seqs, tgt = [], []
    for p, txt in zip(pre, texts):
        tid = tok(txt, add_special_tokens=False)["input_ids"] if txt else []
        for ti in tag_ids:
            ids = torch.tensor(tid + ti, device=p.device)
            seqs.append(torch.cat([p, m.embed_tokens(ids)], 0))
            tgt.append((len(p) + len(tid), ti))
    L = max(len(s) for s in seqs)
    E = torch.zeros(len(seqs), L, seqs[0].shape[1], device=pre[0].device, dtype=seqs[0].dtype)
    A = torch.zeros(len(seqs), L, dtype=torch.long, device=E.device)
    for i, s in enumerate(seqs):
        E[i, :len(s)] = s
        A[i, :len(s)] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = m.llama_model.model(inputs_embeds=E, attention_mask=A).last_hidden_state
        # 語彙（15 万）全体のロジットを全位置で出すとメモリが足りないので，タグの位置だけ lm_head に通す
        sc = torch.empty(len(seqs), device=E.device)
        for i, (st, ti) in enumerate(tgt):
            idx = torch.arange(st - 1, st - 1 + len(ti), device=E.device)
            lg = m.lm_head(h[i, idx]).float().log_softmax(-1)
            sc[i] = lg[torch.arange(len(ti), device=E.device), torch.tensor(ti, device=E.device)].sum()
    return sc.view(-1, len(TAGS)).softmax(-1).cpu()
