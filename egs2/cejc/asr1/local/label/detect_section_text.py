#!/usr/bin/env python3
"""
detect_section_text.py

exp25 用：flat な text ファイルを入力として発話区間末検出を行う。
exp24 からの変更点:
  - モデルを追加: Qwen2.5-7B, Qwen3-8B, Llama-3.1-Swallow-8B, ELYZA-JP-8B
  - Qwen3 の thinking モードを無効化（enable_thinking=False）

exp19/exp21 からの変更点:
  - 行頭の [重複] フラグを廃止し、発話テキスト中に <○○と重複>…</○○と重複>
    のタグを挿入する方式に変更した。○○ には重複相手の話者名が入り、
    「誰の発話と・どの位置で」重複しているかが分かるようにした。
  - 単語ごとのタイムスタンプは無いため、各発話の開始・終了時刻から
    重なり区間の文字位置を時間比で概算する（話速一定を仮定した近似）。

INPUT (text ファイル, 1行=1発話):
  開始=0.048 終了=1.866 話者=IC01_玲子 発話=えそれで落ち(D ム)落ち武者は何
  ...

処理:
  各行（インデックス i）について、前後 context_n / post_context_n 行をそれぞれ
  直前履歴・後続履歴として利用する。
  先頭 context_n 行と末尾 post_context_n 行はスキップする。

OUTPUT:
  [LINE 11] 話者=IC04_美沙
  [HISTORY]
  - 開始=... 終了=... 話者=... 発話=...
  [TARGET]
  - 開始=... 終了=... 話者=... 発話=...
  [POST_HISTORY]
  - 開始=... 終了=... 話者=... 発話=...
  [RESULT] 1
"""

import sys
import types

try:
    import threadpoolctl  # noqa: F401
except ModuleNotFoundError:
    _tpc = types.ModuleType('threadpoolctl')

    class _NullCM:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    _tpc.threadpool_limits = lambda *a, **kw: _NullCM()
    _tpc.threadpoolctl_info = lambda: []

    class _NullController:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def limit(self, *a, **kw): return _NullCM()

    _tpc.ThreadpoolController = _NullController
    sys.modules['threadpoolctl'] = _tpc
    print('[INFO] threadpoolctl mock applied', file=sys.stderr, flush=True)

import os
import re
import argparse
import traceback
import importlib
import importlib.util
from dataclasses import dataclass
from typing import List, Optional, Tuple

MODEL_PRESETS = {
    'llm-jp3.7b':          'llm-jp/llm-jp-3-3.7b-instruct',
    'llm-jp7.2b':          'llm-jp/llm-jp-3-7.2b-instruct',
    'SB-sarashina3B':      'sbintuitions/sarashina2.2-3b-instruct-v0.1',
    'llm-jp4.8b':          'llm-jp/llm-jp-4-8b-base',
    'llm-jp4.8b-thinking': 'llm-jp/llm-jp-4-8b-thinking',
    # exp25 追加
    'Qwen2.5-7B':   'Qwen/Qwen2.5-7B-Instruct',
    'Qwen3-8B':     'Qwen/Qwen3-8B',
    'Swallow-8B':   'tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3',
    'ELYZA-8B':     'elyza/Llama-3-ELYZA-JP-8B',
}

# ===== 事前相槌判定 =====
# 正規化後にこのセット内に含まれる発話はLLMを呼ばず直接 label=2 にする
# 基準: フィラー・笑い・言い直し・不明瞭箇所を除いた実質内容が、
#       聞き手応答語のみからなる（条件①②③を完全に満たす）発話

_AIZUCHI_STRIP_RE = re.compile(
    r'\(F\s+[^)]*\)'          # (F フィラー)
    r'|\(D\s+[^)]*\)'         # (D 言い直し)
    r'|\(R\s+[^)]*\)'         # (R 固有名詞代替)
    r'|<unclear>'              # 不明瞭
    r'|<mask>'                 # マスク済み固有名詞
    r'|<[^<>]+と重複>'         # 重複開始タグ
    r'|</[^<>]+と重複>'        # 重複終了タグ
    r'|\(笑\)'                 # 笑い
)

AIZUCHI_SET: frozenset = frozenset({
    # うん系
    'うん', 'うーん', 'うんうん', 'うんうんうん', 'うんうんうんうん',
    'うんうんうんうんうん', 'うんうんうんうんうんうん',
    'うんー', 'うんそう',
    'うーんうん', 'うんうーん', 'うーんうーん',
    'うーんうんうん', 'うんうんうーん', 'うーんうんうんうん',
    # はい系
    'はい', 'はいはい', 'はいはいはい', 'はいはいはいはい', 'はーい',
    # ええ系
    'ええ', 'ええええ',
    # へー系
    'へー', 'へえ', 'へーへー',
    # ふーん系
    'ふーん', 'ふん', 'ふんふん',
    # あー系
    'あー', 'あーあー', 'あーあーあー', 'あーあーあーあー',
    'あっ', 'ああ', 'ああー', 'あーあ', 'あ',
    # おー系
    'おー', 'おーおー', 'お',
    # えー系
    'えー', 'えーえー', 'え', 'えっ',
    # そう系（述語なし短形のみ）
    'そう', 'そうそう', 'そうそうそう', 'そうそうそうそう', 'そうそうそうそうそう',
    'そっか', 'そうか', 'そうかそうか', 'そっかそっか', 'そ',
    # なるほど系
    'なるほど', 'なるほどなるほど',
    # ほんと系
    'ほんと', 'ほんとに',
    # たしかに系
    'たしかに', '確かに',
    # ん系
    'ん', 'んー', 'んんん', 'んん',
    # ほー系
    'ほー', 'ほーほー', 'ほう',
    # まじ系
    'まじ', 'まじで', 'まじか',
    # その他
    'いやー', 'うわー', 'はー',
    # 複合形（あ/あー + 応答語）
    'あそう', 'あーそう', 'あっそう',
    'あはい', 'あーはい', 'あーはいはい',
    'あーうん',
    'あーそっか', 'あそっか', 'あーそうか',
    'あほんと', 'あ本当',
    'あーなるほど', 'あーなるほどね',
})


def _normalize_for_aizuchi(text: str) -> str:
    """フィラー・笑い・記号類を除去し、実質的な発話内容を返す。"""
    return _AIZUCHI_STRIP_RE.sub('', text).strip()


def is_aizuchi(utt: 'EvalUtterance') -> bool:
    """正規化後の発話内容がAIZUCHI_SETに含まれる、またはすべて除去されて空になる場合True。"""
    norm = _normalize_for_aizuchi(utt.text)
    return norm == '' or norm in AIZUCHI_SET


def is_turn_end_next_different_speaker(
    target: 'EvalUtterance',
    post_history: 'List[EvalUtterance]',
) -> bool:
    """target 終了後に最初に開始する発話が別話者なら True。
    post_history は開始時刻順で渡す。target.end より前に開始するものはスキップ。"""
    target_end = float(target.end)
    for utt in post_history:
        if float(utt.start) >= target_end:
            return utt.speaker != target.speaker
    return False

_tqdm_spec = importlib.util.find_spec('tqdm')
if _tqdm_spec is not None:
    tqdm = importlib.import_module('tqdm').tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable


# ===== データ構造 =====

@dataclass
class EvalUtterance:
    start: str
    end: str
    speaker: str
    text: str


# ===== パーサ =====

_TEXT_LINE_RE = re.compile(
    r'^開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$'
)


def parse_text_file(path: str) -> List[List[EvalUtterance]]:
    """空行で区切られた会話ブロックごとに EvalUtterance リストを返す。
    戻り値: blocks (list of list of EvalUtterance)
    """
    blocks: List[List[EvalUtterance]] = []
    current: List[EvalUtterance] = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip():
                # 空行 = 会話境界
                if current:
                    blocks.append(current)
                    current = []
                continue
            m = _TEXT_LINE_RE.match(line.strip())
            if m is None:
                print(f'[WARN] パース失敗: {line[:80]}', file=sys.stderr, flush=True)
                continue
            start, end, speaker, text = m.groups()
            current.append(EvalUtterance(start=start, end=end, speaker=speaker, text=text))
    if current:
        blocks.append(current)
    return blocks


# ===== ヘルパー関数 =====

def speaker_short(speaker: str) -> str:
    """IC05_玲子 → 玲子"""
    return speaker.split('_')[-1] if '_' in speaker else speaker


# (F …) (D …) (笑) <mask> <unclear> など、内部を分割したくないトークン
_TOKEN_RE = re.compile(r'\([^()（）]*\)|（[^()（）]*）|<[^<>＜＞]*>|＜[^<>＜＞]*＞')


def _snap_index(text: str, idx: int, to_end: bool = False) -> int:
    """idx が (…) や <…> のトークン内部に入る場合、トークンの外へ寄せる。
    to_end=False ならトークン先頭へ、True ならトークン末尾へ寄せる
    （開始タグは手前へ、終了タグは後ろへ寄せ、トークンを分割しない）。"""
    for m in _TOKEN_RE.finditer(text):
        if m.start() < idx < m.end():
            return m.end() if to_end else m.start()
    return idx


def _overlap_spans(
    utt: EvalUtterance,
    others: List[EvalUtterance],
) -> List[Tuple[int, int, str]]:
    """utt のテキスト中で、他話者発話と時間的に重なる区間を
    (開始文字index, 終了文字index, 相手話者短縮名) のリストで返す。

    単語ごとのタイムスタンプは無いため、発話の継続時間に対する
    重なり時間の比率で文字位置を概算する（話速一定を仮定した近似）。
    """
    s_u, e_u = float(utt.start), float(utt.end)
    dur = e_u - s_u
    length = len(utt.text)
    spans: List[Tuple[int, int, str]] = []
    if dur <= 0 or length == 0:
        return spans
    for o in others:
        if o is utt or o.speaker == utt.speaker:
            continue
        s_o, e_o = float(o.start), float(o.end)
        ov_s = max(s_u, s_o)
        ov_e = min(e_u, e_o)
        if ov_e <= ov_s:
            continue
        i_s = _snap_index(utt.text, max(0, min(length, round((ov_s - s_u) / dur * length))))
        i_e = _snap_index(utt.text, max(0, min(length, round((ov_e - s_u) / dur * length))), to_end=True)
        if i_e <= i_s:
            i_e = _snap_index(utt.text, min(length, i_s + 1), to_end=True)
        if i_e <= i_s:  # i_s が末尾に達しているケース
            i_s = max(0, length - 1)
            i_e = length
        spans.append((i_s, i_e, speaker_short(o.speaker)))
    return spans


def annotate_overlaps(utt: EvalUtterance, all_utts: List[EvalUtterance]) -> str:
    """utt.text に <相手と重複>…</相手と重複> タグを挿入した文字列を返す。"""
    spans = _overlap_spans(utt, all_utts)
    if not spans:
        return utt.text
    text = utt.text
    opens: dict = {}
    closes: dict = {}
    for i_s, i_e, name in spans:
        opens.setdefault(i_s, []).append(f'<{name}と重複>')
        closes.setdefault(i_e, []).append(f'</{name}と重複>')
    out: List[str] = []
    for i in range(len(text) + 1):
        for tag in closes.get(i, []):
            out.append(tag)
        for tag in opens.get(i, []):
            out.append(tag)
        if i < len(text):
            out.append(text[i])
    return ''.join(out)


def format_utt_line(utt: EvalUtterance, all_utts: List[EvalUtterance]) -> str:
    duration = float(utt.end) - float(utt.start)
    name = speaker_short(utt.speaker)
    text = annotate_overlaps(utt, all_utts)
    return f'[{name} | {duration:.1f}秒] {text}'


def _format_with_pauses_between(
    utts: List[EvalUtterance],
    all_utts: List[EvalUtterance],
    leading_pause_from: Optional[EvalUtterance] = None,
    trailing_pause_to: Optional[EvalUtterance] = None,
    empty_placeholder: str = '（なし）',
) -> str:
    """発話リストを〔ポーズ: X.X秒〕行付きで整形する。
    leading_pause_from: その発話終端からリスト先頭までのポーズを先頭行に追加。
    trailing_pause_to: リスト末尾からその発話開始までのポーズを末尾行に追加。
    ポーズが0以下（重複）の場合は行を追加しない。
    """
    if not utts:
        return empty_placeholder
    lines: List[str] = []
    if leading_pause_from is not None:
        pause = float(utts[0].start) - float(leading_pause_from.end)
        if pause > 0:
            lines.append(f'〔ポーズ: {pause:.1f}秒〕')
    for i, u in enumerate(utts):
        lines.append(format_utt_line(u, all_utts))
        if i + 1 < len(utts):
            pause = float(utts[i + 1].start) - float(u.end)
            if pause > 0:
                lines.append(f'〔ポーズ: {pause:.1f}秒〕')
    if trailing_pause_to is not None:
        pause = float(trailing_pause_to.start) - float(utts[-1].end)
        if pause > 0:
            lines.append(f'〔ポーズ: {pause:.1f}秒〕')
    return '\n'.join(lines)


def get_next_same_speaker_info(
    target: EvalUtterance,
    post_history: List[EvalUtterance],
    history: List[EvalUtterance] = None,
) -> str:
    # post_history から次の同話者発話と、その手前の他話者発話を取得
    next_same = None
    intervening = []
    for utt in post_history:
        if utt.speaker == target.speaker:
            next_same = utt
            break
        intervening.append(utt)

    if next_same is None:
        return '（以降同話者の発話なし）'

    gap = float(next_same.start) - float(target.end)

    # history 内で「終了時刻がターゲット終了より後かつ次同話者開始より前」の
    # 他話者発話（ギャップ中も継続している発話）を追加でカウントする
    if history:
        target_end  = float(target.end)
        next_start  = float(next_same.start)
        for utt in history:
            if utt.speaker == target.speaker:
                continue
            utt_end = float(utt.end)
            # ギャップ期間（target.end 〜 next_same.start）に終了または継続している発話
            if utt_end > target_end and float(utt.start) <= next_start:
                if utt not in intervening:
                    intervening.append(utt)

    if intervening:
        names = '・'.join(speaker_short(u.speaker) for u in intervening)
        return f'{gap:.1f}秒（その間に他話者{len(intervening)}発話あり: {names}）'
    else:
        return f'{gap:.1f}秒（その間に他話者の発話なし）'


# ===== LLM 検出器 =====

class LLMDetector:
    def __init__(
        self,
        model_name: str,
        device: str = 'auto',
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch

        self.prompt_template: Optional[str] = None
        if prompt_file is not None:
            with open(prompt_file, encoding='utf-8') as pf:
                self.prompt_template = pf.read()
            print(f'[プロンプト読み込み] {prompt_file}', flush=True)

        self.system_prompt: Optional[str] = None
        if system_prompt_file is not None:
            with open(system_prompt_file, encoding='utf-8') as sf:
                self.system_prompt = sf.read().strip()
            print(f'[システムプロンプト読み込み] {system_prompt_file}', flush=True)

        print(f'[モデル読み込み] {model_name}', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_ctx = self.model.config.max_position_embeddings
        self.is_thinking = 'thinking' in model_name.lower()
        self.is_qwen3 = 'qwen3' in model_name.lower()
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'   # 生成タスク: 末尾(質問)を残し先頭から切る
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f'モデル読み込み完了 (最大コンテキスト長: {self.max_ctx} tokens)', flush=True)
        if self.is_thinking:
            print('[INFO] Thinking モデル: max_new_tokens=2048', flush=True)

    def make_prompt(
        self,
        history: List[EvalUtterance],
        target: EvalUtterance,
        post_history: List[EvalUtterance],
    ) -> str:
        all_utts = history + [target] + post_history

        hist_pre_lines = _format_with_pauses_between(
            history, all_utts,
            trailing_pause_to=target,
            empty_placeholder='（履歴なし）',
        )

        hist_post_lines = _format_with_pauses_between(
            post_history, all_utts,
            leading_pause_from=target,
            empty_placeholder='（なし）',
        )

        duration = float(target.end) - float(target.start)
        name = speaker_short(target.speaker)
        target_text = annotate_overlaps(target, all_utts)
        target_line = f'[{name} | {duration:.1f}秒] {target_text}'
        next_gap = get_next_same_speaker_info(target, post_history, history)

        if self.prompt_template is not None:
            return self.prompt_template.format(
                hist_pre_lines=hist_pre_lines,
                target_line=target_line,
                next_gap=next_gap,
                hist_post_lines=hist_post_lines,
            )

        # デフォルトプロンプト（--prompt_file 未指定時のフォールバック）
        return (
            '以下は日本語会話の一部。\n'
            '【判定対象発話】を「はい」「いいえ」「相槌」のいずれかで判定して。\n'
            '手順1: 相手の発話を受けただけの本来の相槌（うん・はい・へえ等の短い形）→「相槌」。\n'
            '手順2: 後続で別話者にターンが移っている（同話者継続間隔が「（以降同話者の発話なし）」）→「はい」。\n'
            '手順3: 同話者が後でまた話している場合、判定対象発話が独り言・明確な質問・'
            '「〜よね」等のターンを譲る文末形式なら「はい」、どれにも当たらなければ「いいえ」。\n'
            '【直前の発話履歴】\n'
            f'{hist_pre_lines}\n'
            '【判定対象発話】\n'
            f'{target_line}\n'
            f'同話者継続間隔: {next_gap}\n'
            '【後続の発話履歴】\n'
            f'{hist_post_lines}\n'
            '以下の1行だけを出力して。余計な前置きやコードブロックは禁止。\n'
            '判定: はい または いいえ または 相槌\n'
        )

    @staticmethod
    def _parse_response(text: str) -> int:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines = [
            ln for ln in lines
            if '判定: はい または いいえ' not in ln  # covers "〜または いいえ" and "〜または いいえ または 相槌"
            and '以下の1行だけを出力して' not in ln
            and 'のいずれかで答えよ' not in ln        # プロンプト冒頭エコー対策: 「はい」か「いいえ」のいずれかで答えよ。
            and 'のどれかに当てはまれば' not in ln    # 判断基準行エコー対策: 〜のどれかに当てはまれば「はい」
        ]
        compact = '\n'.join(lines)

        m = re.search(r'^判定\s*[:：]\s*(はい|いいえ|相槌)\s*$', compact, flags=re.MULTILINE)
        if not m:
            m = re.search(r'^(はい|いいえ|相槌)\s*$', compact, flags=re.MULTILINE)
        if not m:
            for ln in lines[:3]:
                m2 = re.search(r'(はい|いいえ|相槌)', ln)
                if m2:
                    m = m2
                    break

        if m:
            word = m.group(1)
            return 1 if word == 'はい' else (2 if word == '相槌' else 0)
        return 0

    def predict_label(
        self,
        history: List[EvalUtterance],
        target: EvalUtterance,
        post_history: List[EvalUtterance],
    ) -> Tuple[int, str]:
        prompt = self.make_prompt(history, target, post_history)

        if self.system_prompt is not None:
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user',   'content': prompt},
            ]
        else:
            messages = [{'role': 'user', 'content': prompt}]

        ct_kwargs = {'enable_thinking': False} if self.is_qwen3 else {}
        chat_out = self.tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            add_generation_prompt=True,
            **ct_kwargs,
        )
        if hasattr(chat_out, 'input_ids'):
            input_ids = chat_out.input_ids.to(self.model.device)
        else:
            input_ids = chat_out.to(self.model.device)

        token_len = input_ids.shape[1]
        if token_len > self.max_ctx:
            print(
                f'[WARN] 入力トークン長 {token_len} が最大 {self.max_ctx} を超えています。',
                file=sys.stderr, flush=True,
            )

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_new_tokens = 2048 if self.is_thinking else 64
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        gen_ids = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        label = self._parse_response(response)
        return label, prompt

    def predict_label_batch(
        self,
        batch_inputs: List[Tuple['List[EvalUtterance]', 'EvalUtterance', 'List[EvalUtterance]']],
    ) -> Tuple[List[int], List[str]]:
        """複数の (history, target, post_history) をまとめて推論する。
        Returns: (labels, raw_responses)
        """
        chat_strs = []
        for hist, target, post_hist in batch_inputs:
            prompt = self.make_prompt(hist, target, post_hist)
            if self.system_prompt is not None:
                messages = [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user',   'content': prompt},
                ]
            else:
                messages = [{'role': 'user', 'content': prompt}]
            ct_kwargs = {'enable_thinking': False} if self.is_qwen3 else {}
            chat_strs.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **ct_kwargs,
                )
            )

        # 各プロンプトのトークン長を事前確認（切り詰め発生時に警告）
        for i, cs in enumerate(chat_strs):
            tlen = len(self.tokenizer.encode(cs))
            if tlen > self.max_ctx:
                print(
                    f'  [WARNING] バッチ内インデックス{i}: プロンプト {tlen} tokens > '
                    f'max_ctx {self.max_ctx} tokens → 左側を切り詰め',
                    file=sys.stderr, flush=True,
                )

        enc = self.tokenizer(
            chat_strs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_ctx,
        ).to(self.model.device)
        # llm-jp など token_type_ids を返すトークナイザー対策
        enc = {k: v for k, v in enc.items() if k in ('input_ids', 'attention_mask')}

        pad_id = self.tokenizer.pad_token_id
        max_new_tokens = 2048 if self.is_thinking else 64
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        input_len = enc['input_ids'].shape[1]
        labels = []
        responses = []
        for out in output_ids:
            gen_ids = out[input_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            responses.append(response)
            labels.append(self._parse_response(response))
        return labels, responses


# ===== メイン処理 =====

def process(
    text_file: str,
    result_file: str,
    use_llm: bool,
    llm_model: str,
    llm_device: str,
    context_n: int,
    post_context_n: int,
    prompt_file: Optional[str] = None,
    system_prompt_file: Optional[str] = None,
    batch_size: int = 8,
    resume: bool = False,
    max_targets: Optional[int] = None,
    log_responses: Optional[str] = None,
    max_llm_targets: Optional[int] = None,
) -> None:
    print(f'読み込み中: {text_file}', flush=True)
    blocks = parse_text_file(text_file)
    n_blocks = len(blocks)
    n_utts_total = sum(len(b) for b in blocks)
    print(f'  会話ブロック数: {n_blocks}', flush=True)
    print(f'  総発話数      : {n_utts_total}', flush=True)

    # 各ブロックで先頭 context_n・末尾 post_context_n を除いた対象を列挙
    targets = []  # (block_idx, utt_idx_in_block)
    for bi, block in enumerate(blocks):
        for ui in range(context_n, len(block) - post_context_n):
            targets.append((bi, ui))

    print(f'  処理対象      : {len(targets)} 発話', flush=True)

    if not targets:
        raise RuntimeError('処理対象の発話がありません（全ブロックが短すぎる可能性があります）。')

    # レジューム: 既存ファイルの [RESULT] 行数分をスキップ
    n_done = 0
    file_mode = 'w'
    if resume and os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as rf:
            n_done = sum(1 for ln in rf if ln.startswith('[RESULT]'))
        if n_done > 0:
            file_mode = 'a'
            print(f'  レジューム: {n_done} 件スキップ', flush=True)

    remaining = targets[n_done:]
    if max_targets is not None:
        remaining = remaining[:max_targets]
        print(f'  [--max_targets {max_targets}] 先頭 {len(remaining)} 件に限定', flush=True)
    if max_llm_targets is not None:
        print(f'  [--max_llm_targets {max_llm_targets}] LLM処理件数が {max_llm_targets} 件に達したら終了', flush=True)
    print(f'  残り処理対象  : {len(remaining)} 発話 (batch_size={batch_size})', flush=True)

    detector: Optional[LLMDetector] = None
    if use_llm:
        detector = LLMDetector(
            model_name=llm_model,
            device=llm_device,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

    count = {0: 0, 1: 0, 2: 0}
    count_aizuchi_pretag = 0    # 事前タグ付けで相槌判定された件数
    count_nextdiff_pretag = 0   # 事前タグ付けで発話区間末（直後別話者）と判定された件数
    count_llm = 0               # LLMが実際に処理した件数
    global_line = n_done  # 出力上の通し番号
    stop_requested = False      # max_llm_targets 到達フラグ

    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
    log_f = None
    if log_responses:
        os.makedirs(os.path.dirname(os.path.abspath(log_responses)), exist_ok=True)
        log_f = open(log_responses, 'w', encoding='utf-8')
        log_f.write('# raw_response\tparsed_label\tutterance_text\n')

    with open(result_file, file_mode, encoding='utf-8') as f:
        for batch_start in tqdm(range(0, len(remaining), batch_size), desc='判定中'):
            if stop_requested:
                break
            batch_idx = remaining[batch_start : batch_start + batch_size]

            # 事前相槌判定: AIZUCHI_SETに該当するものはLLMをスキップ
            labels = [None] * len(batch_idx)
            raw_responses = [None] * len(batch_idx)
            llm_positions = []  # LLMに渡すbatch_idx内のインデックス
            llm_inputs = []

            for pos, (bi, ui) in enumerate(batch_idx):
                block = blocks[bi]
                target = block[ui]
                pre_hist = block[ui - context_n : ui]
                post_hist = block[ui + 1 : ui + 1 + post_context_n]
                if is_aizuchi(target):
                    labels[pos] = 2
                    count_aizuchi_pretag += 1
                elif is_turn_end_next_different_speaker(target, block[ui + 1:]):
                    labels[pos] = 1
                    count_nextdiff_pretag += 1
                else:
                    # max_llm_targets に達していれば LLM 判定をスキップして 1 扱い
                    if max_llm_targets is not None and count_llm >= max_llm_targets:
                        labels[pos] = 1
                        stop_requested = True
                    else:
                        llm_positions.append(pos)
                        llm_inputs.append((
                            pre_hist,
                            target,
                            post_hist,
                        ))

            if detector is not None and llm_inputs:
                llm_labels, llm_responses = detector.predict_label_batch(llm_inputs)
                for pos, lbl, resp in zip(llm_positions, llm_labels, llm_responses):
                    labels[pos] = lbl
                    raw_responses[pos] = resp
                count_llm += len(llm_inputs)
                if max_llm_targets is not None and count_llm >= max_llm_targets:
                    stop_requested = True
            elif not detector:
                for pos in llm_positions:
                    labels[pos] = 1
                count_llm += len(llm_inputs)

            for idx_in_batch, ((bi, ui), label) in enumerate(zip(batch_idx, labels)):
                block     = blocks[bi]
                target    = block[ui]
                hist      = block[ui - context_n : ui]
                post_hist = block[ui + 1 : ui + 1 + post_context_n]
                global_line += 1
                count[label] += 1

                f.write(f'[LINE {global_line}] 会話={bi + 1} 発話位置={ui + 1} 話者={target.speaker}\n')
                f.write('[HISTORY]\n')
                for u in hist:
                    f.write(f'- 開始={u.start} 終了={u.end} 話者={u.speaker} 発話={u.text}\n')
                f.write('[TARGET]\n')
                f.write(f'- 開始={target.start} 終了={target.end} 話者={target.speaker} 発話={target.text}\n')
                f.write('[POST_HISTORY]\n')
                for u in post_hist:
                    f.write(f'- 開始={u.start} 終了={u.end} 話者={u.speaker} 発話={u.text}\n')
                f.write(f'[RESULT] {label}\n')
                f.write('\n')

                if log_f is not None and raw_responses[idx_in_batch] is not None:
                    raw = raw_responses[idx_in_batch].replace('\t', ' ').replace('\n', '\\n')
                    log_f.write(f'{raw}\t{label}\t{target.text}\n')
            f.flush()
            if log_f is not None:
                log_f.flush()

    if log_f is not None:
        log_f.close()
        print(f'  応答ログ保存  : {log_responses}', flush=True)

    n_proc = sum(count.values())
    n_safe = max(n_proc, 1)
    n_llm_safe = max(count_llm, 1)
    print('\n' + '=' * 60)
    print(f'保存完了: {result_file}')
    print(f'  処理件数            : {n_proc}')
    print(f'    うちLLM処理       : {count_llm}')
    print(f'  発話区間末      (1) : {count[1]} ({100 * count[1] / n_safe:.1f}%)')
    print(f'  非末尾          (0) : {count[0]} ({100 * count[0] / n_safe:.1f}%)')
    print(f'  相槌            (2) : {count[2]} ({100 * count[2] / n_safe:.1f}%)')
    print(f'    うち事前タグ付け  : {count_aizuchi_pretag} ({100 * count_aizuchi_pretag / n_safe:.1f}%)')
    print(f'  発話区間末(事前) (1) : {count_nextdiff_pretag} ({100 * count_nextdiff_pretag / n_safe:.1f}%)')
    print(f'  LLM判定 はい    (1) : {count[1] - count_nextdiff_pretag} / {count_llm}')
    print(f'  LLM判定 いいえ  (0) : {count[0]} / {count_llm}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='flat text ファイルを入力として発話区間末を判定する (exp23)'
    )
    parser.add_argument('--text_file', type=str, required=True,
                        help='入力 text ファイル')
    parser.add_argument('--result_file', type=str, required=True,
                        help='出力ファイル')
    parser.add_argument('--no_llm', action='store_true', default=False,
                        help='LLM を使わず全件 1 にする（動作確認用）')
    parser.add_argument('--model', type=str, default='SB-sarashina3B',
                        help='モデル短縮名または HuggingFace モデル ID')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--context_n', type=int, default=10,
                        help='直前履歴の件数（先頭 context_n 行はスキップ）')
    parser.add_argument('--post_context_n', type=int, default=10,
                        help='後続履歴の件数（末尾 post_context_n 行はスキップ）')
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--system_prompt_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='バッチ推論サイズ（デフォルト: 8）')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='既存の result_file の続きから処理を再開する')
    parser.add_argument('--max_targets', type=int, default=None,
                        help='処理件数の上限（全件数ベース、試験実行用）')
    parser.add_argument('--max_llm_targets', type=int, default=None,
                        help='LLM処理件数の上限（事前タグを除いた実LLM判定件数）')
    parser.add_argument('--log_responses', type=str, default=None,
                        help='LLM の生応答をタブ区切りで保存するファイルパス')
    args = parser.parse_args()

    try:
        resolved_model = MODEL_PRESETS.get(args.model, args.model)
        print(f'[モデル指定] {args.model} -> {resolved_model}', flush=True)

        process(
            text_file=args.text_file,
            result_file=args.result_file,
            use_llm=not args.no_llm,
            llm_model=resolved_model,
            llm_device=args.device,
            context_n=args.context_n,
            post_context_n=args.post_context_n,
            prompt_file=args.prompt_file,
            system_prompt_file=args.system_prompt_file,
            batch_size=args.batch_size,
            resume=args.resume,
            max_targets=args.max_targets,
            log_responses=args.log_responses,
            max_llm_targets=args.max_llm_targets,
        )
    except Exception as e:
        print(f'エラー: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
