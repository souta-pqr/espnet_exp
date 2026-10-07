#!/usr/bin/env python
"""発話区間末検出の作業報告スライドを作る（python-pptx）。

初めて見る人が、背景 → タスク → 何をしたか → 何が分かったか の順で追えるようにする。
図は矢印・箱・折れ線をその場で描く（外部画像に依存しない）。

  python local/make_slides.py --out 発話区間末検出_報告.pptx
"""
import argparse

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt

# ---- 配色（落ち着いた寒色＋強調色）----
INK = RGBColor(0x1A, 0x1A, 0x17)
SOFT = RGBColor(0x55, 0x55, 0x4E)
FAINT = RGBColor(0x8A, 0x8A, 0x82)
LINE = RGBColor(0xD5, 0xD5, 0xCF)
BG = RGBColor(0xFA, 0xFA, 0xF8)
BLUE = RGBColor(0x2C, 0x5F, 0x7C)
BLUEW = RGBColor(0xDD, 0xE8, 0xEF)
ORANGE = RGBColor(0xA8, 0x5A, 0x1E)
ORANGEW = RGBColor(0xF4, 0xE5, 0xD6)
GREEN = RGBColor(0x1D, 0x6B, 0x4A)
GREENW = RGBColor(0xD8, 0xEB, 0xE2)
RED = RGBColor(0x9C, 0x3B, 0x3B)
GRAY = RGBColor(0xE9, 0xE9, 0xE5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Yu Gothic"
W, H = Cm(33.87), Cm(19.05)          # 16:9


def txbox(slide, x, y, w, h, text, size=14, bold=False, color=INK,
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=1.25, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Cm(0)
    tf.margin_top = tf.margin_bottom = Cm(0)
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = space
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = FONT
    return tb


def box(slide, x, y, w, h, fill=WHITE, line=LINE, lw=1.0, shape=MSO_SHAPE.RECTANGLE):
    sh = slide.shapes.add_shape(shape, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    sh.text_frame.word_wrap = True
    return sh


def label_box(slide, x, y, w, h, title, body, accent=BLUE, fill=WHITE,
              tsize=13, bsize=11):
    """左に色帯を持つカード。"""
    box(slide, x, y, w, h, fill=fill, line=LINE)
    bar = box(slide, x, y, Cm(0.12), h, fill=accent, line=None)
    bar.shadow.inherit = False
    txbox(slide, x + Cm(0.45), y + Cm(0.3), w - Cm(0.75), Cm(0.7),
          title, size=tsize, bold=True, color=accent)
    txbox(slide, x + Cm(0.45), y + Cm(1.0), w - Cm(0.75), h - Cm(1.2),
          body, size=bsize, color=SOFT, space=1.3)


def new_slide(prs, title=None, eyebrow=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = box(s, 0, 0, W, H, fill=BG, line=None)
    bg.shadow.inherit = False
    y = Cm(1.0)
    if eyebrow:
        txbox(s, Cm(1.6), Cm(0.85), Cm(30), Cm(0.6), eyebrow, size=10.5,
              color=FAINT, bold=True)
        y = Cm(1.6)
    if title:
        txbox(s, Cm(1.6), y, Cm(30.6), Cm(1.2), title, size=24, bold=True,
              color=INK)
        ln = box(s, Cm(1.6), y + Cm(1.35), Cm(30.6), Cm(0.03), fill=LINE, line=None)
        ln.shadow.inherit = False
    return s


def table(slide, x, y, colw, rows, head_fill=GRAY, size=11.5, rowh=Cm(0.82),
          hl=None, first_bold=True):
    """rows[0] を見出しとする簡易表。hl は (row, col) → 色。"""
    hl = hl or {}
    cy = y
    for ri, row in enumerate(rows):
        cx = x
        for ci, cell in enumerate(row):
            fill = head_fill if ri == 0 else WHITE
            key = (ri, ci)
            colr = INK
            bold = (ri == 0) or (ci == 0 and first_bold)
            if key in hl:
                colr = hl[key]
                bold = True
            sh = box(slide, cx, cy, colw[ci], rowh, fill=fill, line=LINE, lw=0.75)
            tf = sh.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Cm(0.15)
            tf.margin_top = tf.margin_bottom = Cm(0)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(cell)
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = colr
            r.font.name = FONT
            cx += colw[ci]
        cy += rowh
    return cy


def arrow(slide, x1, y1, x2, y2, color=SOFT, lw=1.4):
    from pptx.enum.shapes import MSO_CONNECTOR
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    return c


# ════════════════════════════════════════════════════════════════════
def build(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ── 1. 表紙 ─────────────────────────────────────────────
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = box(s, 0, 0, W, H, fill=BG, line=None); bg.shadow.inherit = False
    band = box(s, 0, 0, Cm(0.5), H, fill=BLUE, line=None); band.shadow.inherit = False
    txbox(s, Cm(2.4), Cm(5.2), Cm(28), Cm(1), "CEJC · 発話区間末検出", size=13,
          color=FAINT, bold=True)
    txbox(s, Cm(2.4), Cm(6.2), Cm(28), Cm(3.6),
          "発話区間末検出の\n精度と早さを詰める", size=36, bold=True, color=INK, space=1.15)
    txbox(s, Cm(2.4), Cm(10.6), Cm(26), Cm(2.4),
          "「相手が言い終えたか」を音声から判定する。\n"
          "応答ミスと応答遅延を減らすための実験と、その結論。",
          size=15, color=SOFT, space=1.5)
    txbox(s, Cm(2.4), Cm(15.6), Cm(28), Cm(1),
          "藤江研究室　小堀聡太", size=13, color=SOFT)

    # ── 2. 背景 ─────────────────────────────────────────────
    s = new_slide(prs, "対話システムが困る 2 つの場面", "背景")
    y = Cm(4.2)
    label_box(s, Cm(1.6), y, Cm(14.6), Cm(4.4), "① 応答ミス",
              "ユーザはもう言い終えているのに、システムが\n"
              "「まだ続く」と判断して応答しない。\n"
              "会話が止まり、ユーザが困る。", accent=RED)
    label_box(s, Cm(17.4), y, Cm(14.8), Cm(4.4), "② 応答遅延",
              "言い終えたかの判断が遅く、応答を作り始める\n"
              "のが遅れる。\n"
              "無言の間が空き、テンポが悪くなる。", accent=ORANGE)

    y2 = Cm(9.6)
    txbox(s, Cm(1.6), y2, Cm(30), Cm(0.8),
          "どちらも「相手が言い終えたか」を正しく・早く判定できれば解決する",
          size=15, bold=True, color=INK)

    label_box(s, Cm(1.6), y2 + Cm(1.2), Cm(14.6), Cm(4.6),
              "既存技術では足りない",
              "音声区間検出（VAD）… 音が鳴っているかどうかだけ。\n"
              "　→ 考えながらの沈黙と、言い終えた沈黙を区別できない\n\n"
              "音声活動予測（VAP）… 次に誰が話し始めるかを音から予測。\n"
              "　→ 予測するのはタイミングであって、発話内容の意味ではない",
              accent=FAINT)
    label_box(s, Cm(17.4), y2 + Cm(1.2), Cm(14.8), Cm(4.6),
              "本研究のタスク：発話区間末検出",
              "発話内容の意味に基づいて「伝えたいことを言い終えたか」を判定する。\n"
              "音声認識と同時に行い、認識誤りの影響を受けにくくする。",
              accent=BLUE, fill=BLUEW)

    # ── 3. タスク定義 ────────────────────────────────────────
    s = new_slide(prs, "3 つのタグを判定する", "タスク定義")
    rows = [["タグ", "意味", "例"],
            ["完了", "伝えたいことを言い終えた",
             "「明日はちょっと無理なんだよね」→ システムが応答してよい"],
            ["継続", "まだ言い終えていない",
             "同じ文でも、続けて「だから来週に…」と言うなら継続"],
            ["相槌", "聞き手からの短い発話", "「うん」「へえ」など"]]
    table(s, Cm(1.6), Cm(4.3), [Cm(3.2), Cm(8.6), Cm(18.8)], rows, rowh=Cm(1.15))

    label_box(s, Cm(1.6), Cm(9.0), Cm(30.6), Cm(3.0),
              "ここが難しい：文の形だけでは決まらない",
              "上の例は完了と継続でまったく同じ文である。文法的に完結していても、"
              "本人がまだ続けるつもりなら継続になる。\n"
              "つまり「文が終わったか」ではなく「話し手が伝え終えたか」を当てる必要がある。",
              accent=ORANGE, fill=ORANGEW)

    label_box(s, Cm(1.6), Cm(12.6), Cm(30.6), Cm(4.6),
              "データ：日本語日常会話コーパス（CEJC）",
              "学習 80.5 時間／評価 9.9 時間。評価は 23,444 区間（完了 10,906・継続 6,255・相槌 6,283）。\n"
              "タグは大規模言語モデル（Qwen3-8B）が書き起こしとタイミングから付与した。"
              "人手正解 1,000 件に対する付与精度は F1 0.805。\n"
              "つまり正解ラベル自体に約 2 割の誤りが含まれており、これが後で効いてくる。",
              accent=BLUE)

    # ── 4. モデル ───────────────────────────────────────────
    s = new_slide(prs, "音声認識モデルに判定機能を後付けする", "提案モデル")
    top = Cm(4.4)
    # 過去発話側
    box(s, Cm(2.0), top, Cm(8.4), Cm(1.1), fill=GRAY)
    txbox(s, Cm(2.0), top + Cm(0.3), Cm(8.4), Cm(0.7), "過去発話の音声（N 発話）",
          size=11.5, align=PP_ALIGN.CENTER, color=SOFT)
    box(s, Cm(12.4), top, Cm(8.4), Cm(1.1), fill=GRAY)
    txbox(s, Cm(12.4), top + Cm(0.3), Cm(8.4), Cm(0.7), "現在の発話の音声",
          size=11.5, align=PP_ALIGN.CENTER, color=SOFT)

    enc_y = top + Cm(1.8)
    for x in (Cm(2.0), Cm(12.4)):
        sh = box(s, x, enc_y, Cm(8.4), Cm(1.3), fill=WHITE, line=FAINT)
        sh.line.dash_style = 4
        txbox(s, x, enc_y + Cm(0.35), Cm(8.4), Cm(0.7),
              "音声認識エンコーダ（凍結）", size=11.5, align=PP_ALIGN.CENTER, color=SOFT)
    box(s, Cm(22.8), enc_y, Cm(6.4), Cm(1.3), fill=WHITE, line=FAINT)
    txbox(s, Cm(22.8), enc_y + Cm(0.35), Cm(6.4), Cm(0.7), "デコーダ（凍結）",
          size=11.5, align=PP_ALIGN.CENTER, color=SOFT)
    arrow(s, Cm(20.8), enc_y + Cm(0.65), Cm(22.8), enc_y + Cm(0.65))
    txbox(s, Cm(22.8), enc_y + Cm(1.6), Cm(6.4), Cm(0.7), "→ 認識テキスト",
          size=11, align=PP_ALIGN.CENTER, color=FAINT)

    arrow(s, Cm(6.2), top + Cm(1.1), Cm(6.2), enc_y)
    arrow(s, Cm(16.6), top + Cm(1.1), Cm(16.6), enc_y)

    head_y = enc_y + Cm(2.1)
    box(s, Cm(2.0), head_y, Cm(18.8), Cm(1.3), fill=BLUEW, line=BLUE, lw=1.4)
    txbox(s, Cm(2.0), head_y + Cm(0.35), Cm(18.8), Cm(0.7),
          "Self-Attention（1 層）　← ここだけ学習する（全体の約 1%）",
          size=11.5, align=PP_ALIGN.CENTER, color=BLUE, bold=True)
    arrow(s, Cm(6.2), enc_y + Cm(1.3), Cm(6.2), head_y)
    arrow(s, Cm(16.6), enc_y + Cm(1.3), Cm(16.6), head_y)

    mlp_y = head_y + Cm(1.9)
    box(s, Cm(13.6), mlp_y, Cm(5.6), Cm(1.1), fill=BLUEW, line=BLUE, lw=1.4)
    txbox(s, Cm(13.6), mlp_y + Cm(0.25), Cm(5.6), Cm(0.7), "MLP",
          size=11.5, align=PP_ALIGN.CENTER, color=BLUE, bold=True)
    arrow(s, Cm(16.4), head_y + Cm(1.3), Cm(16.4), mlp_y, color=BLUE)
    txbox(s, Cm(16.4), mlp_y + Cm(1.3), Cm(10), Cm(0.7),
          "→　完了 / 継続 / 相槌", size=13, bold=True, color=INK)

    label_box(s, Cm(1.6), Cm(15.0), Cm(30.6), Cm(2.6),
              "設計の要点",
              "音声認識の部分は一切変更しない（凍結）ので、認識精度は元のまま（CER 23.3%）。"
              "完成した「耳」に小さな判定器を後付けする形。\n"
              "読み出しは常に「渡された音声の最後の位置」なので、"
              "発話を途中で切って渡せば、そのままその時点での判定になる。これが後の早期確定に効く。",
              accent=BLUE)

    # ── 5. 全体像 ───────────────────────────────────────────
    s = new_slide(prs, "2 つの軸で改善を試みた", "作業の全体像")
    label_box(s, Cm(1.6), Cm(4.3), Cm(15.0), Cm(6.4),
              "軸 A：精度を上げる（応答ミスを減らす）",
              "・過去の発話を文脈として渡す（要約方法 7 種 × 発話数 2 条件）\n"
              "・エンコーダの部分適応（LoRA / Adapter）\n"
              "・学習率スケジュールの見直し\n"
              "・損失関数の変更、ヘッドの大型化、相手の音声の追加\n"
              "・複数モデルのアンサンブル",
              accent=BLUE)
    label_box(s, Cm(17.4), Cm(4.3), Cm(14.8), Cm(6.4),
              "軸 B：早く判定する（応答遅延を減らす）",
              "・発話を途中で切ったときの精度低下を測る\n"
              "・学習時にも途中で切って見せる（打ち切り学習）\n"
              "・いつ判定を確定するかの規則を設計する\n"
              "　（信頼度閾値・経過時間ゲート・コスト最小化など）\n"
              "・実運用条件（発話後の無音・音声区間検出）で測り直す",
              accent=ORANGE)

    label_box(s, Cm(1.6), Cm(11.4), Cm(30.6), Cm(2.4),
              "評価の作法",
              "評価用データは一度も学習に使わない。閾値の調整は開発用データだけで行う。"
              "モデルは乱数の種を変えて 3 本学習し、"
              "「差」が偶然の範囲を超えているかを毎回確かめた。",
              accent=FAINT)

    txbox(s, Cm(1.6), Cm(14.6), Cm(30.6), Cm(3),
          "以下、それぞれの軸で何が分かったかを述べる。",
          size=14, color=SOFT)

    # ── 6. 軸 A の結論 ───────────────────────────────────────
    s = new_slide(prs, "精度は 0.72 前後で頭打ちになった", "軸 A：精度")
    rows = [["試したこと", "結果"],
            ["過去発話を文脈として渡す（19 条件）", "圧縮せず全部渡しても「過去なし」と差なし"],
            ["過去発話の数を 1 → 5 に増やす", "同一手法で完全に同値"],
            ["エンコーダの部分適応（LoRA）", "わずかに改善するが認識精度が劣化"],
            ["学習率スケジュールの修正", "改善（これが最大の効果）"],
            ["損失関数の変更・ヘッドの大型化", "効かないか悪化"],
            ["相手の音声を入れる", "有意に悪化"]]
    hl = {(1, 1): SOFT, (4, 1): GREEN, (6, 1): RED}
    table(s, Cm(1.6), Cm(4.3), [Cm(14.4), Cm(16.2)], rows, rowh=Cm(0.95), hl=hl)

    label_box(s, Cm(1.6), Cm(11.6), Cm(15.0), Cm(5.6),
              "頭打ちの原因は、モデルではなく正解ラベル",
              "正解は言語モデルが書き起こしから付けたもので、人手正解に対する精度は F1 0.805。\n"
              "同じ入力を別の言語モデルに与えて付け直すと、判断が分かれる区間が 3 分の 1 あった。\n"
              "その「2 者が一致する区間」だけで測るとモデルの性能は大きく上がる。\n"
              "つまり残りの誤りの多くは、ラベル自体が定まらない区間に由来する。",
              accent=RED, fill=ORANGEW)
    label_box(s, Cm(17.4), Cm(11.6), Cm(14.8), Cm(5.6),
              "それでも分かった、設計を支持する結果",
              "「音声認識してからテキスト分類器に通す」方式と比べた。\n\n"
              "・実際の認識結果を使った方式より本手法が上\n"
              "・正解の書き起こしを使った理想的な方式とほぼ同等\n\n"
              "認識誤りで劣化しないことが、音声から直接判定する設計の根拠になる。",
              accent=GREEN, fill=GREENW)

    # ── 7. 軸 B の説明 ───────────────────────────────────────
    s = new_slide(prs, "早さの軸：いつ判定を確定するか", "軸 B：早さ")
    txbox(s, Cm(1.6), Cm(4.2), Cm(30.6), Cm(1.2),
          "発話を最後まで聞かずに確定できれば、その分だけ応答が早くなる。"
          "ただし早く切るほど情報が減り、判定を誤りやすくなる。",
          size=14, color=SOFT, space=1.4)

    ty = Cm(6.4)
    txbox(s, Cm(1.6), ty, Cm(30), Cm(0.8), "時間の流れ", size=12, bold=True, color=FAINT)
    bar = box(s, Cm(1.6), ty + Cm(0.9), Cm(17.0), Cm(1.1), fill=BLUEW, line=BLUE)
    txbox(s, Cm(1.6), ty + Cm(1.15), Cm(17.0), Cm(0.7), "ユーザが話している",
          size=11.5, align=PP_ALIGN.CENTER, color=BLUE)
    box(s, Cm(18.6), ty + Cm(0.9), Cm(4.2), Cm(1.1), fill=GRAY, line=LINE)
    txbox(s, Cm(18.6), ty + Cm(1.15), Cm(4.2), Cm(0.7), "無音", size=11.5,
          align=PP_ALIGN.CENTER, color=SOFT)
    box(s, Cm(22.8), ty + Cm(0.9), Cm(9.4), Cm(1.1), fill=GREENW, line=GREEN)
    txbox(s, Cm(22.8), ty + Cm(1.15), Cm(9.4), Cm(0.7), "システムが応答",
          size=11.5, align=PP_ALIGN.CENTER, color=GREEN)

    m1 = Cm(18.6)
    ln = box(s, m1, ty + Cm(0.6), Cm(0.04), Cm(1.7), fill=INK, line=None)
    txbox(s, Cm(15.6), ty + Cm(2.4), Cm(6.4), Cm(0.7), "発話末", size=11,
          align=PP_ALIGN.CENTER, color=INK, bold=True)
    m2 = Cm(22.8)
    ln2 = box(s, m2, ty + Cm(0.6), Cm(0.04), Cm(1.7), fill=SOFT, line=None)
    txbox(s, Cm(20.2), ty + Cm(2.4), Cm(8.4), Cm(0.7), "区間検出が発火（0.25 秒後）",
          size=11, align=PP_ALIGN.CENTER, color=SOFT)
    m3 = Cm(15.2)
    ln3 = box(s, m3, ty + Cm(0.6), Cm(0.04), Cm(1.7), fill=ORANGE, line=None)
    txbox(s, Cm(11.0), ty + Cm(2.4), Cm(8.4), Cm(0.7), "早期確定はここで決める",
          size=11, align=PP_ALIGN.CENTER, color=ORANGE, bold=True)

    label_box(s, Cm(1.6), Cm(11.4), Cm(15.0), Cm(5.8),
              "比較の基準をどこに置くか",
              "実運用では、発話が終わると音声区間検出が 0.25 秒ほどで発火し、判定を促す。\n"
              "したがって「最後まで待つ」ではなく\n"
              "「区間検出まで待つ」が正しい比較基準になる。\n\n"
              "早期確定の価値は、そこからさらに前倒しできる分だけである。",
              accent=BLUE)
    label_box(s, Cm(17.4), Cm(11.4), Cm(14.8), Cm(5.8),
              "評価条件を間違えると結論が変わる",
              "同じモデル・同じデータでも、発話後の無音をどう扱うかで\n"
              "最良の停止規則が入れ替わった。\n\n"
              "・無音なし　　　　　→ ある規則が最良\n"
              "・無音を無制限に聞く → 性能が大きく崩れる\n"
              "・区間検出つき　　　→ また別の規則が最良\n\n"
              "早期確定の評価では、無音と区間検出の扱いを必ず明記すべき。",
              accent=ORANGE, fill=ORANGEW)

    # ── 8. 評価軸の取り替え ──────────────────────────────────
    s = new_slide(prs, "評価の指標を、研究の動機に合わせ直した", "今回の見直し ①")
    label_box(s, Cm(1.6), Cm(4.3), Cm(15.0), Cm(4.2),
              "これまで：macro-F1",
              "3 つのクラスを対等に扱う平均値。\n"
              "しかし継続を当てても、システムは聞き続けるだけで行動は変わらない。\n"
              "動機（応答ミス・応答遅延）と対応していない。",
              accent=FAINT)
    label_box(s, Cm(17.4), Cm(4.3), Cm(14.8), Cm(4.2),
              "今回：失敗の種類で測る",
              "応答ミス率　… 言い終えたのに応答しない割合\n"
              "誤割り込み率… 話の途中で遮ってしまう割合\n"
              "応答の遅延　… 言い終えてから応答するまでの時間",
              accent=BLUE, fill=BLUEW)

    txbox(s, Cm(1.6), Cm(9.2), Cm(30.6), Cm(1.0),
          "応答ミスと誤割り込みは一方を下げれば他方が上がる。"
          "そこで誤割り込みを揃えて比べるのが公平になる。",
          size=13, color=SOFT)

    label_box(s, Cm(1.6), Cm(10.8), Cm(30.6), Cm(2.2),
              "いちばん知りたい問い",
              "どちらの失敗も基準より悪化させずに、どれだけ早く応答できるか。",
              accent=GREEN, fill=GREENW, bsize=14)

    label_box(s, Cm(1.6), Cm(13.6), Cm(30.6), Cm(3.6),
              "指標を変えると結論が変わった例",
              "macro-F1 で比べると、ある規則の取り分は 0.15 秒に見えた。\n"
              "しかし応答ミスと誤割り込みの両方を基準以下に縛ると、実際の取り分は 0.118 秒だった。\n"
              "指標が動機とずれていると、成果を過大に見積もってしまう。",
              accent=ORANGE)

    # ── 9. 主要な成果 ────────────────────────────────────────
    s = new_slide(prs, "完了クラスだけに閾値を置くと、応答が 0.12 秒早くなる", "今回の成果")
    txbox(s, Cm(1.6), Cm(4.1), Cm(30.6), Cm(1.6),
          "早く確定して意味があるのは「完了」だけである。継続を早く確定しても聞き続けるだけ。\n"
          "ところが従来の規則は 3 クラスすべてに早期確定を許し、継続・相槌の誤りまで前倒ししていた。",
          size=13.5, color=SOFT, space=1.4)

    rows = [["乱数の種", "応答ミス", "誤割り込み", "応答が早くなる量"],
            ["seed 30", "0.246", "0.264", "0.121 秒"],
            ["seed 1", "0.274", "0.236", "0.107 秒"],
            ["seed 2", "0.266", "0.246", "0.125 秒"],
            ["平均", "—", "—", "0.118 秒（ばらつき 0.008）"]]
    hl = {(1, 3): GREEN, (2, 3): GREEN, (3, 3): GREEN, (4, 3): GREEN}
    table(s, Cm(1.6), Cm(6.4), [Cm(5.4), Cm(5.0), Cm(5.4), Cm(8.2)], rows,
          rowh=Cm(0.92), hl=hl)

    label_box(s, Cm(25.4), Cm(6.4), Cm(6.8), Cm(4.6),
              "読み方",
              "応答ミスも誤割り込みも\n基準より悪くせずに、\nこれだけ早く\n応答できる。",
              accent=GREEN, fill=GREENW)

    label_box(s, Cm(1.6), Cm(11.8), Cm(15.0), Cm(5.4),
              "従来の推奨規則を全シードで上回った",
              "従来は「全クラス共通の閾値 ＋ 経過時間の下限」を推奨していた。\n"
              "完了だけに閾値を置く方式は、3 本すべてで\n\n"
              "・2 倍以上早く確定し\n"
              "・macro-F1 も高く\n"
              "・応答ミスは基準より低い\n\n"
              "調整するパラメータが 1 つだけなのも利点。",
              accent=GREEN, fill=GREENW)
    label_box(s, Cm(17.4), Cm(11.8), Cm(14.8), Cm(5.4),
              "なぜ効くのか",
              "応答を始めるかどうかを決めるのは「完了」の判定だけである。\n"
              "そこにだけ早期確定を許し、それ以外は区間検出まで待てばよい。\n\n"
              "以前に試した「クラスごとに別々の閾値」が"
              "「相槌に高い閾値・完了に低い閾値」を選んでいたのと同じ構造を、"
              "パラメータ 1 つに畳んだ形になっている。",
              accent=BLUE)

    # ── 10. 否定的結果 ───────────────────────────────────────
    s = new_slide(prs, "無音を学習に入れても効果はなかった", "今回の検証 ②")
    txbox(s, Cm(1.6), Cm(4.1), Cm(30.6), Cm(1.6),
          "従来の見立て：「実運用で性能が落ちるのは、発話の後に無音が続く状況を"
          "モデルが学習で一度も見ていないからだ。無音を含めて学習するしかない」。\n"
          "これを実際に確かめた。",
          size=13.5, color=SOFT, space=1.4)

    rows = [["条件", "応答ミス（基準）", "早くなる量", "判定"],
            ["既存モデル", "0.269", "0.117 秒", "—"],
            ["無音込み（条件を 4 つ変えた）", "0.298", "0.114 秒", "悪化に見えた"],
            ["無音込み（差を無音だけに絞った）", "0.269", "0.125 秒", "差なし"]]
    hl = {(2, 1): RED, (2, 3): RED, (3, 1): INK, (3, 3): INK}
    table(s, Cm(1.6), Cm(6.6), [Cm(11.6), Cm(6.4), Cm(5.6), Cm(6.4)], rows,
          rowh=Cm(1.0), hl=hl)

    label_box(s, Cm(1.6), Cm(11.6), Cm(15.0), Cm(5.6),
              "最初の実験は交絡していた",
              "最初は無音の追加と同時に、学習時の切り出し幅なども変えてしまっていた。\n"
              "そのため性能が落ち、無音が悪いように見えた。\n\n"
              "差を「無音を見せるかどうか」だけに絞って学習し直すと、"
              "基準の応答ミスは 0.269 で完全に一致した。\n"
              "劣化の正体は無音ではなく、切り出し幅の拡大だった。",
              accent=ORANGE, fill=ORANGEW)
    label_box(s, Cm(17.4), Cm(11.6), Cm(14.8), Cm(5.6),
              "結論：効果も害もない",
              "交絡を取り除いても、応答ミス・誤割り込み・早くなる量のいずれも"
              "既存モデルと区別できなかった。\n\n"
              "従来の見立ては否定された。\n"
              "害もないので、設定が単純な既存モデルを使い続けるのが妥当。\n\n"
              "あわせて試した「h 秒以内に終わるか」を予測する仕組みも、"
              "完了の確率と同じ情報しか持たなかった。",
              accent=BLUE)

    # ── 11. まとめ ──────────────────────────────────────────
    s = new_slide(prs, "分かったこと", "まとめ")
    label_box(s, Cm(1.6), Cm(4.3), Cm(15.0), Cm(6.0),
              "精度の軸（応答ミス）",
              "・0.72 前後で頭打ち。原因はモデルではなく正解ラベルの不確かさ\n"
              "・過去発話の文脈は、どう渡しても効果がない\n"
              "・学習率スケジュールの修正がいちばん効いた\n"
              "・音声から直接判定する設計は、認識を経由する方式より強い",
              accent=BLUE)
    label_box(s, Cm(17.4), Cm(4.3), Cm(14.8), Cm(6.0),
              "早さの軸（応答遅延）",
              "・学習時にも発話を途中で切って見せると、途中判定が安定する\n"
              "・完了クラスだけに閾値を置くと 0.118 秒早くなる（3 シードで確認）\n"
              "・発話後の無音を学習に入れても効果はない\n"
              "・評価条件（無音・区間検出）を明記しないと結論を誤る",
              accent=ORANGE)

    label_box(s, Cm(1.6), Cm(11.0), Cm(30.6), Cm(2.6),
              "作業を通して得られた教訓",
              "指標は研究の動機に合わせる。条件を一度に複数変えると、否定的な結果の原因が特定できない。"
              "乱数の種を変えた再現性の確認を、結論を出す前に必ず行う。",
              accent=GREEN, fill=GREENW)

    label_box(s, Cm(1.6), Cm(14.2), Cm(30.6), Cm(3.0),
              "次にやること",
              "・精度の軸を動かすには正解ラベルの質を上げるしかない。複数の言語モデルの判断を統合する方法が候補\n"
              "・早さの軸は、フレーム単位で連続に判定する構成なら、まだ改善の余地がありうる",
              accent=FAINT)

    prs.save(path)
    print(f"保存: {path}　スライド {len(prs.slides.__iter__.__self__._sldIdLst)} 枚")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="発話区間末検出_報告.pptx")
    build(ap.parse_args().out)
