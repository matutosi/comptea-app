"""表の載っていないページから「1回出現種」の続きを拾う

組成表の下には「出現1回の種 Außerdem je einmal in Lfd. Nr. 1: …」の流し込みと，
調査地・調査年月日・出典の注記が付く．これが紙面に収まらないと**次のページへ
あふれる**．あふれた先は本文や写真だけのページなので，`looks_like_no_table()`
がページごと捨てていた(2026-09-07 にユーザ指摘で判明．手元の 88 枚では 9 枚)．

`once_species` は検出クラスにあるが，**表の下にある形でしか学習していない**ので，
単独で置かれた塊は出ない(9 枚で conf 0.30 では 0 枚，0.05 まで下げても 3 枚)．
そこで検出には頼らず，**行の文字から塊の範囲を決める**．

塊の見分け方
    始まり  `出現1回の種` / `Außerdem je einmal` の目印がある行．
            目印が無ければ，前のページからの続きとしてページの先頭から見る
    終わり  調査地・調査年月日・出典の注記の手前，
            または種の並びに見えなくなった行の手前

**読むのは EasyOCR ではなく AI(目)を想定している**．学名は斜体で，EasyOCR では
0.62 までしか合わない(2026-09-01 に測定)．ここでは塊の**位置を決めて切り出す**
までを担い，読んだ文字列は `parse_text.parse_once_species()` に渡す．
"""
import re

from . import parse_text

# 塊の始まりの目印．OCR では空白や濁点が崩れるので，ゆるく見る
START_MARK = re.compile(r'出\s*現\s*1?\s*回|Au[sß]{1,2}erdem|einmal', re.I)
# 注記の始まり．ここから先は種の並びではない
NOTE_MARK = re.compile(
    r'調査地|調査年月日|既発表資料|原調査資料|Lage|Fundorte|Datum|Nachweis', re.I)
# 見出しや図表の題．`26. イブキシモツケ群落` `Fig. 54.` `Tab. 26`
HEADING = re.compile(r'^\s*(?:\d{1,3}\s*[.．]\s*\S|Fig\.|Tab\.|Abb\.)', re.I)
# 和名(カタカナ)の連なり
_KANA = re.compile(r'[ァ-ヶー]{2,}')
# 被度らしい断片．`S—+` `K—+・2` `B—2・3` `+・2` `1・2`，
# 常在度表では `II(+)` `I(1)`(kinki_064)
_COVERISH = re.compile(
    r'(?:[BTSKHM][12]?\s*[—–\-ー―一]\s*)?'
    r'(?:[IVivl]{1,4}\s*[(\[{【〔]|[+r十1-5]\s*[・.,]?\s*[1-5]?)')
# 種の並びとみなすのに要る，1行あたりの最小の件数
MIN_HITS = 2
# 前のページからの続きを，ゆるい見分けで探す行数(ページの頭の数行)
HEAD_LINES = 2
# 種の並びに見えない行が，これだけ続いたら塊の終わりとみなす
MAX_MISS = 2


def looks_like_entries(text: str) -> bool:
    """その行が「1回出現種」の並びに見えるか

    種の並びは**ラテン文字の学名・カタカナの和名・被度**が繰り返す．
    本文は地の文なので，カタカナが続いても被度が並ばない．
    """
    s = parse_text.normalize(text)
    if not s or not re.search(r'[A-Za-z]', s):
        return False
    hits = 0
    for m in _KANA.finditer(s):
        tail = s[m.end():m.end() + 8]
        if _COVERISH.match(tail.lstrip(' 　')):
            hits += 1
    return hits >= MIN_HITS


def find_once_span(texts):
    """行の文字の並びから，「1回出現種」の塊の範囲を決める

    Args:
        texts: ページの行の文字(上から順)
    Returns:
        (始まりの行, 終わりの行の次) の組．見つからなければ None
    """
    start = None
    for i, t in enumerate(texts):
        if START_MARK.search(parse_text.normalize(t)):
            start = i
            break
    if start is None:
        # 目印が無いページは，前のページからの続き．置かれ方は2とおりある．
        #   ページの頭から続く       (kinki_005・062・064・073・080)
        #   本文の下に流し込まれる   (kinki_051・082)
        # 頭の数行はゆるく見る(種が1つだけ残ることがある)．
        # それより下は**種の並びがはっきりしている行**だけを手がかりにする
        for i, t in enumerate(texts[:HEAD_LINES]):
            if looks_like_entries(t) or _tail_of_entries(t):
                start = i
                break
            if NOTE_MARK.search(parse_text.normalize(t)):
                break
    if start is None:
        for i, t in enumerate(texts):
            if looks_like_entries(t):
                start = i
                break
        if start is None:
            return None
        # 塊は続いているので，前の行がまだ種の並びなら遡る
        while start > 0 and (looks_like_entries(texts[start - 1])
                             or _tail_of_entries(texts[start - 1])):
            start -= 1
    # 終わりは**注記か見出し**で決める．OCR が崩れた行で切ってしまうと
    # 塊の後ろが落ちるので，種の並びに見えない行は 2 行続いたときだけ切る
    # (切り出しは読ませるためのもので，多めに取るぶんには害が無い)
    stop = len(texts)
    miss = 0
    for i in range(start + 1, len(texts)):
        s = parse_text.normalize(texts[i])
        if NOTE_MARK.search(s) or HEADING.match(s):
            stop = i
            break
        if looks_like_entries(s) or _tail_of_entries(s):
            miss = 0
            continue
        miss += 1
        if miss >= MAX_MISS:
            stop = i - miss + 1
            break
    return (start, stop)


def _tail_of_entries(text: str) -> bool:
    """塊の最後の1行か

    最後の行は種が1つだけのことがあり(`trys ヤマフジ S—+.`)，
    `looks_like_entries()` の件数には届かない．
    学名(ラテン文字)を要るようにして，本文の行を巻き込まないようにする．
    末尾の句点は `一` と読まれる(`tys ヤマフジ S一十一` = `trys ヤマフジ S—+.`)
    ので，被度のあとの区切りらしい字は落として見る．
    """
    s = parse_text.normalize(text)
    return (bool(_KANA.search(s)) and bool(re.search(r'[A-Za-z]', s))
            and bool(re.search(
                r'[+r十1-5]\s*[・.,]?\s*[1-5]?\s*[.。,，一ー―—–・]*\s*$', s)))


def note_text(lines, span):
    """塊の後ろに続く注記の行をつないで返す

    注記は見出し(`調査地` `Datum` など)で始まり，地点ごとの並びが続く．
    始まったあとは「見出しか地点の指定がある行」だけを取り，どちらも無い行が
    来たら止める(本文の段落を巻き込まないため)．
    """
    if not lines:
        return ''
    start = span[1] if span else 0
    out = []
    for ln in lines[start:]:
        s = parse_text.normalize(ln['text'])
        if HEADING.match(s):
            break
        cont = NOTE_MARK.search(s) or parse_text._PLOT_SPEC.search(s)
        if out and not (cont or _note_wrap(s)):
            break
        if cont or out:
            out.append(s)
    return ' '.join(out)


def _note_wrap(text: str) -> bool:
    """注記の折り返しの行か(`Nov 1973` のように見出しも地点も無い続き)

    本文はひらがなが続くので，**ひらがなの連なりが無く，欧字か数字がある**
    行だけを続きとみなす．
    """
    s = parse_text.normalize(text)
    return bool(re.search(r'[A-Za-z0-9]', s)) and not re.search(r'[ぁ-ん]{4,}', s)


def block_box(lines, span, pad=8):
    """塊の外形(x1, y1, x2, y2)．切り出して読ませるために使う"""
    part = lines[span[0]:span[1]]
    if not part:
        return None
    return (min(ln['x1'] for ln in part) - pad,
            min(ln['y1'] for ln in part) - pad,
            max(ln['x2'] for ln in part) + pad,
            max(ln['y2'] for ln in part) + pad)


def read_lines(image, reader=None, line_tol: float = 0.6):
    """ページを丸ごと OCR して，行ごとにまとめる

    位置を決めるためだけの読みなので，**中身の正しさは要らない**．
    塊を切り出したあと，改めて AI に読ませる．
    """
    import numpy as np
    from PIL import Image

    if reader is None:
        import easyocr
        reader = easyocr.Reader(['ja', 'en'])
    img = Image.open(image) if isinstance(image, (str, bytes)) else image
    results = reader.readtext(np.array(img.convert('RGB')))
    return parse_text.group_lines(results, line_tol)


def find_block(image, reader=None):
    """ページから塊を見つけ，外形・注記・目印の有無を返す

    種の並びが無くても**注記だけ**が載るページがある(kinki_011)ので，
    どちらか一方でも見つかれば返す．

    Returns:
        None(どちらも無い) か
        {'box': 外形 or None, 'span': 行の範囲 or None, 'lines': 行の一覧,
         'has_mark': 目印があったか(無ければ前のページからの続き),
         'note': 注記の文字列}
    """
    lines = read_lines(image, reader)
    if not lines:
        return None
    span = find_once_span([ln['text'] for ln in lines])
    note = note_text(lines, span)
    if span is None and not note:
        return None
    return {'box': block_box(lines, span) if span else None, 'span': span,
            'lines': lines,
            'has_mark': bool(span) and bool(START_MARK.search(
                parse_text.normalize(lines[span[0]]['text']))),
            'note': note}
