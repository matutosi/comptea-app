"""物体検出を使わずに部分を推定する別案 (noyolo.py)

いまの工程では使っていない．検出器が落ちる紙面のための控え (2026-09-10 ユーザ提案)．
"""
import numpy as np
from PIL import Image

from comptea import noyolo


PITCH = 30


def _sheet(w=900, h=900):
    """学名 (ラテン)・和名 (カナ)・組成 (短い値) を模した紙面"""
    img = Image.new('L', (w, h), 255)
    px = img.load()

    def fill(x1, x2, y1, y2):
        for x in range(int(x1), int(x2)):
            for y in range(int(y1), int(y2)):
                px[x, y] = 0

    for i in range(20):
        y = 100 + i * PITCH
        for x in range(30, 300, 16):          # 長い連なり (種名)
            fill(x, x + 11, y, y + 14)
        for x in range(340, 520, 16):
            fill(x, x + 11, y, y + 14)
        for x in (600, 700, 800):             # 短い値 (組成)
            fill(x, x + 6, y + 4, y + 10)
    return img


def test_行の高さを自己相関から推定する():
    from comptea import ink
    dark = ink.binarize(_sheet())
    assert abs(noyolo.guess_pitch(dark) - PITCH) <= 2


def test_字のある行だけを粗い行にする():
    from comptea import ink
    dark = ink.binarize(_sheet())
    rows = noyolo.rough_rows(dark, PITCH)
    assert rows and all(90 <= y <= 720 for y in rows)


def test_ラテン文字の列は学名():
    assert noyolo.text_kind(['Quercus glauca', 'Pinus densiflora']) == 'sname'


def test_カタカナの列は和名():
    assert noyolo.text_kind(['アカマツ', 'スダジイ']) == 'species_col'


def test_短い値の列は組成():
    assert noyolo.text_kind(['1.2', '+', '2.2']) == 'comp'


def test_階層の記号は階層():
    assert noyolo.text_kind(['B1', 'S', 'K']) == 'layer'


def test_読めない列は空():
    assert noyolo.text_kind(['', '  ', None]) == 'empty'


def test_ラテンとカナが混じれば多い方():
    # 学名の列に和名が少し混じっても学名 (14_p1 で起きた)
    assert noyolo.text_kind(['Polygonum filiforme', 'ミス', 'Boenninghausenia']) == 'sname'


def test_粗い列は隙間で切る():
    from comptea import ink
    dark = ink.binarize(_sheet())
    rows = noyolo.rough_rows(dark, PITCH)
    cols = noyolo.rough_cols(dark, rows, PITCH)
    assert len(cols) >= 2
    assert all(b > a for a, b in cols)


# --- 辞書で確かめる (案 10・11・12) ---------------------------------------

def test_表頭の語彙の割合():
    assert noyolo.item_score(['通し番号', '調査年月日', 'Quercus']) == 2 / 3
    assert noyolo.item_score([]) == 0.0


def test_被度の形の割合():
    # 「・」だけ (非出現) も通る．種名は通らない
    assert noyolo.comp_score(['1.2', '+', '・', '']) == 1.0
    assert noyolo.comp_score(['アカマツ', 'Pinus']) == 0.0


def test_種名の辞書の割合():
    assert noyolo.name_score(['アカマツ', 'スダジイ', 'ズイナ群集']) >= 2 / 3


def test_行の役割():
    assert noyolo.row_kind('通し番号') == 'item'
    assert noyolo.row_kind('調査年月日') == 'item'
    assert noyolo.row_kind('アカマツ') == 'species'
    assert noyolo.row_kind('ア カ マ ツ') == 'species'      # 空白入りも詰めて引く
    # **学名も種名**とみなす (2026-09-11．09_p5 の最下 4 行は和名が読めず
    # 学名だけが読めていて，本物の行が落ちていた)
    assert noyolo.row_kind('Quercus glauca') == 'species'
    # **1 行に何種も並ぶ行は流し込み** (多めに取ってから OCR で除外する)
    assert noyolo.row_kind('Skimmia japonica ミヤマシキミ K-+, '
                           'Torreya nucifera カヤ B1-1・1') == 'flow'
    assert noyolo.row_kind('Quercus glauca アラカシ K') == 'species'
    assert noyolo.row_kind('Paederia scandens var. mairei ヘクソカズラ B2,S') == 'species'
    assert noyolo.row_kind('') == 'other'


def test_表頭と本体を分ける():
    kinds = [(100, 'item', '通し番号'), (130, 'item', '調査番号'),
             (160, 'other', '群集標徴種'), (190, 'species', 'アカマツ'),
             (220, 'item', '調査地')]           # 本体の下の項目名は表頭に入れない
    head, top = noyolo.split_header(kinds, 30)
    assert head == [100, 130]
    assert top == 190


def test_種名の行が無ければ最後の項目名の次が本体():
    kinds = [(100, 'item', '通し番号'), (130, 'item', '調査番号')]
    head, top = noyolo.split_header(kinds, 30)
    assert head == [100, 130] and top == 160
    assert noyolo.split_header([], 30) == ([], None)


def test_表題のカナ行は本体の上端にしない():
    kinds = [(70, 'species', 'ネザサーススキ群集'),      # 表題．項目名より前
             (490, 'item', '群落記号'), (560, 'item', '通し番号'),
             (840, 'item', '海抜高度'),
             (1200, 'species', 'コナラ'), (1235, 'species', 'アカマツ')]
    head, top = noyolo.split_header(kinds, 35)
    assert head == [490, 560, 840]
    assert top == 1200


def test_名前らしい読みの数():
    assert noyolo._n_name_like(['', '1.2', 'ズイナ群集', '']) == 1
    assert noyolo._n_name_like(['Quercus', 'アカマツ']) == 2


def test_種の行はカナの連なりで見る():
    # 学名・和名・階層・値が 1 行に並ぶ
    assert noyolo.row_kind("Caesalpinia japonica シャケツイバラ S 3・4") == 'species'
    assert noyolo.row_kind('Praf ): H 島ッ 1 s 8 8') == 'other'


def test_黒画素の密度で本体の上端():
    from comptea import ink
    img = Image.new('L', (400, 700), 255)
    px = img.load()
    for i in range(20):                       # 上 6 行は数字 (濃い)，下は「・」
        y = 100 + i * PITCH
        for x in range(100, 380, 40):
            if i < 6:
                for xx in range(x, x + 20):
                    for yy in range(y + 5, y + 20):
                        px[xx, yy] = 0
            else:
                for xx in range(x + 8, x + 11):
                    for yy in range(y + 12, y + 15):
                        px[xx, yy] = 0
    dark = ink.binarize(img)
    rows = list(range(100, 100 + 20 * PITCH, PITCH))
    top = noyolo.body_top_from_ink(dark, (100, 380), rows, PITCH, max_frac=1.0)
    assert top == 100 + 6 * PITCH


# --- 使える案だけを組み合わせたもの (guess_parts_v2) -----------------------

def test_列の等間隔性で組成部の左端():
    """案 6: 右から続く「周期の強い範囲」の左端を返す"""
    from comptea import ink
    img = Image.new('L', (1600, 600), 255)
    px = img.load()
    for r in range(15):                       # 左は種名 (不規則)，右は等間隔の値
        y = 40 + r * 35
        for x in range(30, 700, 23):
            for xx in range(x, x + 13):
                for yy in range(y, y + 16):
                    px[xx, yy] = 0
        for x in range(800, 1560, 80):        # 80 px の周期
            for xx in range(x + 30, x + 44):
                for yy in range(y + 4, y + 14):
                    px[xx, yy] = 0
    dark = ink.binarize(img)
    left = noyolo.comp_left_by_period(dark, 40, 40 + 15 * 35, 35)
    assert left is not None and 600 <= left <= 900


def test_横罫線で本体の上端():
    """案 7: いちばん下の長い横罫線を表頭の下線とみなす"""
    from comptea import ink
    img = Image.new('L', (900, 400), 255)
    px = img.load()
    for x in range(100, 800):                 # 表頭の下線
        for y in (150, 151):
            px[x, y] = 0
    for x in range(200, 260):                 # 本体の短い下線 (罫線ではない)
        px[x, 300] = 0
    dark = ink.binarize(img)
    assert noyolo.body_top_by_rule(dark, 100, 800, 30) in (150.0, 151.0)


def test_罫線が無ければ決まらない():
    from comptea import ink
    img = Image.new('L', (900, 400), 255)
    px = img.load()
    for x in range(200, 260):          # 短い下線だけ (罫線ではない)
        px[x, 300] = 0
    assert noyolo.body_top_by_rule(ink.binarize(img), 100, 800, 30) is None
