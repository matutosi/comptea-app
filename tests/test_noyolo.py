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
