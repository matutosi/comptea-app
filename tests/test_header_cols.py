"""表頭の独文と和文を分ける縦の境 (header_cols.py．対策 B)"""
import numpy as np
import pytest
from PIL import Image

from comptea import header_cols, ink


W, H = 1600, 400
BANDS = [(40.0 + 40 * i, 72.0 + 40 * i) for i in range(8)]     # 項目行 8 本
X_LEFT, X_VALUE = 100, 1500                                    # 領域 (項目名の左端〜値の左端)


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _text(px, x1, x2, y1, y2, w=8, gap=5):
    x = int(x1)
    while x + w <= int(x2):
        _fill(px, x, x + w, y1, y2)
        x += w + gap


def _sheet(de=(120, 700), ja=(900, 1300), title=None, rule=True, bands=BANDS):
    """独文 `de`・和文 `ja` の列に字を置いた表頭 (値の列は 1500 から)"""
    img = Image.new('L', (W, H), 255)
    px = img.load()
    for y1, y2 in bands:
        _text(px, de[0], de[1], y1 + 4, y2 - 4)
        _text(px, ja[0], ja[1], y1 + 4, y2 - 4)
        _text(px, X_VALUE + 20, W - 20, y1 + 4, y2 - 4)        # 値の列
    if title:                                                   # 表題や学名の行 (1 行だけ横切る)
        y1, y2 = bands[title]
        _text(px, de[0], ja[1], y1 + 4, y2 - 4)
    if rule:                                                    # 表頭の下の罫線
        _fill(px, 0, W, bands[-1][1] + 4, bands[-1][1] + 7)
    return img


def _split(img, bands=BANDS, left=X_LEFT, value=X_VALUE):
    return header_cols.split_x(ink.binarize(img), (left, value), bands)


def test_独文と和文のあいだの隙間を境にする():
    x = _split(_sheet())
    assert x is not None and 700 < x < 900


def test_和文と値のあいだのより広い隙間は選ばない():
    # 和文の右の余白 (1300〜1500) が独文と和文のあいだ (700〜900) より広い紙面
    x = _split(_sheet(de=(120, 700), ja=(900, 1150)))
    assert x is not None and 700 < x < 900


def test_表題の行が横切っても谷は埋まらない():
    # 黒画素の和では谷が埋まるが，行ごとの票なら 1 行では埋まらない
    img = _sheet(title=2)
    dark = ink.binarize(img)
    prof = dark[int(BANDS[0][0]):int(BANDS[-1][1]), X_LEFT:X_VALUE].sum(axis=0)
    assert (prof[700 - X_LEFT:900 - X_LEFT] > 0).any()          # 和では埋まっている
    x = _split(img)
    assert x is not None and 700 < x < 900


def test_罫線があっても谷が出る():
    img = _sheet(rule=True)
    assert _split(img) is not None


def test_語間の狭い隙間は選ばない():
    x = _split(_sheet(de=(120, 700), ja=(900, 1300)))
    assert x is not None
    # 語間 (5 px) ではなく本物の隙間 (200 px) の中にある
    assert 700 < x < 900


def test_和文が無ければ境を出さない():
    img = Image.new('L', (W, H), 255)
    px = img.load()
    for y1, y2 in BANDS:
        _text(px, 120, 700, y1 + 4, y2 - 4)
        _text(px, X_VALUE + 20, W - 20, y1 + 4, y2 - 4)
    # 700〜1500 が丸ごと空白 (谷の右端が右に寄りすぎ) なので採らない
    assert _split(img) is None


def test_項目行が少なすぎれば境を出さない():
    assert _split(_sheet(bands=BANDS[:2]), bands=BANDS[:2]) is None


def test_境は和文の字に触らない():
    img = _sheet(de=(120, 700), ja=(900, 1300))
    x = _split(img)
    dark = ink.binarize(img)
    xi = int(round(x))
    top, bot = int(BANDS[0][0]), int(BANDS[-1][1])
    assert dark[top:bot, xi].sum() == 0          # 項目行の中では境の線上に黒画素が無い


def test_票の谷は連続した空白を返す():
    votes = np.array([3, 3, 0, 0, 0, 2, 3, 0, 0, 3])
    assert header_cols.valleys(votes, 0) == [(2, 5), (7, 9)]
    assert header_cols.valleys(votes, 2) == [(2, 6), (7, 9)]   # 票 2 の所も谷に入る


@pytest.mark.parametrize('ja_left', [800, 900, 1000, 1100])
def test_和文の左端が動いても境はそのすぐ左に来る(ja_left):
    x = _split(_sheet(de=(120, 700), ja=(ja_left, ja_left + 300)))
    assert x is not None and 700 < x < ja_left
