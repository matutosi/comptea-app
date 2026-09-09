"""罫線の消し方と，単位を切るときの雑音の床 (row_track.py)

傾いた縦罫線は，まっすぐな連なりを探す消し方では残る．残ると，帯の全部の行に
黒画素があることになり，**単位が縦に数珠つなぎ**になって「字を割った」と誤って数える
(2026-09-09 の調査．階層の切れ 15% の大半がこれだった)．
"""
import numpy as np
from PIL import Image

from comptea import ink, row_track


def _img(w=200, h=600):
    return Image.new('L', (w, h), 255)


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _slanted(px, x0, y1, y2, slope, width=3):
    """右下がり (slope px/px) に傾いた縦線"""
    for y in range(int(y1), int(y2)):
        x = int(round(x0 + slope * (y - y1)))
        _fill(px, x, x + width, y, y + 1)


PITCH = 34.0


def test_傾いた縦罫線も消す():
    img = _img()
    px = img.load()
    _slanted(px, 100, 20, 560, 0.01)          # 540 px で 5 px ずれる縦罫線
    dark = ink.binarize(img)
    out = row_track.clean_rules(dark, 0, 200, 0, 600, PITCH)
    assert dark.sum() > 1000
    assert out.sum() <= dark.sum() * 0.05     # ほとんど消える


def test_字の縦棒は消さない():
    img = _img()
    px = img.load()
    for i in range(12):                        # 各行に 20 px の縦棒 (「1」など)
        _fill(px, 100, 104, 30 + 34 * i, 50 + 34 * i)
    dark = ink.binarize(img)
    out = row_track.clean_rules(dark, 0, 200, 0, 600, PITCH)
    assert out.sum() >= dark.sum() * 0.9


def test_傾いた罫線を消せば単位がつながらない():
    img = _img()
    px = img.load()
    _slanted(px, 60, 20, 560, 0.01)
    for i in range(12):                        # 行ごとの記号 (階層の B1 など)
        _fill(px, 100, 130, 34 * i + 30, 34 * i + 46)
    dark = ink.binarize(img)
    band = row_track.clean_rules(dark, 0, 200, 0, 600, PITCH)
    spans = row_track.unit_spans(band)
    assert len(spans) >= 10                    # 12 個の記号がばらばらに取れる


def test_1画素の斑点では単位をつなげない():
    img = _img()
    px = img.load()
    for i in range(12):
        _fill(px, 100, 130, 34 * i + 30, 34 * i + 46)
    for y in range(20, 560):                   # 全行に 1 px の斑点
        px[150, y] = 0
    dark = ink.binarize(img)
    spans = row_track.unit_spans(dark[0:600, 0:200])
    assert len(spans) >= 10


def test_床は薄い点を消さない():
    # 「・」は小さいが 2 px より太い．消してはいけない
    img = _img()
    px = img.load()
    for i in range(12):
        _fill(px, 100, 105, 34 * i + 32, 34 * i + 37)
    dark = ink.binarize(img)
    spans = row_track.unit_spans(dark[0:600, 0:200])
    assert len(spans) == 12


def test_split_unitsは斑点で数珠つなぎにならない():
    img = _img()
    px = img.load()
    for i in range(12):
        _fill(px, 100, 130, 34 * i + 30, 34 * i + 46)
    for y in range(20, 560):
        px[150, y] = 0
    dark = ink.binarize(img)
    edges = np.arange(20.0, 20 + 34 * 13, 34)
    n_bad, n_all = row_track.split_units(dark, 0, 200, edges, PITCH)
    assert n_all >= 10 and n_bad == 0
