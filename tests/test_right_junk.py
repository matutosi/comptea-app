"""右端の，字がほとんど無い列を捨てる (layer_col._drop_right_junk)

2026-09-15 に足した．紙面の右の欄外まで格子が伸びることがある
(04_p1 は 16 列のうち右 2 列が x 2582-2747 の欄外で，本体に字のある行が
7.5%・8.1% しかないのに地点として下流へ流れ，`1` や `ダ;n;!` を値として
書き込んでいた)．

既存の `_mark_summary_columns` は**表頭が空**であることを求めるので，
欄外に回転した表題がかかる紙面では止まる．こちらは**本体だけ**を見る．
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from comptea import ink, layer_col

PITCH = 10
N_ROWS = 40


def _sheet_and_grid(n_cols=6, junk=(), share=1.0, width=20):
    """列ごとに字を置いた紙面と，その格子

    `junk` の列は `share` の割合の行にだけ字を置く
    """
    img = Image.new('L', (width * (n_cols + 1), PITCH * (N_ROWS + 2)), 'white')
    d = ImageDraw.Draw(img)
    rows = []
    for c in range(n_cols):
        x1 = width * (c + 1) - width
        for k in range(N_ROWS):
            y = PITCH * (k + 1)
            put = True if c not in junk else (k % max(1, int(1 / share)) == 0)
            if put:
                d.rectangle((x1 + 4, y + 2, x1 + width - 6, y + 7), fill=0)
            rows.append({'obj_name': 'comp', 'block': 1, 'row': k + 1,
                         'col': c + 1, 'x1': x1, 'x2': x1 + width,
                         'y1': y, 'y2': y + PITCH})
    return img.convert('RGB'), pd.DataFrame(rows)


def _dark(img):
    return ink.binarize(img)


def test_右端の空に近い列を捨てる():
    """04_p1: 右 2 列の字のある行は 7.5%・8.1% (他は 98.8%)"""
    img, df = _sheet_and_grid(n_cols=6, junk=(4, 5), share=0.1)
    out, gone = layer_col._drop_right_junk(out=df, dark=_dark(img))
    assert sorted(gone) == [5, 6]
    assert sorted(out['col'].unique().tolist()) == [1, 2, 3, 4]


def test_字のある列は残す():
    img, df = _sheet_and_grid(n_cols=6)
    out, gone = layer_col._drop_right_junk(out=df, dark=_dark(img))
    assert gone == []
    assert out is df


def test_半分ほど字のある列は残す():
    """23_p1 の右端は 0.436 (中央値 0.974)．疎な地点の列は捨てない"""
    img, df = _sheet_and_grid(n_cols=6, junk=(5,), share=0.5)
    out, gone = layer_col._drop_right_junk(out=df, dark=_dark(img))
    assert gone == []


def test_左端の空の列は触らない():
    """左端は `_scan_left_columns` の持ち場"""
    img, df = _sheet_and_grid(n_cols=6, junk=(0,), share=0.05)
    out, gone = layer_col._drop_right_junk(out=df, dark=_dark(img))
    assert gone == []


def test_列が少なければ触らない():
    img, df = _sheet_and_grid(n_cols=3, junk=(2,), share=0.05)
    out, gone = layer_col._drop_right_junk(out=df, dark=_dark(img))
    assert gone == []                      # 中央値が当てにならない


def test_捨てるのは右端から続く分だけ():
    """内側の空の列は，地点が抜けているだけかもしれないので捨てない"""
    img, df = _sheet_and_grid(n_cols=6, junk=(3,), share=0.05)
    out, gone = layer_col._drop_right_junk(out=df, dark=_dark(img))
    assert gone == []


def test_字のある行の割合で見る():
    """黒画素の量ではなく行の数．04_p1 は量では中央値の 2 割で紛らわしい"""
    img, df = _sheet_and_grid(n_cols=6, junk=(5,), share=0.1)
    share = layer_col._ink_share(_dark(img), df)
    assert share[6] < 0.2
    assert np.median([share[c] for c in (1, 2, 3, 4, 5)]) > 0.9
