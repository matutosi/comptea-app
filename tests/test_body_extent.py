"""組成部の縦の範囲と，種名との隙間 (body_rows.py)

2026-09-15 に足した．`find_gutter`・`body_extent`・`expected_row_edges` が
的で呼ばれていなかった．

**格子の範囲で数えてはいけない** (2026-09-03)．行の検出が表の一部にしか
出ないと格子もそこだけになり，その中で谷を数えても比が 1.0 になって検査が
鳴らない．**列の検出は表の全高に伸びる**ので，そちらで下端を測る．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import body_rows


def _dark(h=400, w=600, bands=()):
    """縦帯にインクを置いた紙面 (True が黒)"""
    d = np.zeros((h, w), dtype=bool)
    for x1, x2 in bands:
        d[:, x1:x2] = True
    return d


def _loc(rows):
    return pd.DataFrame(
        [{'obj_name': o, 'x1': float(a), 'y1': float(b),
          'x2': float(c), 'y2': float(e), 'row': 0, 'col': 0}
         for o, a, b, c, e in rows])


def _det(rows):
    return pd.DataFrame(
        [{'source_image': 'a.png', 'obj_name': o, 'confidence': 0.9,
          'x1': float(a), 'y1': float(b), 'x2': float(c), 'y2': float(e)}
         for o, a, b, c, e in rows])


# --- 種名と組成のあいだの隙間 ----------------------------------------------

def test_組成部にいちばん近い隙間を返す():
    dark = _dark(bands=[(0, 100), (150, 250), (300, 600)])
    got = body_rows.find_gutter(dark, 0, 300, 0, 400)
    assert got is not None
    assert 250 <= got[0] < got[1] <= 300          # 右側の隙間


def test_狭い隙間は拾わない():
    dark = _dark(bands=[(0, 290), (295, 600)])    # 隙間 5 px
    assert body_rows.find_gutter(dark, 0, 300, 0, 400) is None


def test_範囲が空なら隙間も無い():
    dark = _dark()
    assert body_rows.find_gutter(dark, 100, 100, 0, 400) is None


# --- 組成部の縦の範囲 ------------------------------------------------------

def test_列の検出で下端を測る():
    """**列の検出は表の全高に伸びる**ので，格子より広い範囲が返る"""
    loc = _loc([('comp', 300, 100, 600, 300)])
    det = _det([('col', 300, 80, 600, 900)])
    y1, y2 = body_rows.body_extent(loc, det)
    assert y2 > 300


def test_検出が無ければ格子の範囲():
    loc = _loc([('comp', 300, 100, 600, 300)])
    y1, y2 = body_rows.body_extent(loc)
    assert (y1, y2) == (100.0, 300.0)


def test_種名の箱まで広げられる():
    """`col` の箱が途中で止まると，その下の行がまるごと落ちる (07_p2)"""
    loc = _loc([('comp', 300, 100, 600, 300)])
    det = _det([('col', 300, 100, 600, 320),
                ('sname', 0, 100, 250, 800)])
    _a, without = body_rows.body_extent(loc, det)
    _b, with_names = body_rows.body_extent(loc, det, names=True)
    assert with_names > without


def test_組成部が無ければ範囲も無い():
    got = body_rows.body_extent(_loc([('sname', 0, 0, 100, 100)]))
    assert got is None or got[0] is None or got[1] is None
