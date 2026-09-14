"""地点の多い表を短冊に分けて検出する段 (strips.py)

2026-09-15 に足した．公開 8 つのうち 6 つが的で呼ばれていなかった．

**縮尺の問題ではない** (`imgsz` を上げても行は 0 件のまま)．行の**横幅**が
学習時の分布から外れるのが原因で，種名の列を頭に付けたまま組成部を
地点のかたまりに分け，**箱を元の座標へ戻す**．戻し方を間違えると段の幅を
測り違えるので，ここを的で固める．
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from comptea import strips


COLS = ['source_image', 'obj_name', 'confidence', 'x1', 'y1', 'x2', 'y2']


def _det(rows):
    return pd.DataFrame(
        [{'source_image': 'a.png', 'obj_name': o, 'confidence': 0.9,
          'x1': float(a), 'y1': float(b), 'x2': float(c), 'y2': float(e)}
         for o, a, b, c, e in rows], columns=COLS)


# --- 種名の側と組成部の境 --------------------------------------------------

def test_種名の側と組成部を分ける():
    det = _det([('sname', 100, 100, 500, 2000),
                ('species_col', 500, 100, 800, 2000),
                ('col', 900, 100, 1000, 2000),
                ('col', 1000, 100, 1100, 2000)])
    got = strips.layout(det, width=3000)
    assert got is not None
    name_x2, body_x1, body_x2 = got
    # **組成部の左は，まるごと種名の側とする** (検出された種名の右端では
    # 切らない．取れなかった列が短冊から外れるため)
    assert name_x2 == body_x1 == 900
    assert body_x2 == 1100


def test_検出が足りなければ決められない():
    assert strips.layout(_det([]), width=3000) is None


# --- 列の境から `col` を作り直す --------------------------------------------

def test_境から列の検出を作る():
    """`like` は雛形の行 (`source_image` などを引き継ぐ)"""
    like = {'source_image': 'a.png', 'obj_name': 'col', 'confidence': 0.5,
            'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    got = strips.cols_from_edges([100.0, 200.0, 300.0], 50.0, 900.0, like)
    assert len(got) == 2
    assert set(got['source_image']) == {'a.png'}
    assert set(got['obj_name']) == {'col'}
    assert float(got['x1'].min()) == 100.0
    assert float(got['x2'].max()) == 300.0
    assert float(got['y1'].min()) == 50.0


def test_境が1本なら列はできない():
    like = {'source_image': 'a.png', 'obj_name': 'col', 'confidence': 0.5,
            'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    assert len(strips.cols_from_edges([100.0], 0.0, 100.0, like)) == 0


# --- 短冊の画像 ------------------------------------------------------------

def test_種名の列と組成の一部をつなぐ():
    im = Image.new('RGB', (2000, 1000), (255, 255, 255))
    got = strips.make_strip(im, table_x1=100, name_x2=500,
                            x1=1200, x2=1600, y1=0, y2=1000)
    # 種名の幅 (500-100) + 組成の幅 (1600-1200)
    assert got.size == (400 + 400, 1000)


# --- 元の座標へ戻す --------------------------------------------------------

def test_短冊の座標を元へ戻す():
    """短冊の x は「種名の幅 + 組成の切り出しからの位置」"""
    df = _det([('col', 450, 10, 500, 900)])      # 種名 400 幅の右 50-100
    got = strips.map_back(df, table_x1=100, name_x2=500, x1=1200,
                          body_x1=500, body_x2=1900, y1=20)
    assert float(got['x1'].iloc[0]) == 1250.0    # 1200 + (450 - 400)
    assert float(got['y1'].iloc[0]) == 30.0      # 縦にも戻す


def test_組成部をまたぐ列は左端を揃える():
    """組成部を丸ごと 1 列とみなした `col` は，短冊の継ぎ目をまたいで
    種名の側から始まることがある．そのまま戻すと**表の幅いっぱいの巨大な
    列**になり，格子が数本に潰れる (2026-09-02 に p3 で起きた)．
    左端を組成部の始まりに揃えてから戻す．
    """
    df = _det([('col', 50, 10, 700, 900)])       # 種名の側から始まる
    got = strips.map_back(df, table_x1=100, name_x2=500, x1=1200,
                          body_x1=500, body_x2=1900)
    assert len(got) == 1
    # 左端は組成の切り出しの先頭 (1200)．種名の側 (100〜500) へは戻らない
    assert float(got['x1'].iloc[0]) == 1200.0


def test_組成部に出た種名の列も捨てる():
    df = _det([('sname', 450, 10, 700, 900)])
    got = strips.map_back(df, table_x1=100, name_x2=400, x1=1200,
                          body_x1=400, body_x2=1900)
    assert got.empty


def test_空なら空を返す():
    got = strips.map_back(_det([]), 100, 500, 1200, 500, 1900)
    assert got.empty
