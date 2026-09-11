"""できあがった格子を検査する 4 つ

どれも「行の数も列の数もそろっているのに壊れている」表を見つけるために
足したもの (docs/lessons.md「物差しの落とし穴」)．**鳴るべきときに鳴り，
正しい表では鳴らない**ことを，組み立てた格子で確かめる．
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from comptea import checks


def grid(rows=20, cols=6, x0=200, y0=100, w=60, h=30, obj='comp', first=1):
    """行 x 列のふつうの格子を作る(`first` は行番号の始まり)"""
    out = []
    for r in range(first - 1, first - 1 + rows):
        for c in range(cols):
            out.append({'obj_name': obj, 'row': r + 1, 'col': c + 1,
                        'x1': x0 + c * w, 'x2': x0 + (c + 1) * w,
                        'y1': y0 + (r - first + 1) * h,
                        'y2': y0 + (r - first + 2) * h})
    return pd.DataFrame(out)


# --- 行の高さのばらつき -------------------------------------------------

def test_ふつうの格子では鳴らない():
    assert checks.check_row_heights(grid()) == []


def test_1行だけ極端に高いと鳴る():
    """表題の上に出た `row` の誤検出 1 本で，格子が上へ伸びる

    s01115_07_p1 の最大の行は 1,246 px で他の 37 倍だった．行の数も列の数も
    基準を満たしていたので，この物差しでしか見つからない(2026-09-03)．
    """
    df = grid()
    df.loc[df['row'] == 1, 'y1'] -= 1200          # 表頭を丸ごと飲み込んだ行
    warn = checks.check_row_heights(df)
    assert len(warn) == 1 and '行の高さが極端に不揃い' in warn[0]


def test_傾き補正で列ごとに_y_がずれても高さは変わらない():
    """行の高さは**セルごと**に測る (2026-09-12)

    傾き補正 (`row_skew`・`row_track.fix_offsets`) のあとは同じ行でも列ごとに
    y が違う．行の全セルの min/max で測ると高さが水増しされ (14_p1 は 35 px の
    行が 49 px)，物差しが紙面の傾きで動いてしまう．
    """
    df = grid(h=30)
    for c in range(1, 7):
        df.loc[df['col'] == c, ['y1', 'y2']] += (c - 1) * 3      # 最大 15 px ずれ
    h = (df.assign(h=df['y2'] - df['y1']).groupby('row')['h'].median())
    assert h.median() == 30                                     # セルごとなら 30 px
    g = df.groupby('row').agg(a=('y1', 'min'), b=('y2', 'max'))
    assert (g['b'] - g['a']).median() == 45                     # 帯なら 45 px
    assert checks.check_row_heights(df) == []


def test_行が少なすぎるときは判じない():
    """2 行の表で変動係数を出しても当てにならない"""
    assert checks.check_row_heights(grid(rows=2)) == []


# --- 表頭の行が組成部に下りていないか -----------------------------------

def test_表頭が上にあれば鳴らない():
    body = grid(rows=10, y0=500)
    head = grid(rows=3, y0=300, obj='header_value')
    assert checks.check_header_rows(pd.concat([body, head])) == []


def test_表頭の行が組成部の中にあると鳴る():
    """項目行の誤検出が，組成部の行を表頭として二重に切る

    s01115_05_p2 は組成部の中に 149 行あった(2026-09-04)．
    """
    body = grid(rows=10, y0=500)
    head = pd.concat([grid(rows=2, y0=300, obj='header_value'),
                      grid(rows=3, y0=600, obj='header_value', first=10)])
    warn = checks.check_header_rows(pd.concat([body, head]))
    assert len(warn) == 1 and '3 行が組成部の中にある' in warn[0]


def test_表頭が無ければ判じない():
    assert checks.check_header_rows(grid()) == []


# --- 画像を見る 2 つ ----------------------------------------------------

def page(df, tmp_path):
    """格子に合わせた紙面を描いて置く(検査はファイルを開く)"""
    a = np.full((int(df['y2'].max()) + 50, int(df['x2'].max()) + 50), 255,
                dtype=np.uint8)
    for _, r in df[df['obj_name'] == 'comp'].iterrows():
        cx = int((r['x1'] + r['x2']) / 2)
        cy = int((r['y1'] + r['y2']) / 2)
        a[cy - 6:cy + 6, cx - 10:cx + 10] = 0      # 値のかたまり
    p = tmp_path / 'page.png'
    Image.fromarray(a).save(p)
    return str(p)


def test_列の境が隙間に乗っていれば鳴らない(tmp_path):
    df = grid(cols=14)
    warn = checks.check_grid_columns(page(df, tmp_path), df)
    assert warn == []


def test_列が少ない表は見ない(tmp_path):
    """地点の少ない表では，隙間が地点の区切りとは限らない(2026-09-02)"""
    df = grid(cols=5)
    assert checks.check_grid_columns(page(df, tmp_path), df) == []


def cols_of(df):
    """組成部の縦の範囲を決める `col` の検出(検査はこれで下端を測る)

    **格子自身から測ってはいけない**．行が落ちている表では谷も落ちた範囲の
    中でしか数えず，検査が鳴らない(68 表中 28 表がこれで見逃されていた．
    2026-09-04)．
    """
    comp = df[df['obj_name'] == 'comp']
    out = []
    for c, g in comp.groupby('col'):
        out.append({'obj_name': 'col', 'row': None, 'col': c,
                    'x1': g['x1'].min(), 'x2': g['x2'].max(),
                    'y1': g['y1'].min(), 'y2': g['y2'].max()})
    return pd.DataFrame(out)


def test_行が落ちていれば鳴る(tmp_path):
    """組成部は非出現でも「・」があるので，どの行にも字がある

    印字が 20 行あるのに格子が 5 行しか覆っていない表を作って確かめる．
    """
    full = grid(rows=20)
    img = page(full, tmp_path)
    few = full[full['row'] <= 5]
    warn = checks.check_grid_rows(img, few, df_det=cols_of(full))
    assert warn and '行が丸ごと落ちている' in warn[0]


def test_行が足りていれば鳴らない(tmp_path):
    df = grid(rows=20)
    assert checks.check_grid_rows(page(df, tmp_path), df,
                                  df_det=cols_of(df)) == []


@pytest.mark.parametrize("fn", [checks.check_row_heights, checks.check_header_rows])
def test_空の格子でも落ちない(fn):
    empty = pd.DataFrame(columns=['obj_name', 'row', 'col',
                                  'x1', 'x2', 'y1', 'y2'])
    assert fn(empty) == []
