"""表頭のセルを作る段 (locate.py の `_locate_header` から切り出した 3 段)

2026-09-14 に 266 行の `_locate_header` を分けたときに足した．
それまで**この段には的が 1 つも無く**，格子の SHA を突き合わせる形でしか
守れていなかった．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import locate


COLS = ['source_image', 'obj_name', 'x1', 'y1', 'x2', 'y2', 'conf']


def _det(rows):
    """検出の表 (`located.csv` と同じ形の最小版)

    **空でも列は持つ** (工程は CSV から読むので，行が無くても列はある)．
    """
    return pd.DataFrame(
        [{'source_image': 'a.png', 'obj_name': o,
          'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': 0.9}
         for o, x1, y1, x2, y2 in rows],
        columns=COLS)


# --- 項目行の箱を選ぶ ------------------------------------------------------

def test_項目行をそのまま返す():
    df = _det([('plot_row', 0, 10, 100, 30), ('plot_row', 0, 40, 100, 60)])
    got = locate._header_rows(df, 'a.png', [])
    assert len(got) == 2


def test_検出が無ければ空():
    got = locate._header_rows(_det([]), 'a.png', [])
    assert got.empty


def test_組成部の中の誤検出は捨てる():
    """**表頭は組成部より上にしかない** (11_p1 の総合表で出た型)"""
    rows = [('plot_row', 0, 10, 100, 30),        # 表頭 (本体より上)
            ('plot_row', 0, 900, 100, 920)]      # 本体の途中 (捨てる)
    rows += [('row', 0, 500 + i * 50, 100, 540 + i * 50) for i in range(4)]
    warns = []
    got = locate._header_rows(_det(rows), 'a.png', warns)
    assert len(got) == 1
    assert any('組成部の中' in w for w in warns)


def test_すべて組成部の中なら知らせる():
    rows = [('plot_row', 0, 900, 100, 920)]
    rows += [('row', 0, 100 + i * 50, 100, 140 + i * 50) for i in range(4)]
    warns = []
    got = locate._header_rows(_det(rows), 'a.png', warns)
    assert got.empty
    assert any('すべて組成部の中' in w for w in warns)


def test_行の検出が少なければ捨てない():
    """`row` が 3 本未満のときは，上端が当てにならないので触らない"""
    rows = [('plot_row', 0, 900, 100, 920), ('row', 0, 100, 100, 140)]
    got = locate._header_rows(_det(rows), 'a.png', [])
    assert len(got) == 1


# --- 帯と境を作る ----------------------------------------------------------

def test_帯の境は中点に置く():
    df = _det([('plot_row', 0, 10, 100, 30), ('plot_row', 0, 40, 100, 60)])
    rows = locate._header_rows(df, 'a.png', [])
    bands, edges, med, head = locate._header_bands(df, 'a.png', rows, [])
    assert len(bands) == 2
    assert list(edges) == [10, 35, 60]          # 30 と 40 の中点
    assert med == 20


def test_境は下がらない():
    """箱が重なって並ぶと中点が前の境より上に来て，高さが負のセルができる"""
    df = _det([('plot_row', 0, 10, 100, 80), ('plot_row', 0, 20, 100, 40)])
    rows = locate._header_rows(df, 'a.png', [])
    _b, edges, _m, _h = locate._header_bands(df, 'a.png', rows, [])
    assert all(b >= a for a, b in zip(edges, edges[1:]))


def test_表頭の上下の取りこぼしを補う():
    """`header` の範囲に，項目行として拾えていない帯があれば足す"""
    df = _det([('plot_row', 0, 100, 100, 120), ('plot_row', 0, 130, 100, 150),
               ('header', 0, 40, 100, 240)])
    rows = locate._header_rows(df, 'a.png', [])
    warns = []
    bands, _e, _m, _h = locate._header_bands(df, 'a.png', rows, warns)
    assert len(bands) == 4                      # 上と下に 1 つずつ足りた
    assert any('端にある' in w for w in warns)


def test_あいだが広ければ知らせる():
    df = _det([('plot_row', 0, 10, 100, 30), ('plot_row', 0, 100, 100, 120)])
    rows = locate._header_rows(df, 'a.png', [])
    warns = []
    locate._header_bands(df, 'a.png', rows, warns)
    assert any('1行ぶん以上あいた' in w for w in warns)


# --- 値のセルを作る --------------------------------------------------------

def test_値のセルを作る():
    edges = np.array([10.0, 35.0, 60.0])
    out, hx = locate._header_value_cells([0.0, 50.0, 100.0], edges, 2, None, [])
    assert len(out) == 1
    assert list(hx) == [0.0, 50.0, 100.0]       # 画像が無ければずらさない


def test_値のセルの数は列かける行():
    edges = np.array([10.0, 35.0, 60.0])
    out, _hx = locate._header_value_cells([0.0, 50.0, 100.0], edges, 2, None, [])
    cells = out[0]
    n = len(cells) if isinstance(cells, list) else len(cells.get('x1', []))
    assert n == 4                                # 2 列 x 2 行
