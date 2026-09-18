"""列を整理する段 (layer_col.py の `fix_columns` から切り出した 3 段)

2026-09-14 に 187 行の `fix_columns` を分けたときに足した．

**切り出しで 1 つ穴が出た**ので，その型をここで押さえる: `hc`・`hx1` は
元は `if have_head:` の中でしか作られず，**表頭の無い表** (11_p1 の総合表) では
変数そのものが無かった．返り値にした拍子に落ちるようになり，代表 5 表の
格子の突き合わせで見つかった (一式は通っていた)．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import layer_col


def _cells(x_edges, y1, y2, obj_name):
    """列ごとに 1 行を持つ格子 (`_col_table` が読める形)"""
    return pd.DataFrame(
        [{'obj_name': obj_name, 'col': i, 'row': 0,
          'x1': float(a), 'x2': float(b), 'y1': float(y1), 'y2': float(y2)}
         for i, (a, b) in enumerate(zip(x_edges[:-1], x_edges[1:]))])


X = [100, 160, 220, 280, 340]


def test_表頭が無くても測れる():
    """**切り出しで落ちた型**．`have_head` が False でも返り値がそろう"""
    dark = np.zeros((400, 400), dtype=bool)
    comp = _cells(X, 100, 380, 'comp')
    cols = layer_col._col_table(comp)
    got = layer_col._column_ink(dark, cols, comp, comp.iloc[0:0])
    hy1, hy2, have_head, widths, body, hb, hc, hx1 = got
    assert have_head is False
    assert hb is None and hc is None and hx1 == 0
    assert len(widths) == len(cols) and len(body) == len(cols)


def test_表頭があれば列ごとに測る():
    dark = np.zeros((400, 400), dtype=bool)
    dark[20:60, 100:160] = True                  # 1 列目の表頭にだけ字
    comp = _cells(X, 100, 380, 'comp')
    head = _cells(X, 10, 80, 'header_value')
    cols = layer_col._col_table(comp)
    _hy1, _hy2, have_head, _w, _b, hb, _hc, _hx1 = layer_col._column_ink(
        dark, cols, comp, head)
    assert have_head is True
    assert hb is not None and len(hb) == len(cols)
    assert hb[0] > hb[1]                         # 字のある列の方が濃い


# --- 右端の要約の列 --------------------------------------------------------

def test_右端の表頭が空なら要約の列にする():
    comp = _cells(X, 100, 380, 'comp')
    cols = layer_col._col_table(comp)
    hb = np.array([0.5, 0.5, 0.5, 0.0])          # 右端だけ表頭が空
    out, summary = layer_col._mark_summary_columns(
        comp.copy(), cols, hb, [], [], len(cols), 0.3)
    assert summary == [3]
    assert list(out.loc[out['col'] == 3, 'obj_name']) == ['summary']
    assert set(out.loc[out['col'] < 3, 'obj_name']) == {'comp'}


def test_表頭を測っていなければ触らない():
    comp = _cells(X, 100, 380, 'comp')
    cols = layer_col._col_table(comp)
    out, summary = layer_col._mark_summary_columns(
        comp.copy(), cols, None, [], [], len(cols), 0.3)
    assert summary == []
    assert set(out['obj_name']) == {'comp'}


def test_右端に字があれば止まる():
    comp = _cells(X, 100, 380, 'comp')
    cols = layer_col._col_table(comp)
    hb = np.array([0.5, 0.5, 0.5, 0.5])
    _out, summary = layer_col._mark_summary_columns(
        comp.copy(), cols, hb, [], [], len(cols), 0.3)
    assert summary == []


# --- 左端の走査 ------------------------------------------------------------

def test_表頭に字のある列に当たったら止める():
    """左端がどれも地点の列なら，何も捨てない"""
    dark = np.zeros((400, 400), dtype=bool)
    comp = _cells(X, 100, 380, 'comp')
    cols = layer_col._col_table(comp)
    hb = np.array([0.5, 0.5, 0.5, 0.5])
    body = np.array([0.4, 0.4, 0.4, 0.4])
    widths = (cols['x2'] - cols['x1']).to_numpy(dtype=float)
    out, dropped, layered, trimmed, split_col = layer_col._scan_left_columns(
        comp.copy(), cols, dark, [], 0.0, None, widths, body, hb, None, 0,
        10.0, 80.0, True, 0.3, 0.5, 0.3)
    assert (dropped, layered, trimmed, split_col) == ([], [], [], None)
    assert len(out) == len(comp)


