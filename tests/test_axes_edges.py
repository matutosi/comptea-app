"""1 つの軸の境を組み立てる段 (axes.py)

2026-09-15 に足した．公開関数 7 つのうち 6 つが的で呼ばれていなかった．
**検出そのものを境にする**のがこの段の考え方で，「表全体を等分する」のは
やめてある．検出漏れの区間だけを内挿し，重複はまとめる．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import axes


def _det(rows, obj_name='row', axis='y'):
    """検出の表 (行なら y の並び，列なら x の並びを渡す)"""
    out = []
    for a, b in rows:
        if axis == 'y':
            out.append({'source_image': 'a.png', 'obj_name': obj_name,
                        'confidence': 0.9, 'x1': 0.0, 'x2': 100.0,
                        'y1': float(a), 'y2': float(b)})
        else:
            out.append({'source_image': 'a.png', 'obj_name': obj_name,
                        'confidence': 0.9, 'y1': 0.0, 'y2': 100.0,
                        'x1': float(a), 'x2': float(b)})
    cols = ['source_image', 'obj_name', 'confidence', 'x1', 'y1', 'x2', 'y2']
    return pd.DataFrame(out, columns=cols)


# --- 範囲 ------------------------------------------------------------------

def test_xの範囲を返す():
    df = _det([(100, 200), (300, 400)], obj_name='sname', axis='x')
    got = axes.locate_x_range(df, 'a.png', obj_name='sname')
    assert float(got['x1'].iloc[0]) == 100.0
    assert float(got['x2'].iloc[0]) == 400.0


def test_yの範囲を返す():
    df = _det([(100, 200), (300, 400)])
    got = axes.locate_y_range(df, 'a.png', obj_name='row')
    assert float(got['y1'].iloc[0]) == 100.0
    assert float(got['y2'].iloc[0]) == 400.0


def test_検出が無ければ範囲も無い():
    assert axes.locate_x_range(_det([]), 'a.png', obj_name='sname') is None


# --- 境の組み立て ----------------------------------------------------------

def test_並んだ検出をそのまま境にする():
    rows = [(100 + i * 50, 145 + i * 50) for i in range(5)]
    edges, interp, warns = axes.locate_edges(_det(rows), 'a.png')
    assert len(edges) == 6                       # 5 行 → 境 6 本
    assert not interp.any()                      # 内挿は無い


def test_抜けた区間だけ内挿する():
    """**検出漏れの区間だけ**内挿する (表全体を等分はしない)"""
    rows = [(100, 145), (150, 195), (250, 295), (300, 345)]
    edges, interp, _w = axes.locate_edges(_det(rows), 'a.png')
    assert int(interp.sum()) == 1                # 抜けた 1 行ぶん
    assert len(edges) == 6                       # 4 本 + 内挿 1 行


def test_検出が1本なら間隔は要らない():
    """1 地点だけの組成表は `col` が 1 本しかない"""
    edges, interp, _w = axes.locate_edges(
        _det([(100, 400)], obj_name='col', axis='x'), 'a.png',
        obj_name='col', axis='x')
    assert list(edges) == [100.0, 400.0]
    assert len(interp) == 1


def test_検出が無ければ境も無い():
    edges, interp, _w = axes.locate_edges(_det([]), 'a.png')
    assert edges is None and interp is None


# --- セルの座標 ------------------------------------------------------------

def test_境の並びからセルを作る():
    x = np.array([0.0, 50.0, 100.0])
    y = np.array([0.0, 30.0, 60.0, 90.0])
    got = axes.coord_item(x, y, obj_name='comp')
    assert len(got) == 6                         # 2 列 x 3 行
    assert set(got['obj_name']) == {'comp'}
    assert float(got['x2'].max()) == 100.0
    assert float(got['y2'].max()) == 90.0


def test_番号はここでは振らない():
    """`col`・`row` は後段の `locate.number_cells` が振る

    この段が返すのは座標と `note` だけ．**段をまたいで通し番号を振る**ので，
    ここで振ると段が 2 つある紙面で番号が衝突する．
    """
    got = axes.coord_item(np.array([0.0, 50.0]), np.array([0.0, 30.0]))
    assert list(got.columns) == ['x1', 'x2', 'y1', 'y2', 'obj_name', 'note']


def test_セルは列ごとに行が並ぶ():
    """並びは np.repeat / np.tile と同じ (列ごとに行が並ぶ)"""
    x = np.array([0.0, 50.0, 100.0])
    y = np.array([0.0, 30.0, 60.0])
    got = axes.coord_item(x, y, obj_name='comp')
    assert list(got['x1'])[:2] == [0.0, 0.0]      # 1 列目の 2 行
    assert list(got['y1'])[:2] == [0.0, 30.0]


def test_印はセルごとに合わせる():
    got = axes.coord_item(np.array([0.0, 50.0]), np.array([0.0, 30.0, 60.0]),
                          x_notes=[['snapped']],
                          y_notes=[['interpolated'], []])
    assert list(got['note']) == ['snapped;interpolated', 'snapped']


# --- セルごとの印 ----------------------------------------------------------

def test_内挿とスナップの印を並べる():
    """`snapped` は**境ごと** (長さ n+1)．接する両側のセルに配る"""
    notes = axes.axis_notes(np.array([True, False]),
                            np.array([False, False, True]),
                            np.array([False, False, False]), n=2)
    assert notes[0] == ['interpolated']
    assert notes[1] == ['snapped']


def test_印が無ければ空():
    notes = axes.axis_notes(n=3)
    assert notes == [[], [], []]


def test_ずらしても重なったままは印に残す():
    notes = axes.axis_notes(np.array([False]), np.array([True, False]),
                            np.array([True, False]), n=1)
    assert 'on_text' in notes[0]


def test_境の印は両側のセルに付く():
    """真ん中の境をずらしたら，その左右どちらのセルにも印が付く"""
    notes = axes.axis_notes(snapped=np.array([False, True, False]), n=2)
    assert notes[0] == ['snapped'] and notes[1] == ['snapped']
