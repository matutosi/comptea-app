"""2 つの通しを比べる物差し (comptea.compare)

2026-09-16 に使い捨てのスクリプトから切り出した．
要確認の数だけでは直しの良し悪しを決められない (同じ日に 4 回)．
"""
import math

import numpy as np
import pandas as pd

from comptea import compare


def _long(rows):
    return pd.DataFrame(rows, columns=['plot', 'row_no', 'comp_raw'])


def test_同じ表なら変化0():
    a = _long([(1, 10, '+'), (2, 10, '1;2')])
    d = compare.diff_long(a, a.copy())
    assert d['changed'] == 0 and d['cells'] == []


def test_値の変化を数える():
    a = _long([(1, 10, '+'), (1, 11, '2'), (2, 10, '1')])
    b = _long([(1, 10, '+;2'), (1, 11, '2'), (2, 10, '1')])
    d = compare.diff_long(a, b)
    assert d['changed'] == 1
    assert d['cells'] == [(1, 10, '+', '+;2')]
    assert d['by_plot'] == {1: 1}


def test_消えた行と増えた行も数える():
    a = _long([(1, 10, '1'), (1, 11, '2')])
    b = _long([(1, 10, '1'), (1, 12, '+')])
    d = compare.diff_long(a, b)
    assert d['changed'] == 2                     # 行 11 が消え，行 12 が増えた
    assert (1, 11, '2', None) in d['cells']
    assert (1, 12, None, '+') in d['cells']


def test_行番号の無い行を毎回別物と数えない():
    """**2026-09-16 の誤り**: `(地点, NaN)` を鍵にすると `NaN != NaN` で
    同じ行が毎回別物になり，押し出しの副作用を 21 件でなく 253 件と数えていた
    """
    nan = math.nan
    a = _long([(2, nan, '+;2'), (2, nan, '3'), (1, 10, '1')])
    b = _long([(2, nan, '3'), (2, nan, '+;2'), (1, 10, '1')])  # 並びだけ違う
    d = compare.diff_long(a, b)
    assert d['changed'] == 0


def test_行番号の無い行は多重集合で比べる():
    nan = math.nan
    a = _long([(2, nan, '+'), (2, nan, '+')])
    b = _long([(2, nan, '+')])
    d = compare.diff_long(a, b)
    assert d['changed'] == 1 and d['by_plot'] == {2: 1}


def test_地点ごとに数える():
    """直しが狙いの地点 (左端の列なら地点 1) に閉じているかを見る"""
    a = _long([(1, 10, '1'), (1, 11, '1'), (3, 10, '2')])
    b = _long([(1, 10, '+'), (1, 11, '+'), (3, 10, '2')])
    assert compare.diff_long(a, b)['by_plot'] == {1: 2}


# --- 箱の境にインクが乗るセル ----------------------------------------------

def _cells(boxes):
    return pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])


def test_境に字が乗るセルを数える():
    d = np.zeros((100, 100), dtype=bool)
    d[10:20, 30:32] = True                       # x 30 に縦の線
    cells = _cells([(30, 0, 50, 40), (60, 0, 80, 40)])
    assert compare.boundary_ink(d, cells) == 1


def test_左だけを見られる():
    d = np.zeros((100, 100), dtype=bool)
    d[10:20, 50] = True                          # 右の境 (x2=50) にだけ字
    cells = _cells([(30, 0, 50, 40)])
    assert compare.boundary_ink(d, cells) == 1
    assert compare.boundary_ink(d, cells, sides=('x1',)) == 0


def test_上下に分けて数える():
    """一様でない歪みが見える (21_p4 は上 23% → 21% なのに下 7% → 44%)"""
    d = np.zeros((200, 100), dtype=bool)
    d[150:190, 30] = True                        # 下の方にだけ字
    cells = _cells([(30, 0, 50, 50), (30, 50, 50, 100),
                    (30, 100, 50, 150), (30, 150, 50, 200)])
    (top, nt), (bot, nb) = compare.boundary_ink_halves(d, cells)
    assert (top, nt) == (0, 2)
    assert (bot, nb) == (1, 2)


def test_空の表でも落ちない():
    d = np.zeros((10, 10), dtype=bool)
    assert compare.boundary_ink_halves(d, _cells([])) == ((0, 0), (0, 0))
