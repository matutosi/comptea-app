"""格子の幾何(等間隔の当てはめ・境が字を割る回数)

行と列は**検出ではなくアルゴリズム**で決める(2026-09-02 の方針)．
その中核の 2 つを，合成した並びで確かめる．
"""
import numpy as np
import pytest

from comptea import col_edges


# --- 等間隔の格子を当てる -----------------------------------------------

def test_隙間から等間隔の格子を組み立てる():
    """地点の列は等間隔に組まれている

    隙間はそれを写したものだが，地点の中で割れた偽の隙間が混ざり，
    値の詰まった所では隙間が出ない．そこで**格子を当てはめる**．
    """
    edges = col_edges.column_edges_from_gaps([100, 200, 300, 400], 0, 500)
    assert edges is not None
    assert len(edges) == 6
    assert np.allclose(np.diff(edges), 100)


def test_隙間が抜けていても間隔は保たれる():
    """値が詰まって隙間の出ない所があっても，格子は同じ刻みで通る"""
    edges = col_edges.column_edges_from_gaps([100, 200, 400], 0, 500)
    assert edges is not None and len(edges) == 6


def test_隙間が少なすぎれば当てない():
    assert col_edges.column_edges_from_gaps([100, 200], 0, 500) is None


# --- 境が字を割る回数 ---------------------------------------------------

def test_字を割る回数を数える():
    """**濃さで重みを付けない**のが要(2026-09-05)

    組成部は大半が `・` なので，濃さで測ると `・` の位置だけで決まり，
    数の少ない値の行が無視される(17_p1 は線上の黒画素が中央値の 0.56 倍と
    良い値なのに，`2・3` の `3` が境で切れていた)．
    """
    dark = np.zeros((30, 100), dtype=bool)
    dark[5:10, 20:40] = True                  # 1 行目の字のかたまり
    dark[15:20, 20:40] = True                 # 2 行目にも同じ位置
    cross = col_edges.crossing_counts(dark, [(0, 12), (12, 30)], 100)
    assert cross[30] == 2                     # かたまりの内側は 2 回割る
    assert cross[50] == 0                     # 字の無い所は 0


def test_かたまりの外は数えない():
    dark = np.zeros((20, 60), dtype=bool)
    dark[5:10, 10:20] = True
    cross = col_edges.crossing_counts(dark, [(0, 20)], 60)
    assert cross[10] == 0 and cross[20] == 0  # 端は割らない
    assert cross[15] == 1
