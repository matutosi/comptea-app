"""行を「単位の連なり」として追う段 (row_track.py)

2026-09-15 に足した．単位の取り出し・列ごとのずれ・行ごとの追従が，
的で呼ばれていなかった．

**行の識別子は全列で共有し，y は列ごとに持つ**というのがこの段の考え方
(タイプ打ちでは文字と「・」の高さが 10 px 違い，紙面も傾く)．
"""
import numpy as np
import pytest

from comptea import row_track


def _band(h=300, w=60, units=()):
    """指定した y に字の塊を置いた帯 (True が黒)"""
    d = np.zeros((h, w), dtype=bool)
    for y1, y2 in units:
        d[y1:y2, 10:w - 10] = True
    return d


# --- 単位を取り出す --------------------------------------------------------

def test_字の塊を単位にする():
    band = _band(units=[(20, 40), (80, 100), (140, 160)])
    got = row_track.units_in_band(band, 0)
    assert len(got) == 3
    cys = [c for c, _h, _ink in got]
    assert 25 <= cys[0] <= 35


def test_画像の座標で返す():
    """`y0` を足した座標で返る (帯は紙面の一部を切ったもの)"""
    band = _band(units=[(20, 40)])
    got = row_track.units_in_band(band, 1000)
    assert 1020 <= got[0][0] <= 1035


def test_小さすぎる塊は数えない():
    """点や濁点の切れ端を単位にしない"""
    band = np.zeros((300, 60), dtype=bool)
    band[50:52, 30:32] = True
    assert row_track.units_in_band(band, 0, min_area=20) == []


def test_字が無ければ空():
    assert row_track.units_in_band(np.zeros((100, 40), dtype=bool), 0) == []


# --- 境が単位を割る数 ------------------------------------------------------

def test_単位の内側を通る境を数える():
    # `spans` は (上, 下, 面積) の 3 つ組
    spans = [(20.0, 40.0, 200), (80.0, 100.0, 200)]
    assert row_track.split_count(spans, [30.0]) == 1      # 1 つ目を割る
    assert row_track.split_count(spans, [60.0]) == 0      # 字の間
    assert row_track.split_count(spans, [30.0, 90.0]) == 2


def test_境が無ければ割らない():
    assert row_track.split_count([(20.0, 40.0, 200)], []) == 0


def test_単位が無ければ割らない():
    assert row_track.split_count([], [30.0]) == 0


# --- 列ごとのずれ ----------------------------------------------------------

def test_列のずれを中央値で出す():
    """単位が行の中心より 10 px 下にあれば，ずれは +10"""
    centers = [100.0, 150.0, 200.0]
    cys = [110.0, 160.0, 210.0]
    dy, iqr, n = row_track.column_offset(cys, centers, 50.0)
    assert 9 <= dy <= 11
    assert iqr < 5                                    # そろっている
    assert n == 3                                     # 測った単位の数


def test_ばらついていれば四分位範囲が大きい():
    centers = [100.0, 150.0, 200.0]
    cys = [110.0, 140.0, 215.0]
    _dy, iqr, _n = row_track.column_offset(cys, centers, 50.0)
    assert iqr > 5


def test_単位が無ければずれは出ない():
    dy, iqr, n = row_track.column_offset([], [100.0, 150.0], 50.0)
    assert (dy, iqr, n) == (0.0, 0.0, 0)


# --- 行ごとの追従 ----------------------------------------------------------

def test_行ごとのずれを滑らかにする():
    """前後 k 行の中央値で均す (1 行だけ跳ねた値に引きずられない)"""
    centers = [100.0, 150.0, 200.0, 250.0, 300.0]
    cys = [105.0, 155.0, 175.0, 255.0, 305.0]     # 3 行目だけ外れ
    got = row_track.row_offsets(cys, centers, 50.0, k=2)
    assert len(got) == len(centers)
    assert abs(got[2] - 5.0) < 5                  # 外れに引きずられない
