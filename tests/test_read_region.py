"""領域を 1 回読んで，読みをセルに割り当てる (read_region.py)

セルを 1 つずつ読む代わりに，**表頭のような領域をまとめて読む**．
新しい読み手 (ndlocr-lite・yomitoku) は位置つきで返すので，読みの中心が
どのセルに入るかで割り当てられる．呼び出し回数が桁で減る．
"""
import pandas as pd
import pytest

from comptea import read_region


def _cells(rows):
    """(cell_id, x1, y1, x2, y2) の並びから格子を作る"""
    return pd.DataFrame(
        [{'cell_id': c, 'x1': a, 'y1': b, 'x2': x, 'y2': y,
          'obj_name': 'header_item'} for c, a, b, x, y in rows])


def test_読みをセルへ中心で割り当てる():
    cells = _cells([(1, 0, 0, 100, 50), (2, 0, 50, 100, 100)])
    found = [((10, 10, 60, 40), '通し番号'), ((10, 60, 60, 90), '調査番号')]
    got = read_region.assign(cells, found)
    assert got == {1: '通し番号', 2: '調査番号'}


def test_同じセルの読みは左から順につなぐ():
    cells = _cells([(1, 0, 0, 200, 50)])
    found = [((120, 10, 180, 40), '(県名)'), ((10, 10, 90, 40), '調査地')]
    assert read_region.assign(cells, found) == {1: '調査地 (県名)'}


def test_セルの外の読みは捨てる():
    cells = _cells([(1, 0, 0, 100, 50)])
    found = [((10, 10, 60, 40), '通し番号'), ((300, 300, 360, 340), '本文')]
    assert read_region.assign(cells, found) == {1: '通し番号'}


def test_読みが無いセルは入らない():
    cells = _cells([(1, 0, 0, 100, 50), (2, 0, 50, 100, 100)])
    found = [((10, 10, 60, 40), '通し番号')]
    assert read_region.assign(cells, found) == {1: '通し番号'}


def test_空の読みは数えない():
    cells = _cells([(1, 0, 0, 100, 50)])
    found = [((10, 10, 60, 40), '   '), ((20, 20, 60, 40), '通し番号')]
    assert read_region.assign(cells, found) == {1: '通し番号'}


def test_領域の箱はセルを囲む():
    cells = _cells([(1, 10, 20, 100, 50), (2, 0, 50, 120, 100)])
    assert read_region.region_box(cells, pad=5) == (0, 15, 125, 105)


def test_セルが無ければ領域は_None():
    assert read_region.region_box(_cells([]).assign()) is None


class FakeReader:
    """読み手の形 (`read_boxes(img, box)` → [(箱, 文字列)]) を満たす偽物"""

    def __init__(self, found):
        self.found = found
        self.calls = 0

    def read_boxes(self, img, box):
        self.calls += 1
        return self.found


def test_領域を1回だけ読む():
    """セルの数だけ読まない (呼び出し回数が減ることが狙い)"""
    cells = _cells([(1, 0, 0, 100, 50), (2, 0, 50, 100, 100),
                    (3, 0, 100, 100, 150)])
    r = FakeReader([((10, 10, 60, 40), 'あ'), ((10, 60, 60, 90), 'い'),
                    ((10, 110, 60, 140), 'う')])
    got = read_region.read_cells(None, cells, r)
    assert r.calls == 1
    assert got == {1: 'あ', 2: 'い', 3: 'う'}


def test_読み手が落ちても空を返す():
    class Bad:
        def read_boxes(self, img, box):
            raise RuntimeError('読めない')

    cells = _cells([(1, 0, 0, 100, 50)])
    assert read_region.read_cells(None, cells, Bad()) == {}


# --- 1 行の読みを，セルの境で切り分ける ----------------------------------
#
# NDLOCR-Lite は**行 (テキストライン) 単位**で返すので，表頭の値のように
# 小さな値が格子状に並ぶ所では 1 行にまとめられる．中心だけで割り当てると
# 1 セルにしか入らない (実測: 3883 セル中 904 しか埋まらなかった)．

def test_1行の読みを文字の位置でセルへ分ける():
    cells = _cells([(1, 0, 0, 100, 50), (2, 100, 0, 200, 50),
                    (3, 200, 0, 300, 50)])
    found = [((0, 10, 300, 40), '135')]          # 3 セルに 1 文字ずつ
    got = read_region.assign(cells, found, split=True)
    assert got == {1: '1', 2: '3', 3: '5'}


def test_分けた結果が空のセルは入らない():
    cells = _cells([(1, 0, 0, 100, 50), (2, 100, 0, 200, 50)])
    found = [((0, 10, 100, 40), '12')]           # 左のセルだけに掛かる
    assert read_region.assign(cells, found, split=True) == {1: '12'}


def test_分けないときは中心のセルへ入る():
    cells = _cells([(1, 0, 0, 100, 50), (2, 100, 0, 200, 50)])
    found = [((0, 10, 200, 40), '12')]
    assert read_region.assign(cells, found, split=False) == {1: '12'}


def test_1セルに収まる読みは分けても変わらない():
    cells = _cells([(1, 0, 0, 200, 50)])
    found = [((10, 10, 90, 40), '調査地'), ((120, 10, 180, 40), '(県名)')]
    assert read_region.assign(cells, found, split=True) == {1: '調査地 (県名)'}
