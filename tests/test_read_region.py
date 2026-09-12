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


# --- 複数の読み手を重ねる ------------------------------------------------
#
# **読み手ごとに落とすセルが違う** (2026-09-12 に真値で確かめた)．
# kinki_047 の和名は EasyOCR 43・yomitoku 42 だが，**合わせて 45**．

def test_先の読み手が読めたセルはそのまま():
    got = read_region.union([{1: 'あ', 2: 'い'}, {1: 'ア', 3: 'ウ'}])
    assert got[1] == 'あ' and got[2] == 'い'


def test_先が読めなかったセルを後が埋める():
    got = read_region.union([{1: 'あ'}, {2: 'い'}, {3: 'う'}])
    assert got == {1: 'あ', 2: 'い', 3: 'う'}


def test_空の読みは埋めたことにしない():
    got = read_region.union([{1: '  '}, {1: 'あ'}])
    assert got == {1: 'あ'}


def test_どこから来たかを返せる():
    got, src = read_region.union([{1: 'あ'}, {2: 'い'}], names=['easy', 'yomi'],
                                 with_source=True)
    assert got == {1: 'あ', 2: 'い'}
    assert src == {1: 'easy', 2: 'yomi'}


def test_食い違いを数えられる():
    """同じセルを両方が読んで中身が違うものは，目視に回す手がかりになる"""
    assert read_region.disagree([{1: 'あ', 2: 'い'}, {1: 'ア', 2: 'い'}]) == {1}


# --- 読みを「質で選ぶ」--------------------------------------------------
#
# `union` は「先の読み手が何か読めていれば採る」ので，**読めてはいるが間違って
# いる**セルを後の読み手が直せない (実測: 真値との一致が EasyOCR 単独と同じ
# 13・11 のまま．yomitoku 単独は 24・31)．**質で選ぶ**必要がある．

def _ok(text):
    """「あ」で始まるものを「通る読み」とみなす偽の判定"""
    return (text or '').startswith('あ')


def test_辞書に当たる方を採る():
    got = read_region.best([{1: 'ア'}, {1: 'あい'}], ok=_ok)
    assert got == {1: 'あい'}


def test_どちらも通らなければ先を採る():
    got = read_region.best([{1: 'ア'}, {1: 'イ'}], ok=_ok)
    assert got == {1: 'ア'}


def test_先が通れば後は見ない():
    got = read_region.best([{1: 'あ'}, {1: 'あいう'}], ok=_ok)
    assert got == {1: 'あ'}


def test_先が読めなければ後で埋める():
    got = read_region.best([{}, {1: 'イ'}], ok=_ok)
    assert got == {1: 'イ'}


def test_どこから採ったかを返せる():
    got, src = read_region.best([{1: 'ア', 2: 'あ'}, {1: 'あい'}], ok=_ok,
                                names=['easy', 'yomi'], with_source=True)
    assert got == {1: 'あい', 2: 'あ'}
    assert src == {1: 'yomi', 2: 'easy'}


def test_判定が無ければ先勝ち():
    got = read_region.best([{1: 'ア'}, {1: 'あい'}])
    assert got == {1: 'ア'}


# --- クラスごとに，信頼する読み手の順を変える ----------------------------
#
# **クラスごとに得意な読み手が違う** (2026-09-12 に真値で確認)．
#   学名 … EasyOCR 11・13 に対し yomitoku 31・24
#   和名 … EasyOCR 42・23 に対し yomitoku 41・27 (ほぼ互角，EasyOCR がやや上)

def test_学名は_yomitoku_を先に見る():
    assert read_region.order('sname')[0] == 'yomi'


def test_和名は_easyocr_を先に見る():
    assert read_region.order('species_col')[0] == 'easy'


def test_知らないクラスは_easyocr_を先に見る():
    assert read_region.order('comp')[0] == 'easy'


def test_順にそって読みを並べ替える():
    maps = {'easy': {1: 'ア'}, 'yomi': {1: 'あ'}}
    got = read_region.pick(maps, 'sname', ok=lambda t: True)
    assert got == {1: 'あ'}                      # 学名は yomitoku が先
    got = read_region.pick(maps, 'species_col', ok=lambda t: True)
    assert got == {1: 'ア'}                      # 和名は EasyOCR が先


def test_無い読み手は飛ばす():
    maps = {'easy': {1: 'ア'}}
    assert read_region.pick(maps, 'sname', ok=lambda t: True) == {1: 'ア'}
