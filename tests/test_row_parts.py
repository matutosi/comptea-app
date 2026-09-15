"""行の高さと行の種類を決める小さな段 (row_heights.py・row_kinds.py)

2026-09-15 に足した．合成した並び・画像で確かめられるものを埋める．
"""
import numpy as np

from comptea import row_heights, row_kinds


def _prof(n=400, pitch=20, ink=6, start=10):
    """`pitch` ごとに字のある並び (黒画素の投影)"""
    p = np.zeros(n)
    for y in range(start, n - ink, pitch):
        p[y:y + ink] = 100.0
    return p


# --- 自己相関で刻みを見る --------------------------------------------------

def test_刻みが合えば相関が高い():
    p = _prof()
    assert row_heights.autocorr(p, 20) > row_heights.autocorr(p, 13)


def test_平らな並びでは相関が立たない():
    assert row_heights.autocorr(np.ones(200), 20) <= 0.5


# --- 字のある区間を数える --------------------------------------------------

def test_字の区間を数える():
    """印字の行の数の目安 (20 px ごとに 1 区間)"""
    n = row_heights.text_runs(_prof(n=200, pitch=20))
    assert 8 <= n <= 10


def test_字が無ければ0():
    assert row_heights.text_runs(np.zeros(200)) == 0


# --- 行の高さで並べ直す ----------------------------------------------------

def test_中央値の刻みで並べ直す():
    p = _prof(n=300, pitch=20)
    got = row_heights.lattice_edges([10.0, 290.0], p, 20.0)
    assert len(got) >= 13                       # 280 px / 20 px
    d = np.diff(got)
    assert abs(float(np.median(d)) - 20.0) <= 2


def test_刻みがゼロなら触らない():
    got = row_heights.lattice_edges([10.0, 100.0], _prof(), 0.0)
    assert list(got) == [10.0, 100.0]


# --- まとめて動かす量 ------------------------------------------------------

def test_境を字の谷へ動かす量を返す():
    """境が字の上にあるとき，まとめて動かすと黒画素の和が減る"""
    p = _prof(n=300, pitch=20, ink=6, start=10)
    inner = [12.0, 32.0, 52.0]                  # 字の上 (10-16 など)
    shift = row_heights.best_shift(inner, p, 20.0)
    assert shift != 0


def test_すでに谷なら動かさない():
    p = _prof(n=300, pitch=20, ink=6, start=10)
    inner = [20.0, 40.0, 60.0]                  # 字と字の間
    assert row_heights.best_shift(inner, p, 20.0) == 0


# --- 下線 ------------------------------------------------------------------

def test_下線を見つける():
    dark = np.zeros((100, 400), dtype=bool)
    dark[80:83, 20:380] = True                  # 帯の下寄りの長い横線
    assert row_kinds.has_underline(dark, 0, 400, 0, 100, 30.0)


def test_字だけなら下線ではない():
    dark = np.zeros((100, 400), dtype=bool)
    dark[40:60, 20:60] = True
    assert not row_kinds.has_underline(dark, 0, 400, 0, 100, 30.0)


# --- 種名の側から本体へまたぐ塊 --------------------------------------------

def test_またぐ塊の幅を返す():
    """凡例は 1 つの塊が種名の側から本体の側へまたぐ"""
    dark = np.zeros((60, 600), dtype=bool)
    dark[20:40, 100:400] = True                 # 境 300 をまたぐ
    got = row_kinds.crossing_run(dark, 0, 600, 0, 60, 300)
    assert got >= 200


def test_またがなければ0():
    dark = np.zeros((60, 600), dtype=bool)
    dark[20:40, 100:280] = True                 # 境の手前で終わる
    assert row_kinds.crossing_run(dark, 0, 600, 0, 60, 300) == 0


# --- 読み手が無いとき ------------------------------------------------------

def test_読み手が無ければ読まない(monkeypatch):
    monkeypatch.setattr(row_kinds, 'reader_or_none', lambda reader=None: None)
    assert row_kinds.read_texts(None, (0, 0, 10, 10)) == []
    assert row_kinds.read_kind(None, (0, 0, 10, 10)) == 'other'


def test_箱が小さすぎれば読まない():
    assert row_kinds.read_texts(None, (0, 0, 2, 2)) == []
    assert row_kinds.read_kind(None, (0, 0, 2, 2)) == 'other'
