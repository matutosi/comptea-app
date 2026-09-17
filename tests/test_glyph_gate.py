"""読み直すかを形で決める (ink.glyph_gate)

2026-09-17 に足した．読めなかったセルを読み直すかを黒画素の割合で決めていた
ときは，はっきりした `+` が数百セル落ちていた (表全体の閾値でも列ごとでも)．
`+` と `・` は量ではなく大きさと形が違うので，セルの中だけで決める．
"""
import numpy as np

from comptea import ink

H, W = 30, 80


def _cell():
    return np.zeros((H, W), dtype=bool)


def _dot(c, x=40, y=15, r=2):
    c[y - r:y + r, x - r:x + r] = True


def _plus(c, x=40, y=15, arm=6):
    c[y - 1:y + 1, x - arm:x + arm] = True
    c[y - arm:y + arm, x - 1:x + 1] = True


def test_点だけなら読み直さない():
    c = _cell()
    _dot(c)
    assert not ink.glyph_gate(c)


def test_十字は読み直す():
    c = _cell()
    _plus(c)
    assert ink.glyph_gate(c)


def test_空なら読み直さない():
    assert not ink.glyph_gate(_cell())


def test_塊が2つなら読み直す():
    """`1・2` の `・` 以外の字が小さくても，並んでいれば値"""
    c = _cell()
    _dot(c, x=20)
    _dot(c, x=50)
    assert ink.glyph_gate(c)


def test_縦の罫線と点は読み直さない():
    """罫線と `・` の組を `1` と読む誤りのもと"""
    c = _cell()
    c[:, 3:6] = True
    _dot(c)
    assert not ink.glyph_gate(c)


def test_横線と点は読み直さない():
    c = _cell()
    c[28:30, :] = True
    _dot(c)
    assert not ink.glyph_gate(c)


def test_汚れは数えない():
    c = _cell()
    _dot(c)
    c[5, 70] = True
    assert not ink.glyph_gate(c)


def test_高さはセルの高さに対する割合で見る():
    """同じ大きさの十字でも，行の高いセルでは `・` の側に入る"""
    c = np.zeros((60, W), dtype=bool)
    _plus(c, y=30, arm=6)                 # 12 px / 60 px = 0.2
    assert not ink.glyph_gate(c)
    assert ink.glyph_gate(c, big=0.15)
