"""境の線上の黒画素 (axes._edge_ink)

**境が画像の外にあることがある**．`dark[:, x1:x2]` は画像の幅で切られるので，
`x2 - x1` を長さとみなすと落ちる (s01115_23_p1 が
`IndexError: index 2315 is out of bounds for axis 0 with size 2315` で
格子を作れなかった．2026-09-10)．
"""
import numpy as np

from comptea.axes import _edge_ink


def _dark(w=200, h=100):
    dark = np.zeros((h, w), dtype=bool)
    for x in range(10, w, 20):
        dark[20:80, x:x + 4] = True       # 字らしい縦の帯
    return dark


def test_境が画像の外にあっても落ちない():
    dark = _dark(w=200)
    # 右端の境が画像の外 (240 > 200)
    edges = np.array([0.0, 60.0, 120.0, 240.0])
    v = _edge_ink(edges, dark, 0, 100)
    assert v is None or v >= 0


def test_内側の境が画像の外でも数えない():
    dark = _dark(w=200)
    edges = np.array([0.0, 60.0, 210.0, 240.0])     # 210 は画像の外
    assert _edge_ink(edges, dark, 0, 100) is not None


def test_ふつうの境は数える():
    dark = _dark(w=200)
    edges = np.array([0.0, 50.0, 100.0, 150.0])
    v = _edge_ink(edges, dark, 0, 100)
    assert v is not None and v >= 0


def test_黒が無ければNone():
    assert _edge_ink(np.array([0.0, 50.0, 100.0]),
                     np.zeros((100, 200), dtype=bool), 0, 100) is None


def test_狭すぎる範囲はNone():
    assert _edge_ink(np.array([0.0, 4.0, 8.0]), _dark(), 0, 100) is None
