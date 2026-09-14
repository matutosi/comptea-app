"""段の格子を組む段 (locate.py の `_locate_block` から切り出した 2 段)

2026-09-14 に 201 行の `_locate_block` を分けたときに足した．
それまで**この段には的が 1 つも無く**，格子の SHA を突き合わせる形でしか
守れていなかった．

画像を渡さない (`img=None`) ときの筋道を確かめる．画像が要る判定
(境を谷へずらす・階層を隙間から補う) は実データの的が別にある．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import locate


def _range(x1, x2):
    """`locate_x_range` が返す形 (x1・x2 を 1 行持つ表)"""
    return pd.DataFrame([{'x1': float(x1), 'x2': float(x2)}])


Y_EDGES = np.array([100.0, 130.0, 160.0, 190.0])
X_EDGES = np.array([500.0, 560.0, 620.0])
CLASSES = ('species_col', 'sname', 'layer')


# --- 学名・和名・階層の列 --------------------------------------------------

def test_範囲のある列だけセルにする():
    items, layer = locate._name_layer_items(
        pd.DataFrame(), None, (_range(100, 300), _range(300, 460), None),
        X_EDGES, Y_EDGES, None, CLASSES, [])
    assert len(items) == 2                       # 学名と和名．階層は無い
    assert layer is None
    assert set(items[0]['obj_name']) == {'species_col'}
    assert len(items[0]) == 3                    # 行は 3 つ


def test_階層が無いときは無いと言い切る():
    """草本群落などでは普通のこと．**黙って落とさず warning に残す**"""
    warns = []
    _items, _layer = locate._name_layer_items(
        pd.DataFrame(), None, (_range(100, 300), None, None),
        X_EDGES, Y_EDGES, None, CLASSES, warns)
    assert any('階層の列は**無い**とみなした' in w for w in warns)


def test_検出の無い列は知らせて飛ばす():
    warns = []
    items, _layer = locate._name_layer_items(
        pd.DataFrame(), None, (None, _range(300, 460), None),
        X_EDGES, Y_EDGES, None, CLASSES, warns)
    assert len(items) == 1
    assert any("'species_col' が1件も検出されなかった" in w for w in warns)


def test_検出された階層はそのまま使う():
    items, layer = locate._name_layer_items(
        pd.DataFrame(), None, (_range(100, 300), _range(300, 460),
                               _range(460, 500)),
        X_EDGES, Y_EDGES, None, CLASSES, [])
    assert len(items) == 3
    assert layer is not None
    lay = items[2]
    assert set(lay['obj_name']) == {'layer'}
    assert float(lay['x1'].iloc[0]) == 460.0     # 列は分けない (端から端まで)
    assert float(lay['x2'].iloc[0]) == 500.0


def test_行の印はそのまま渡る():
    """`y_notes` は**行ごとの印のリスト**．`note` の列に ';' 区切りで入る"""
    notes = [['interpolated'], [], ['snapped', 'on_text']]
    items, _layer = locate._name_layer_items(
        pd.DataFrame(), None, (_range(100, 300), None, None),
        X_EDGES, Y_EDGES, notes, CLASSES, [])
    assert list(items[0]['note']) == ['interpolated', '', 'snapped;on_text']


# --- 境を谷へずらす段 ------------------------------------------------------

def test_画像が無ければ何もしない():
    """`img` は**横の範囲が決められないとき**も None になる (呼ぶ側で落とす)"""
    y_int = np.zeros(3, dtype=bool)
    x_int = np.zeros(2, dtype=bool)
    got = locate._snap_block_edges(
        pd.DataFrame(), 'a.png', None, Y_EDGES, y_int, X_EDGES, x_int, 3,
        (_range(100, 300), None, None), None, 0.3, [])
    assert got[0] is Y_EDGES and got[2] is X_EDGES
    assert got[4] == 3
    assert got[5] is None and got[8] is None     # ずらしの印は作らない


def test_画像が無ければ知らせもしない():
    warns = []
    locate._snap_block_edges(
        pd.DataFrame(), 'a.png', None, Y_EDGES, np.zeros(3, dtype=bool),
        X_EDGES, np.zeros(2, dtype=bool), 3,
        (None, None, None), None, 0.3, warns)
    assert warns == []
