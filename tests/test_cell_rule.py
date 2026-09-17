"""組成のセルの箱から縦罫線を外す (cell_rule)

2026-09-17 に足した．基準の通しで，`1` を含む読みの 931 セルの箱に
縦罫線が入っていた (罫線と `・` の組を `1` と読む)．左端の押し出しは
一律の余白で値の頭を切るため歯止めに拒まれ，内側の罫線には届かない．
そこで**セルごとに**罫線を見つけ，罫線と値のあいだの隙間に境を置く．
"""
import numpy as np
import pandas as pd

from comptea import cell_rule

H, W = 300, 400
ROW = 30


def _page():
    return np.zeros((H, W), dtype=bool)


def _vrule(d, x, width=3, y1=0, y2=H):
    d[y1:y2, x:x + width] = True


def _dot(d, x, y, r=2):
    d[y - r:y + r, x - r:x + r] = True


def _one(d, x, y1, y2):
    """字の `1`: 行の中で切れる縦線"""
    d[y1:y2, x:x + 2] = True


def test_箱の左端に入った罫線を外す():
    d = _page()
    _vrule(d, 110)                      # 箱 (100, 120-150) の左端の内側
    _dot(d, 140, 135)
    x1, x2 = cell_rule.trim_box(d, 100, 120, 180, 150)
    assert x1 > 112                     # 罫線の右
    assert x1 < 138                     # 点の手前
    assert x2 == 180


def test_罫線から離れた値には余白を空ける():
    d = _page()
    _vrule(d, 110)
    _dot(d, 150, 135)
    x1, _ = cell_rule.trim_box(d, 100, 120, 180, 150)
    assert x1 == 112 + cell_rule.GAP    # 罫線の右端 112 から GAP


def test_罫線のすぐ右の値の頭は切らない():
    """二重罫線のすぐ右から値が始まる表 (23_p3)．一律の余白は `+` の腕を切った"""
    d = _page()
    _vrule(d, 110)
    d[130:140, 117:125] = True          # 罫線の右 5 px から始まる字
    x1, _ = cell_rule.trim_box(d, 100, 120, 180, 150)
    assert x1 <= 117
    assert x1 > 112


def test_箱の右端に入った罫線を外す():
    d = _page()
    _vrule(d, 170)
    _dot(d, 130, 135)
    x1, x2 = cell_rule.trim_box(d, 100, 120, 180, 150)
    assert x1 == 100
    assert 132 < x2 < 170


def test_字の1は罫線とみなさない():
    """罫線は行をまたいで続くが，字は行の中で切れる"""
    d = _page()
    _one(d, 108, 124, 147)              # 箱の高さの 77% の縦線 (上下の外には無い)
    x1, x2 = cell_rule.trim_box(d, 100, 120, 180, 150)
    assert (x1, x2) == (100, 180)


def test_箱の中ほどの縦線には触らない():
    d = _page()
    _vrule(d, 138)                      # 幅 80 の真ん中
    assert cell_rule.trim_box(d, 100, 120, 180, 150) == (100, 180)


def test_罫線の縁のぎざぎざを字と取り違えない():
    d = _page()
    _vrule(d, 110)
    for y in range(0, H, 6):
        d[y, 113] = True                # 罫線の右の縁の突起
    _dot(d, 150, 135)
    x1, _ = cell_rule.trim_box(d, 100, 120, 180, 150)
    assert x1 >= 112 + 3


def test_細くなりすぎるなら動かさない():
    d = _page()
    _vrule(d, 125)
    _vrule(d, 155)
    assert cell_rule.trim_box(d, 100, 120, 180, 150, side=0.4, keep_w=0.8) == (100, 180)


def test_組成のセルだけを動かす():
    d = _page()
    _vrule(d, 110)
    df = pd.DataFrame([
        {'obj_name': 'comp', 'x1': 100.0, 'y1': 120.0, 'x2': 180.0, 'y2': 150.0},
        {'obj_name': 'layer', 'x1': 100.0, 'y1': 150.0, 'x2': 180.0, 'y2': 180.0},
    ])
    out, n = cell_rule.trim_rules(df, d)
    assert n == 1
    assert out.at[0, 'x1'] > 100
    assert out.at[1, 'x1'] == 100
    assert df.at[0, 'x1'] == 100        # 元は変えない


def test_罫線が無ければ何もしない():
    d = _page()
    _dot(d, 140, 135)
    df = pd.DataFrame([{'obj_name': 'comp', 'x1': 100.0, 'y1': 120.0,
                        'x2': 180.0, 'y2': 150.0}])
    out, n = cell_rule.trim_rules(df, d)
    assert n == 0
    assert out.equals(df)
