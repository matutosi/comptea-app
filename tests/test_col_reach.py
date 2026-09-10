"""組成部の右端の外にある地点の列を足す (col_reach.py)

`col` の検出が右端まで届かない表 (kinki_006・08_p2・04_p2・15_p3．2026-09-10 ユーザ指摘)．
**本体にも表頭にも字がある**帯だけを足す．常在度の字のはみ出し (表頭が空) は足さない．
"""
import numpy as np

from comptea import col_reach


PITCH = 60
Y_EDGES = np.array([200.0 + 30 * i for i in range(21)])      # 20 行 (行の高さ 30)
Y_HEAD = (60.0, 190.0)                                        # 表頭の項目行
X_EDGES = np.array([400.0 + PITCH * i for i in range(6)])     # 5 列 (400〜700)


def _dots(dark, c, rows=range(20)):
    """列 c に「・」を置く"""
    x = 400 + PITCH * c
    for i in rows:
        y = 200 + 30 * i + 12
        dark[y:y + 6, x + 27:x + 33] = True


def _head(dark, c):
    """列 c の表頭に値を置く"""
    x = 400 + PITCH * c
    for k in range(4):
        y = 70 + 30 * k
        dark[y:y + 16, x + 20:x + 40] = True


def _dark(n_cols=8, head=True, width=1200):
    dark = np.zeros((900, width), dtype=bool)
    for c in range(n_cols):
        _dots(dark, c)
        if head:
            _head(dark, c)
    return dark


def _reach(dark, **kw):
    return col_reach.reach_right(dark, X_EDGES, Y_EDGES, **kw)


def test_本体にも表頭にも字のある列を足す():
    xs, n = _reach(_dark(n_cols=8), y_head=Y_HEAD)
    assert n == 3 and len(xs) == 9
    assert abs(xs[-1] - (400 + PITCH * 8)) <= PITCH * 0.3
    assert (np.diff(xs) > PITCH * 0.7).all()


def test_右の余白には足さない():
    xs, n = _reach(_dark(n_cols=5), y_head=Y_HEAD)
    assert n == 0 and len(xs) == 6


def test_表頭が空の列は足さない():
    # 常在度の列: 本体に字はあるが表頭は空 (063-1・04_p1・10_p2 の型)
    dark = _dark(n_cols=5)
    _dots(dark, 5)
    xs, n = _reach(dark, y_head=Y_HEAD)
    assert n == 0


def test_次の段の手前で止まる():
    xs, n = _reach(_dark(n_cols=8), y_head=Y_HEAD, x_max=700 + PITCH + 10)
    assert n == 1


def test_画像の端で切れる列は足さない():
    # 6 列目が 20 px しか画像に入らない
    xs, n = _reach(_dark(n_cols=8, width=720), y_head=Y_HEAD)
    assert n == 0
    # 8 割入っていれば足す (17_p1 は右端の列が紙の端で少し切れている)
    xs, n = _reach(_dark(n_cols=6, width=760), y_head=Y_HEAD)
    assert n == 1


def test_表頭が無い表では字のある行の割合で見る():
    xs, n = _reach(_dark(n_cols=6, head=False), y_head=None)
    assert n == 1
    # 2 行にしか字が無いのは，隣の字のはみ出し
    dark = _dark(n_cols=5, head=False)
    _dots(dark, 5, rows=(3, 7))
    xs, n = _reach(dark, y_head=None)
    assert n == 0


def test_列が2本未満なら間隔が出ないので触らない():
    xs, n = col_reach.reach_right(_dark(n_cols=8), X_EDGES[:2], Y_EDGES, y_head=Y_HEAD)
    assert n == 0 and len(xs) == 2


def test_左端の外の列を足す():
    """`reach_left`: 本体にも表頭にも字のある帯を左へ足す"""
    dark = np.zeros((300, 400), dtype=bool)
    y_edges = np.array([100.0, 140.0, 180.0, 220.0])
    y_head = (40.0, 80.0)
    x_edges = np.array([200.0, 250.0, 300.0, 350.0])
    for x1, x2 in ((160, 190), (210, 240), (260, 290), (310, 340)):
        dark[45:75, x1:x2] = True          # 表頭
        for a in (105, 145, 185):
            dark[a:a + 30, x1:x2] = True   # 本体
    out, n = col_reach.reach_left(dark, x_edges, y_edges, y_head=y_head, x_min=120.0)
    assert n == 1
    assert out[0] < 200.0 and len(out) == len(x_edges) + 1


def test_表頭が空の左の列は足さない():
    """階層の列 (本体に字・表頭は空) は巻き込まない"""
    dark = np.zeros((300, 400), dtype=bool)
    y_edges = np.array([100.0, 140.0, 180.0, 220.0])
    y_head = (40.0, 80.0)
    x_edges = np.array([200.0, 250.0, 300.0, 350.0])
    for x1, x2 in ((210, 240), (260, 290), (310, 340)):
        dark[45:75, x1:x2] = True
        for a in (105, 145, 185):
            dark[a:a + 30, x1:x2] = True
    for a in (105, 145, 185):              # 左の帯は本体だけ
        dark[a:a + 30, 160:190] = True
    out, n = col_reach.reach_left(dark, x_edges, y_edges, y_head=y_head, x_min=120.0)
    assert n == 0
    assert len(out) == len(x_edges)


def test_表頭が分からなければ左へ伸ばさない():
    """階層の列と地点の列を見分ける手掛かりが無いので，何もしない (10_p2 型)"""
    dark = np.zeros((300, 400), dtype=bool)
    y_edges = np.array([100.0, 140.0, 180.0, 220.0])
    x_edges = np.array([200.0, 250.0, 300.0, 350.0])
    for x1, x2 in ((160, 190), (210, 240), (260, 290), (310, 340)):
        for a in (105, 145, 185):
            dark[a:a + 30, x1:x2] = True
    out, n = col_reach.reach_left(dark, x_edges, y_edges, y_head=None, x_min=120.0)
    assert n == 0
    assert len(out) == len(x_edges)
