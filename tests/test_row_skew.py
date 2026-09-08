"""傾きをセルの座標だけで直す後処理(row_skew.py)"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import row_skew


def _grid(edges, x_edges, obj_name, block=1):
    rows = []
    for c, (xa, xb) in enumerate(zip(x_edges[:-1], x_edges[1:]), 1):
        for r, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
            rows.append(dict(x1=xa, x2=xb, y1=float(ya), y2=float(yb), obj_name=obj_name,
                             note='', block=block, row=r, col=c))
    return pd.DataFrame(rows)


def _skewed_page(edges, x_edges, slope, size=(1400, 700)):
    """行の中央に「・」を置き，右へ行くほど y を slope * dx だけ下げた紙面"""
    img = Image.new('L', size, 255)
    px = img.load()
    x0 = (x_edges[0] + x_edges[-1]) / 2
    for ya, yb in zip(edges[:-1], edges[1:]):
        c = (ya + yb) / 2
        for xa, xb in zip(x_edges[:-1], x_edges[1:]):
            xc = (xa + xb) / 2
            y = int(round(c + slope * (xc - x0)))
            for x in range(int(xc) - 2, int(xc) + 3):
                for yy in range(y - 2, y + 3):
                    px[x, yy] = 0
    return img


def _off_center(img, cells):
    """各セルで，「・」の重心がセルの中心からどれだけ離れているか(px)の最大"""
    dark = row_skew.ink.binarize(img)
    worst = 0.0
    for x, y in row_skew.cell_centroids(dark, cells):
        cell = cells[(cells.x1 < x) & (cells.x2 > x)]
        if cell.empty:
            continue
        centers = (cell.y1 + cell.y2) / 2
        worst = max(worst, float((centers - y).abs().min()))   # いちばん近いセルの中心との差
    return worst


def test_傾いた紙面でセルの座標だけをずらす():
    edges = list(range(100, 100 + 34 * 12 + 1, 34))
    x_edges = list(range(200, 200 + 60 * 16 + 1, 60))        # 16 列，幅 960 px
    slope = 12 / 960                                          # 左右で 12 px (0.7°)
    img = _skewed_page(edges, x_edges, slope)
    comp = _grid(edges, x_edges, 'comp')
    sname = _grid(edges, [0, 180], 'sname')
    df = pd.concat([sname, comp], ignore_index=True)
    # 水平な格子では端の列で「・」がセルの中心から 6 px ずれる
    assert _off_center(img, comp) >= 5
    est, n_rows = row_skew.estimate_slope(row_skew.ink.binarize(img), comp)
    assert n_rows >= 10 and abs(est - slope) < slope * 0.2
    out, warns = row_skew.fix_skew(img, df)
    assert warns and '傾き' in warns[0]
    c = out[out.obj_name == 'comp']
    assert _off_center(img, c) <= 2                           # どの列でも中心に来る
    # 行番号は変わらず，同じ行の中で y が x に応じて単調に変わる
    assert c['row'].nunique() == 12
    r1 = c[c.row == 1].sort_values('x1')
    assert (np.diff(r1['y1']) >= 0).all() and r1['y1'].iloc[-1] - r1['y1'].iloc[0] >= 9
    # 種名の列も同じ傾きでずれる(組成部の中央を基準に左は上へ)
    s = out[(out.obj_name == 'sname') & (out.row == 1)]
    assert s['y1'].iloc[0] < r1['y1'].iloc[0]


def test_傾いていなければ触らない():
    edges = list(range(100, 100 + 34 * 12 + 1, 34))
    x_edges = list(range(200, 200 + 60 * 16 + 1, 60))
    img = _skewed_page(edges, x_edges, 0.0)
    df = _grid(edges, x_edges, 'comp')
    out, warns = row_skew.fix_skew(img, df)
    assert out is df and warns == []


def test_列が少ない表は測らない():
    edges = list(range(100, 100 + 34 * 12 + 1, 34))
    x_edges = [200, 260, 320, 380]                              # 3 列
    img = _skewed_page(edges, x_edges, 12 / 960)
    df = _grid(edges, x_edges, 'comp')
    out, warns = row_skew.fix_skew(img, df)
    assert out is df and warns == []
