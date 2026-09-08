"""行の高さを表の中央値にそろえる後処理(row_heights.py)"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import row_heights


def _grid(edges, x_edges, obj_name, block=1, notes=None):
    rows = []
    for c, (xa, xb) in enumerate(zip(x_edges[:-1], x_edges[1:]), 1):
        for r, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
            rows.append(dict(x1=xa, x2=xb, y1=ya, y2=yb, obj_name=obj_name,
                             note=(notes or ''), block=block, row=r, col=c))
    return pd.DataFrame(rows)


def _page(print_edges, dots=True, name_shift=0, size=(500, 800)):
    """印字の行 `print_edges` ごとに，組成部の「・」と種名の字を置いた画像"""
    img = Image.new('L', size, 255)
    px = img.load()
    for ya, yb in zip(print_edges[:-1], print_edges[1:]):
        c = (ya + yb) // 2
        if dots:
            for x in range(310, 410, 20):
                for y in range(c - 3, c + 4):
                    px[x, y] = 0
        for x in range(10, 150):                 # 種名: 16 px の字
            for y in range(c - 8 + name_shift, c + 8 + name_shift):
                px[x, y] = 0
    return img


def _rows(out, cls):
    return out[out.obj_name == cls].sort_values('row')


def test_直す所が無ければそのまま返す():
    edges = list(range(100, 100 + 34 * 8, 34))
    df = pd.concat([_grid(edges, [0, 200], 'sname'),
                    _grid(edges, [300, 340, 380], 'comp')], ignore_index=True)
    out, warns = row_heights.fix_row_heights(_page(edges), df)
    assert out is df and warns == []


def test_細い行を挟んだ格子は中央値の刻みで並べ直す():
    print_edges = list(range(100, 100 + 34 * 12 + 1, 34))
    # 4 行目のあとに 10 px の行が挟まり，以後が 10 px ずれている
    bad = print_edges[:5] + [print_edges[4] + 10] + [e + 10 for e in print_edges[5:]]
    comp = _grid(bad, [300, 340, 380, 420], 'comp')
    sname = _grid(bad, [0, 200], 'sname', notes='interpolated')
    head = pd.DataFrame([dict(x1=300, x2=340, y1=0, y2=100, obj_name='header_value',
                              note='', block=1, row=0, col=1)])
    df = pd.concat([head, sname, comp], ignore_index=True)
    out, warns = row_heights.fix_row_heights(_page(print_edges), df)
    assert any('並べ直した' in w for w in warns)
    c, s = _rows(out, 'comp'), _rows(out, 'sname')
    assert c['row'].nunique() == 12 and s['row'].nunique() == 12
    assert sorted(c['row'].unique()) == sorted(s['row'].unique())
    h = np.diff(np.sort(c['y1'].unique()))
    assert h.min() >= 34 * 0.8 and h.max() <= 34 * 1.2
    # 境は印字の行間(「・」の上を通らない)
    assert (np.abs(np.sort(c['y1'].unique()) - np.array(print_edges[:-1])) <= 4).all()
    assert (out[out.obj_name == 'header_value']['y2'] == 100).all()
    assert (c['note'] == 'row_fixed').all()
    assert out['row'].min() == 1 and out['row'].max() == 13


def test_許容の中の揺れは触らない():
    edges = [100, 134, 170, 201, 236, 272, 304, 340, 374]     # ±2 割の揺れ
    df = pd.concat([_grid(edges, [0, 200], 'sname'),
                    _grid(edges, [300, 340, 380], 'comp')], ignore_index=True)
    out, warns = row_heights.fix_row_heights(_page(edges), df)
    assert not any('並べ直した' in w for w in warns)


def test_全体が字の上を通る位相はまとめて動かす():
    prof = np.full(400, 1.0)
    for c in range(17, 400, 34):
        prof[c - 8:c + 9] = 100.0
    edges = [0.0] + [float(y) for y in range(12, 340, 34)]
    out, shift = row_heights.rephase_edges(edges, prof, 34)
    assert shift != 0 and abs(shift) >= 8
    inner = np.array(out[1:-1]).astype(int)
    assert (prof[inner] <= 1.0).all()          # 境はすべて谷の上


def test_位相が合っていれば動かさない():
    prof = np.full(400, 1.0)
    for c in range(17, 400, 34):
        prof[c - 4:c + 5] = 100.0
    edges = [float(y) for y in range(0, 341, 34)]
    out, shift = row_heights.rephase_edges(edges, prof, 34)
    assert shift == 0
    assert np.allclose(out, edges)


def test_種名側だけ上下にずれて打たれた紙面は種名の境だけ動かす():
    # 組成部の「・」は行の中央，種名の字は 10 px 下に打たれている(14_p1 型)
    edges = list(range(100, 100 + 34 * 12 + 1, 34))
    comp = _grid(edges, [300, 340, 380, 420], 'comp')
    sname = _grid(edges, [0, 200], 'sname')
    df = pd.concat([sname, comp], ignore_index=True)
    out, warns = row_heights.fix_row_heights(_page(edges, name_shift=10), df)
    assert any('種名側' in w for w in warns)
    c, s = _rows(out, 'comp'), _rows(out, 'sname')
    assert sorted(c['row'].unique()) == sorted(s['row'].unique())     # 行の対応は同じ
    off = (s.groupby('row')['y1'].first() - c.groupby('row')['y1'].first())
    assert (off == off.iloc[0]).all() and 6 <= off.iloc[0] <= 14
    assert np.allclose(c['y1'].unique(), edges[:-1])                 # 組成部は動かない


def test_半分の刻みの格子は2倍の高さに組み直す():
    # 印字は 40 px おき(中央に「・」)．格子は 20 px で刻まれている
    n_print = 20                                # 格子は 40 行(30 行未満では判定しない)
    top = 100
    print_edges = [top + 40 * k for k in range(n_print + 1)]
    edges20 = [float(top + 20 * k) for k in range(2 * n_print + 1)]
    df = pd.concat([_grid(edges20, [0, 200], 'sname'),
                    _grid(edges20, [300, 340, 380, 420], 'comp')], ignore_index=True)
    img = _page(print_edges, size=(500, 1000))
    prof = row_heights._profile(row_heights.ink.binarize(img), 300, 420)
    assert row_heights.is_half_pitch(prof, top, top + 40 * n_print, 20)
    assert not row_heights.is_half_pitch(prof, top, top + 40 * n_print, 40)
    # 行の少ない表では判定しない(kinki_017 は 14 行の正しい格子が半分に減った)
    assert not row_heights.is_half_pitch(prof, top, top + 40 * 10, 20)
    out, warns = row_heights.fix_row_heights(img, df)
    assert any('半分' in w for w in warns)
    c = _rows(out, 'comp')
    assert c['row'].nunique() == n_print
    h = np.diff(np.sort(c['y1'].unique()))
    assert h.min() >= 40 * 0.8 and h.max() <= 40 * 1.2
    for y in c['y1'].unique()[1:]:
        assert prof[int(y)] == 0                # 境は「・」の上を通らない
