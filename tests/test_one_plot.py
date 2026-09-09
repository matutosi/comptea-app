"""1 調査区の 2 段組で，階層と被度の列を黒画素から作る (one_plot.py．対策 G)

紙面は「1 つの調査区の出現種を左右 2 段に折り返し，1 段が
`学名 | 和名 | 階層 (B・S・K) | 被度 (3・3 など)` の 4 つ」からなる (kinki_001・047・
053・077)．地点の列が 1 本しかないので `col` の検出が育たず，被度が丸ごと落ちる．

分かれ目は**行ごとのインクの塊の左端**の山．和名の中の山の間隔は行の高さの
0.6〜0.8 倍だが，和名と階層のあいだだけ 2.1〜3.6 倍あく (2026-09-09 に 8 段で実測)．
"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import one_plot


PITCH = 50
N_ROWS = 24
Y0 = 300
# 1 段の中の 4 つの部分 (学名・和名・階層・被度)
# 和名は左寄せで，右に余白が残る (実データでは和名の最後の字から階層まで
# 行の高さの 2.1〜3.6 倍あく)
LEFT = dict(sname=(260, 780), ja=(790, 1000), layer=(1110, 1150), cover=(1160, 1210))
RIGHT = dict(sname=(1250, 1740), ja=(1750, 1960), layer=(2070, 2110), cover=(2120, 2170))


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _words(px, x1, x2, y, h, w=22, gap=14):
    """字の並び (塊の左端が一定の間隔で並ぶ)"""
    x = int(x1)
    while x + w <= int(x2):
        _fill(px, x, x + w, y, y + h)
        x += w + gap


def _sheet(size=(2470, 1800), rows=N_ROWS, blocks=(LEFT, RIGHT)):
    img = Image.new('L', size, 255)
    px = img.load()
    for b in blocks:
        for i in range(rows):
            y = Y0 + i * PITCH
            _words(px, b['sname'][0], b['sname'][1], y, 24)
            _words(px, b['ja'][0], b['ja'][1], y, 24, w=26, gap=12)
            _fill(px, b['layer'][0] + 4, b['layer'][1] - 4, y, y + 24)
            _fill(px, b['cover'][0] + 4, b['cover'][1] - 6, y, y + 24)
    return img


def _det(blocks=(LEFT, RIGHT), rows=N_ROWS, src='x.jpg'):
    """検出 (学名と和名の箱だけ．col は 0 本，plot_row も 0 本)"""
    out = []
    for b in blocks:
        for key, obj in (('sname', 'sname'), ('ja', 'species_col')):
            out.append(dict(obj_name=obj, x1=float(b[key][0]), x2=float(b[key][1]),
                            y1=float(Y0), y2=float(Y0 + rows * PITCH),
                            confidence=0.9, source_image=src))
        for i in range(rows):
            out.append(dict(obj_name='row', x1=float(b['sname'][0]),
                            x2=float(b['cover'][1]), y1=float(Y0 + i * PITCH),
                            y2=float(Y0 + i * PITCH + PITCH),
                            confidence=0.9, source_image=src))
    return pd.DataFrame(out)


def _added(df, obj):
    return df[df.obj_name == obj].sort_values('x1')


def test_階層と被度の列を作る():
    out, warns = one_plot.columns_from_ink(_sheet(), _det())
    lay, col = _added(out, 'layer'), _added(out, 'col')
    assert len(lay) == 2 and len(col) == 2       # 段ごとに 1 本ずつ
    assert warns and any('1 調査区' in w for w in warns)
    for got, want in zip(lay.itertuples(), (LEFT, RIGHT)):
        assert abs(got.x1 - want['layer'][0]) <= 12
        assert abs(got.x2 - want['layer'][1]) <= 12
    for got, want in zip(col.itertuples(), (LEFT, RIGHT)):
        assert abs(got.x1 - want['cover'][0]) <= 12
        assert abs(got.x2 - want['cover'][1]) <= 18


def test_和名の字間では切らない():
    out, _w = one_plot.columns_from_ink(_sheet(), _det())
    lay = _added(out, 'layer')
    assert (lay['x1'] > LEFT['ja'][1] - 30).all() or len(lay) == 2
    for got, want in zip(lay.itertuples(), (LEFT, RIGHT)):
        assert got.x1 > want['ja'][0] + 100      # 和名の中には入らない


def test_行が少ない段には当てない():
    det = _det(rows=8)
    out, warns = one_plot.columns_from_ink(_sheet(rows=8), det)
    assert (out.obj_name == 'col').sum() == 0
    assert any('行が少ない' in w for w in warns)


def test_地点の列がある表には当てない():
    det = _det()
    det = pd.concat([det, pd.DataFrame([
        dict(obj_name='plot_row', x1=1000.0, x2=2000.0, y1=100.0, y2=200.0,
             confidence=0.9, source_image='x.jpg')])], ignore_index=True)
    out, warns = one_plot.columns_from_ink(_sheet(), det)
    assert out is det and warns == []


def test_段が1つなら当てない():
    out, warns = one_plot.columns_from_ink(_sheet(blocks=(LEFT,)), _det(blocks=(LEFT,)))
    assert (out.obj_name == 'col').sum() == 0
    assert warns == [] or not any('作った' in w for w in warns)


def test_検出の箱は消さない():
    det = _det()
    out, _w = one_plot.columns_from_ink(_sheet(), det)
    for obj in ('sname', 'species_col', 'row'):
        assert (out.obj_name == obj).sum() == (det.obj_name == obj).sum()


def test_階層の幅と被度の幅は行の高さに見合う():
    out, _w = one_plot.columns_from_ink(_sheet(), _det())
    lay, col = _added(out, 'layer'), _added(out, 'col')
    for r in lay.itertuples():
        assert 0.4 * PITCH <= (r.x2 - r.x1) <= 1.1 * PITCH
    for r in col.itertuples():
        assert 0.7 * PITCH <= (r.x2 - r.x1) <= 2.0 * PITCH


def test_横罫線があっても決まる():
    img = _sheet()
    px = img.load()
    _fill(px, 200, 2300, Y0 - 20, Y0 - 16)       # 表の上の横罫線
    out, _w = one_plot.columns_from_ink(img, _det())
    assert (out.obj_name == 'col').sum() == 2


def test_山の間隔が開かない紙面には当てない():
    # 和名と階層のあいだが空いていない (4 部分に分かれない紙面)
    b = dict(sname=(260, 780), ja=(790, 1090), layer=(1105, 1145), cover=(1150, 1200))
    img = _sheet(blocks=(b,))
    out, _w = one_plot.columns_from_ink(img, _det(blocks=(b,)))
    assert (out.obj_name == 'col').sum() == 0
