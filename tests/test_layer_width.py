"""階層の列の幅を黒画素から決め直す (layer_col.refit_layer_width．対策 E)

階層の列は，表頭が空であることから組成部の先頭の列を階層とみなして作るので，
**幅は地点の列のまま**になる．記号 (B1・B2・S・K) の実際の幅とは合わず，
81 段中 66 段で枠の右端が縦罫線の上に乗り，記号が枠をはみ出す (13.5% の行)．
"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import layer_col


X_NAME = [0, 300]
X_COMP = [400, 460, 520, 580]      # 地点の列は幅 60
EDGES = list(range(100, 100 + 34 * 14 + 1, 34))
PITCH = 34.0


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _text(px, x1, x2, y1, y2, w=7, gap=4):
    x = int(x1)
    while x + w <= int(x2):
        _fill(px, x, x + w, y1, y2)
        x += w + gap


def _grid(edges, x_edges, obj_name, block=1, col0=1):
    rows = []
    for c, (xa, xb) in enumerate(zip(x_edges[:-1], x_edges[1:]), col0):
        for r, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
            rows.append(dict(x1=float(xa), x2=float(xb), y1=float(ya), y2=float(yb),
                             obj_name=obj_name, note='', block=block, row=r, col=c))
    return pd.DataFrame(rows)


def _df(layer_x=(400, 460), comp_x=None):
    comp_x = comp_x or X_COMP[1:]
    return pd.concat([
        _grid(EDGES, X_NAME, 'sname'),
        _grid(EDGES, list(layer_x), 'layer'),
        _grid(EDGES, list(comp_x), 'comp', col0=2),
    ], ignore_index=True)


def _sheet(sym=(405, 445), rule_at=None, name_tail=None, size=(700, 700)):
    """階層の記号を `sym` の範囲に置いた紙面

    rule_at: 縦罫線を引く x．name_tail: 種名の末尾を帯まで垂らす x の範囲．
    """
    img = Image.new('L', size, 255)
    px = img.load()
    for ya, yb in zip(EDGES[:-1], EDGES[1:]):
        c = int((ya + yb) // 2)
        _text(px, 10, 250, c - 8, c + 8)                     # 学名
        _fill(px, sym[0], sym[1], c - 8, c + 8)              # 階層の記号
        for xa, xb in zip(X_COMP[1:-1], X_COMP[2:]):
            xc = (xa + xb) // 2
            _fill(px, xc - 2, xc + 3, c - 3, c + 4)          # 「・」
        if name_tail:
            _fill(px, name_tail[0], name_tail[1], c - 6, c + 6)
    if rule_at:
        _fill(px, rule_at, rule_at + 2, EDGES[0], EDGES[-1])
    return img


def _layer_x(df):
    lay = df[df.obj_name == 'layer']
    return float(lay.x1.min()), float(lay.x2.max())


def test_記号がはみ出す枠を広げる():
    # 記号は 395〜465 だが，枠は 400〜460 しかない
    img = _sheet(sym=(395, 465))
    df = _df(layer_x=(400, 460))
    out, warns = layer_col.refit_layer_width(img, df)
    x1, x2 = _layer_x(out)
    assert x1 <= 395 and x2 >= 465
    assert warns and any('階層の列' in w for w in warns)


def test_記号が収まっていれば触らない():
    img = _sheet(sym=(410, 450))
    df = _df(layer_x=(400, 460))
    out, warns = layer_col.refit_layer_width(img, df)
    assert out is df and warns == []


def test_縦罫線は幅に数えない():
    # 枠の右端に縦罫線が乗っている (81 段中 66 段がこの形)．記号は左へはみ出して
    # いるので幅は測り直すが，罫線までは広げない
    img = _sheet(sym=(393, 445), rule_at=458)
    df = _df(layer_x=(400, 460))
    out, warns = layer_col.refit_layer_width(img, df)
    x1, x2 = _layer_x(out)
    assert warns and any('決め直した' in w for w in warns)
    assert x1 <= 393 and x2 <= 455         # 罫線 (458) まで広げない


def test_はみ出していない段は幅を測り直さない():
    # 枠の右端に罫線があっても，記号が収まっていれば触らない
    img = _sheet(sym=(405, 445), rule_at=458)
    df = _df(layer_x=(400, 460))
    out, warns = layer_col.refit_layer_width(img, df)
    assert out is df and warns == []


def test_種名を巻き込まない():
    # 種名の末尾が帯まで垂れている紙面 (kinki_076・049 型)
    img = _sheet(sym=(405, 445), name_tail=(300, 380))
    df = _df(layer_x=(400, 460))
    out, _w = layer_col.refit_layer_width(img, df)
    x1, _x2 = _layer_x(out)
    assert x1 >= 390                       # 種名の側へ逃げない


def test_広げるのは上限まで():
    img = _sheet(sym=(330, 465))           # 帯いっぱいに広がった塊
    df = _df(layer_x=(400, 460))
    out, _w = layer_col.refit_layer_width(img, df)
    x1, x2 = _layer_x(out)
    assert (x2 - x1) <= max(2.0 * PITCH, 1.5 * 60) + 2


def test_行番号と他の列は変わらない():
    img = _sheet(sym=(395, 465))
    df = _df(layer_x=(400, 460))
    out, _w = layer_col.refit_layer_width(img, df)
    assert len(out) == len(df)
    assert out.row.nunique() == df.row.nunique()
    for cls in ('sname', 'comp'):
        a = df[df.obj_name == cls].sort_index()[['x1', 'x2', 'y1', 'y2']].values
        b = out[out.obj_name == cls].sort_index()[['x1', 'x2', 'y1', 'y2']].values
        assert np.array_equal(a, b)


def test_階層が無ければ何もしない():
    img = _sheet()
    df = pd.concat([_grid(EDGES, X_NAME, 'sname'),
                    _grid(EDGES, X_COMP, 'comp')], ignore_index=True)
    out, warns = layer_col.refit_layer_width(img, df)
    assert out is df and warns == []
