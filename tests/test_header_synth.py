"""header_col が検出されなくても，項目名の字があれば項目名の列を補う

s01115_14_p5・15_p1・15_p2・15_p4・21_p3 は項目名 (独文・和文) が紙面にあるのに
`header_col` が検出されず，表頭が値だけになっていた (2026-09-10 ユーザ指摘)．
領域は表頭の枠の左端から値の左端まで．独文と和文の境は対策 B が黒画素で決める．
"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import header_lines, ink, locate


W, H = 1600, 400
BANDS = [[100.0 + 30 * i, 130.0 + 30 * i] for i in range(8)]
X_LEFT, X_VALUE = 40.0, 1200.0


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _sheet(names=True, rule=True):
    img = Image.new('L', (W, H), 255)
    px = img.load()
    for y1, y2 in BANDS:
        if names:
            for x in range(60, 500, 14):            # 独文
                _fill(px, x, x + 8, y1 + 6, y2 - 6)
            for x in range(700, 900, 14):           # 和文
                _fill(px, x, x + 8, y1 + 6, y2 - 6)
        for x in range(1220, 1580, 60):             # 値
            _fill(px, x, x + 20, y1 + 6, y2 - 6)
    if rule:                                        # 値の手前の縦罫線
        _fill(px, 1195, 1198, 90, 340)
    return img


def _guess(img, left=X_LEFT, value=X_VALUE):
    return locate._guess_header_col(ink.binarize(img), left, value, BANDS, 30.0)


def test_字があれば補う():
    assert _guess(_sheet()) == (X_LEFT, X_VALUE)


def test_字が無ければ補わない():
    # 縦罫線だけの領域 (罫線は字ではない)
    assert _guess(_sheet(names=False)) is None


def test_狭ければ補わない():
    assert _guess(_sheet(), left=1150.0) is None


def _det():
    rows = [dict(obj_name='header', x1=X_LEFT, y1=90.0, x2=1580.0, y2=340.0)]
    for y1, y2 in BANDS:
        rows.append(dict(obj_name='plot_row', x1=X_LEFT, y1=y1, x2=1580.0, y2=y2))
    df = pd.DataFrame(rows)
    df['source_image'] = 'x.jpg'
    df['confidence'] = 0.9
    return df


def test_locate_headerが項目名の列を補う(monkeypatch):
    # OCR の検出器は使わない (投影の帯へ戻る)
    monkeypatch.setattr(header_lines, 'header_bands', lambda *a, **k: None)
    x_edges = np.array([X_VALUE + 60 * i for i in range(7)])
    warns = []
    out = locate._locate_header(_det(), 'x.jpg', x_edges, warns, img=_sheet())
    names = {str(o['obj_name'].iloc[0]) for o in out}
    assert {'header_value', 'header_item', 'header_item_ja'} <= names
    item = next(o for o in out if o['obj_name'].iloc[0] == 'header_item')
    ja = next(o for o in out if o['obj_name'].iloc[0] == 'header_item_ja')
    assert float(item['x1'].iloc[0]) == X_LEFT
    assert 500 <= float(item['x2'].iloc[0]) <= 700            # 独文と和文のあいだ
    assert float(ja['x2'].iloc[0]) == X_VALUE
    assert any('補った' in w for w in warns)


def test_字が無ければ値だけ(monkeypatch):
    monkeypatch.setattr(header_lines, 'header_bands', lambda *a, **k: None)
    x_edges = np.array([X_VALUE + 60 * i for i in range(7)])
    warns = []
    out = locate._locate_header(_det(), 'x.jpg', x_edges, warns, img=_sheet(names=False))
    names = {str(o['obj_name'].iloc[0]) for o in out}
    assert names == {'header_value'}
