"""表頭の境を，項目名の行と値の行の対応で置き直す (header_lines.bands_from_pairs・value_lines)

項目名の区切りをそのまま使うと，値が 2〜3 行にわたる項目で値の行を割る
(22_p3 の「調査年月日」．2026-09-10 ユーザ指示: 値の側にも区切りを仮に作り，
項目名の区切りに対応しない値の区切りは落とす)．
"""
import numpy as np

from comptea import header_lines as hl


PITCH = 34.0


def _span(y1, y2):
    return ((y1 + y2) / 2.0, float(y1), float(y2))


def test_値が2行にわたる項目の境は値の行のあいだへ動く():
    # 項目 A (値 1 行)，項目 B (値 2 行)，項目 C (値 1 行)．元の境が B の 2 行目を割っている
    items = [_span(100, 130), _span(160, 190), _span(230, 260)]
    values = [_span(100, 125), _span(150, 175), _span(185, 210), _span(235, 260)]
    edges = np.array([90.0, 145.0, 200.0, 280.0])    # 200 は 185-210 の内側
    out = hl.bands_from_pairs(items, values, edges, pitch=PITCH)
    assert 210 < out[2] < 235                        # B の 2 行目の下へ
    assert out[1] == 145                             # 割っていない境は動かさない


def test_値が1行ずつの項目の境は動かさない():
    # 05_p2 の調査面積と海抜高: 境が値の行にわずかにかかっても，1 行ずつなら触らない
    items = [_span(100, 130), _span(140, 170), _span(180, 210)]
    values = [_span(100, 125), _span(140, 165), _span(180, 205)]
    edges = np.array([90.0, 136.0, 166.0, 220.0])    # 166 は 140-165 の 1 px 外
    out = hl.bands_from_pairs(items, values, edges, pitch=PITCH)
    assert list(out) == list(edges)


def test_2行ぶんより高い項目名の行に隣る境は動かさない():
    # OCR の行が 2 行つながった (05_p2 の 92 px)．中心が当てにならない
    items = [_span(100, 130), _span(140, 232), _span(240, 270)]     # 2 つ目が 92 px
    values = [_span(100, 125), _span(140, 165), _span(180, 205), _span(240, 265)]
    edges = np.array([90.0, 136.0, 190.0, 280.0])    # 190 は 180-205 の内側だが動かさない
    out = hl.bands_from_pairs(items, values, edges, pitch=PITCH)
    assert list(out) == list(edges)


def test_境は昇順を保つ():
    items = [_span(100, 130), _span(160, 190), _span(230, 260)]
    values = [_span(100, 125), _span(150, 175), _span(185, 210), _span(235, 260)]
    edges = np.array([90.0, 145.0, 200.0, 280.0])
    out = hl.bands_from_pairs(items, values, edges, pitch=PITCH)
    assert (np.diff(out) >= 0).all()


def test_値の行は黒画素の連なりから作る():
    # 字らしい短い塊で描く (べた塗りの帯は罫線として消される)
    dark = np.zeros((300, 400), dtype=bool)
    for y in (100, 140, 180, 220):                   # 4 行の値
        for x in range(50, 350, 30):
            dark[y:y + 20, x:x + 12] = True
    dark[165, 50:80] = True                          # 1 px のかけら (句読点)
    vs = hl.value_lines(dark, (0, 80, 400, 260), PITCH)
    assert len(vs) == 4
    assert [round(v[1]) for v in vs] == [100, 140, 180, 220]


def test_黒が無ければ空():
    assert hl.value_lines(np.zeros((100, 100), dtype=bool), (0, 0, 100, 100), PITCH) == []


def test_表頭の値の帯を表頭の傾きで列ごとにずらす():
    """`shear_header_values`: 右下がりの値の行なら，右の列ほど下がる"""
    import pandas as pd
    from PIL import Image
    from comptea import header_lines as hl
    w, h = 1200, 300
    img = Image.new('L', (w, h), 255)
    px = img.load()
    slope = 12 / 1000                            # 1000 px で 12 px 下がる
    for r in range(6):                           # 6 行の値
        for x in range(100, 1100, 50):
            y = 40 + r * 30 + int(slope * (x - 600))
            for xx in range(x, x + 12):
                for yy in range(y, y + 14):
                    px[xx, yy] = 0
    cells = []
    for c, xa in enumerate(range(100, 1100, 100), 1):
        for r in range(6):
            cells.append(dict(obj_name='header_value', block=1, row=r + 1, col=c,
                              x1=xa, x2=xa + 100, y1=30.0 + r * 30, y2=60.0 + r * 30))
    for c, xa in enumerate(range(100, 1100, 100), 1):
        for r in range(4):
            cells.append(dict(obj_name='comp', block=1, row=r + 1, col=c,
                              x1=xa, x2=xa + 100, y1=220.0 + r * 30, y2=250.0 + r * 30))
    df = pd.DataFrame(cells)
    out, warns = hl.shear_header_values(img, df)
    assert warns
    hv = out[out.obj_name == 'header_value']
    left = hv[hv.col == 1].y1.iloc[0]
    right = hv[hv.col == 10].y1.iloc[0]
    assert right - left >= 6                     # 右の列が下がる
    assert (out[out.obj_name == 'comp'].y1.values == df[df.obj_name == 'comp'].y1.values).all()
