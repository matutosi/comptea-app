"""格子を組む部品(短冊・階層の列・種名の列)

どれも実データでしか動かしていなかったところ．組み立てた検出と紙面で，
**決めごとが守られているか**だけを確かめる (数字そのものは 153 表の通しで測る)．
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from comptea import layer_col, name_col, strips


def det(rows):
    return pd.DataFrame(rows)


def box(obj, x1, y1, x2, y2, **kw):
    return {'obj_name': obj, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, **kw}


# --- 短冊に分けるか -----------------------------------------------------

def test_行が取れていれば短冊に分けない():
    """ふつうの表(行も列も取れている)は，この経路に入らない"""
    df = det([box('col', 200 + i * 60, 100, 260 + i * 60, 700) for i in range(6)]
             + [box('row', 0, 100 + i * 30, 560, 130 + i * 30) for i in range(20)]
             + [box('sname', 0, 100, 200, 700)])
    assert strips.needs_strips(df) is False


def test_列が多く行が取れないと短冊に分ける():
    """行の箱が横に長すぎると `row` が 1 本も検出できない

    37 地点の表は，どの `imgsz` でも row が 0 件だった(2026-09-02)．
    縮尺ではなく**行の横幅**が学習時から外れているのが原因．
    """
    df = det([box('col', 200 + i * 60, 100, 260 + i * 60, 700) for i in range(20)]
             + [box('sname', 0, 100, 200, 700)])
    assert strips.needs_strips(df) is True


# --- 短冊の切り方 -------------------------------------------------------

def test_切れ目は隙間の中央にしか置かない():
    gaps = list(range(300, 2000, 100))
    cuts = strips.plan_strips(gaps, name_w=200, body_x1=200, body_x2=2000)
    assert cuts[0][0] == 200 and cuts[-1][1] == 2000
    for _, x2 in cuts[:-1]:
        assert x2 in gaps


def test_短冊は地点1つぶん重ねる():
    """短冊の端に来た地点は検出が弱く，継ぎ目のたびに列を落としていた

    37 地点の表で 32 列しか取れなかった(2026-09-02)．
    """
    gaps = list(range(300, 3000, 100))
    cuts = strips.plan_strips(gaps, name_w=200, body_x1=200, body_x2=3000)
    assert len(cuts) >= 2
    for (a1, a2), (b1, b2) in zip(cuts, cuts[1:]):
        assert b1 < a2                       # 前の短冊の右端より左から始まる


def test_隙間が無ければ切らない():
    assert strips.plan_strips([], 200, 200, 2000) == [(200, 2000)]


# --- 組成部の左端 -------------------------------------------------------

def test_離れた列は左端に使わない():
    """左に離れた `col` 1 本で組成部の左端を誤ると，種名が探索の外に出る

    s01115_13_p1 は x90-139 の 1 本で左端が 90 px になり，本体 x1191 の
    種名が拾えなかった(2026-09-05)．
    """
    df = det([box('col', 90, 100, 139, 700)]                      # 離れた 1 本
             + [box('col', 1200 + i * 60, 100, 1260 + i * 60, 700)
                for i in range(8)])
    assert name_col.body_left(df) >= 1000


def test_ふつうの表では左端がそのまま():
    df = det([box('col', 300 + i * 60, 100, 360 + i * 60, 700) for i in range(8)])
    assert name_col.body_left(df) == 300


# --- 種名の列を黒画素から補う -------------------------------------------

def name_page(tmp_path, sname=(40, 200), jname=(240, 400), body=(460, 1000)):
    """学名・和名・組成部が横に並ぶ紙面を描く"""
    a = np.full((800, 1100), 255, dtype=np.uint8)
    for y in range(120, 700, 30):
        a[y:y + 12, sname[0]:sname[1]] = 0            # 学名(左)
        a[y:y + 12, jname[0]:jname[1]] = 0            # 和名(右)
        for x in range(body[0], body[1], 60):
            a[y + 3:y + 9, x + 20:x + 32] = 0         # 値
    p = tmp_path / 'names.png'
    Image.fromarray(a).save(p)
    return Image.open(p)


def test_種名の列が無ければ黒画素の山から作る(tmp_path):
    """検出が 0 件でも，組成部の左は 2 つの山と谷に見える(2026-09-04)

    左が学名・右が和名．ラベル済み 33 枚すべてでこの並びだった．
    """
    img = name_page(tmp_path)
    df = det([box('col', 460 + i * 60, 100, 520 + i * 60, 700) for i in range(9)]
             + [box('header', 0, 40, 1000, 100)])
    made, _ = name_col.name_columns_from_ink(img, df)
    got = {r['obj_name']: (r['x1'], r['x2'])
           for _, r in made[made['obj_name'].isin(['sname', 'species_col'])].iterrows()}
    assert set(got) == {'sname', 'species_col'}
    assert got['sname'][1] <= got['species_col'][0]      # 学名が左


def test_片方だけ足りないときは足りない側だけ補う(tmp_path):
    """68 表のうち 28 表がこの形だった(2026-09-05)"""
    img = name_page(tmp_path)
    df = det([box('col', 460 + i * 60, 100, 520 + i * 60, 700) for i in range(9)]
             + [box('header', 0, 40, 1000, 100)]
             + [box('sname', 40, 120, 200, 700)])
    made, _ = name_col.name_columns_from_ink(img, df)
    snames = made[made['obj_name'] == 'sname']
    assert len(snames) == 1                              # 元の箱は作り直さない
    assert (snames.iloc[0]['x1'], snames.iloc[0]['x2']) == (40, 200)
    assert (made['obj_name'] == 'species_col').sum() == 1


# --- 階層の列 -----------------------------------------------------------

def layer_page(tmp_path, layer_ink=True):
    """表頭が空で中身のある列(= 階層の列)を持つ紙面"""
    a = np.full((800, 900), 255, dtype=np.uint8)
    for y in range(60, 110, 20):                         # 表頭の値(地点の列)
        for x in range(300, 900, 100):
            a[y:y + 10, x + 20:x + 60] = 0
    for y in range(140, 700, 30):
        if layer_ink:
            a[y:y + 12, 220:250] = 0                     # 階層の列(表頭は空)
        for x in range(300, 900, 100):
            a[y + 3:y + 9, x + 30:x + 50] = 0
    p = tmp_path / 'layer.png'
    Image.fromarray(a).save(p)
    return str(p)


def loc_with_layer_col(rows=18):
    """先頭に「表頭が空の列」を持つ格子"""
    out = []
    xs = [(200, 300)] + [(300 + i * 100, 400 + i * 100) for i in range(6)]
    for r in range(rows):
        for c, (x1, x2) in enumerate(xs, start=1):
            out.append(box('comp', x1, 140 + r * 30, x2, 170 + r * 30,
                           row=r + 1, col=c))
    for c, (x1, x2) in enumerate(xs, start=1):
        out.append(box('header_value', x1, 50, x2, 110, row=1, col=c))
    return det(out)


def test_表頭が空で字のある列は階層にする(tmp_path):
    """地点の列は表頭が埋まる(通し番号・調査日・海抜高)．階層の列だけ空

    先頭の列 0.07-0.16 に対し，地点の列は 0.85 以上(2026-09-02)．
    """
    img = layer_page(tmp_path)
    got, _ = layer_col.fix_columns(img, loc_with_layer_col())
    first = got[(got['obj_name'] == 'layer')]
    assert len(first) > 0
    assert set(first['col']) == {1}                      # 先頭の 1 列だけ


def test_表頭も本体も空の列は捨てる(tmp_path):
    """種名と組成部のあいだの隙間が，地点として数えられていた(2026-09-03)"""
    img = layer_page(tmp_path, layer_ink=False)
    got, _ = layer_col.fix_columns(img, loc_with_layer_col())
    assert (got['obj_name'] == 'layer').sum() == 0
    assert got[got['obj_name'] == 'comp']['col'].min() >= 1


def test_列の検出が無ければ左端は決められない():
    df = det([box('sname', 0, 100, 200, 700)])
    assert name_col.body_left(df) is None


# --- 横倒しに組まれた表 -------------------------------------------------

def rotated_page(tmp_path, turn):
    """字の行が横に走る紙面(`turn` なら 90 度回して縦に走らせる)"""
    # **短辺 1500 px 未満は判定しない**(見出しだけの帯は字の行が数本しかなく，
    # 比が当てにならない)．そこを超える大きさにする
    a = np.full((2400, 1600), 255, dtype=np.uint8)
    for y in range(100, 2300, 30):                 # 字の行
        a[y:y + 14, 100:1500] = 0
    im = Image.fromarray(a)
    return im.transpose(Image.ROTATE_90) if turn else im


def test_横倒しの表を見分けて90度回す(tmp_path):
    """紙面の都合で 90 度回して組んだ表がある(s01115_04 の 3 表)

    そのまま渡すと行と列が入れ替わり，46 地点の表から 102 列の格子ができた．
    **切り出しと回転は 1 か所にまとめてある**(2026-09-07)．前は 3 か所に
    写してあり，アプリだけが回していなかった．
    """
    from comptea import ink, split_sheet

    upright = rotated_page(tmp_path, turn=False)
    sideways = rotated_page(tmp_path, turn=True)
    assert split_sheet.looks_rotated(ink.binarize(upright)) is False
    assert split_sheet.looks_rotated(ink.binarize(sideways)) is True

    cut, turned = split_sheet.cut_table(sideways, (0, 0, *sideways.size))
    assert turned is True
    assert cut.size == (sideways.size[1], sideways.size[0])   # 縦横が入れ替わる

    cut, turned = split_sheet.cut_table(upright, (0, 0, *upright.size))
    assert turned is False and cut.size == upright.size


def test_小さすぎる切れ端は回さない(tmp_path):
    """見出しだけの帯は字の行が数本しかなく，向きの比が当てにならない"""
    from comptea import ink, split_sheet

    small = rotated_page(tmp_path, turn=True).resize((300, 400))
    assert split_sheet.looks_rotated(ink.binarize(small)) is False
