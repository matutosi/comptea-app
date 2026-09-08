"""行を「単位の連なり」として追う後処理 (row_track.py．案 e の段階 1)

段階 1 は 2 つ: (1) 列ごとの y のずれを単位 (黒画素の連なり) の中央値で吸収する
`fix_offsets`．行番号は全列で共有したまま y だけ列ごとにずらす．(2) 目印の列の拍で
行数を検算して警告する `check_beats`．
"""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from comptea import row_track, row_skew


X_COMP = [300, 340, 380, 420]        # 組成 3 列
X_SN = [0, 150]                      # 学名
X_JA = [160, 290]                    # 和名


def _grid(edges, x_edges, obj_name, block=1):
    rows = []
    for c, (xa, xb) in enumerate(zip(x_edges[:-1], x_edges[1:]), 1):
        for r, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
            rows.append(dict(x1=xa, x2=xb, y1=float(ya), y2=float(yb), obj_name=obj_name,
                             note='', block=block, row=r, col=c))
    return pd.DataFrame(rows)


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _text(px, x1, x2, y1, y2, w=6, gap=3):
    """字の並び: 幅 `w` px の塊を `gap` px あけて置く (ベタ塗りだと下線と区別できない)"""
    x = int(x1)
    while x + w <= int(x2):
        _fill(px, x, x + w, y1, y2)
        x += w + gap


def _sheet(edges, name_shift=0, ja_shift=0, ja_rows=None, headings=(), miss_dots=(),
           half_left=(), x_comp=X_COMP, slope=0.0, size=(500, 800)):
    """印字の行 `edges` ごとに，組成の「・」・学名 (16 px)・和名 (16 px) を置いた紙面

    name_shift / ja_shift: 学名・和名を行の中央から下へずらす px (タイプ打ちの 10 px)．
    ja_rows: 和名を置く行 (None なら全行)．headings: 学名だけ + 長い下線の見出し行．
    miss_dots: 「・」を置かない行 (薄くて消えた)．half_left: 左端の列に半行ずれた
    「・」を余分に置く行 (2 段の値)．slope: 右へ行くほど下がる傾き (dy/dx)．
    """
    img = Image.new('L', size, 255)
    px = img.load()
    x0 = (x_comp[0] + x_comp[-1]) / 2.0
    for i, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
        c = (ya + yb) // 2
        if i in headings:
            _text(px, 10, 140, c - 8, c + 8)
            _fill(px, 10, 300, c + 11, c + 13)            # 290 px の下線
            continue
        for k, (xa, xb) in enumerate(zip(x_comp[:-1], x_comp[1:])):
            xc = (xa + xb) // 2
            y = int(round(c + slope * (xc - x0)))
            if i not in miss_dots:
                _fill(px, xc - 2, xc + 3, y - 3, y + 4)
            if k == 0 and i in half_left:
                _fill(px, xc - 2, xc + 3, y + 14, y + 21)
        ys = int(round(c + slope * (75 - x0)))
        _text(px, 10, 140, ys - 8 + name_shift, ys + 8 + name_shift)
        if ja_rows is None or i in ja_rows:
            yj = int(round(c + slope * (225 - x0)))
            _text(px, 165, 280, yj - 8 + ja_shift, yj + 8 + ja_shift)
    return img


def _df(edges, x_comp=X_COMP):
    return pd.concat([_grid(edges, X_SN, 'sname'), _grid(edges, X_JA, 'species_col'),
                      _grid(edges, x_comp, 'comp')], ignore_index=True)


def _rows(out, cls):
    return out[out.obj_name == cls].sort_values(['row', 'col'])


EDGES = list(range(100, 100 + 34 * 12 + 1, 34))     # 12 行，高さ 34


def test_学名が下にずれて打たれた紙面は学名の列だけ下へずらす():
    img = _sheet(EDGES, name_shift=10)
    df = _df(EDGES)
    out, warns = row_track.fix_offsets(img, df)
    assert any('sname' in w and 'ずらした' in w for w in warns)
    c, s = _rows(out, 'comp'), _rows(out, 'sname')
    assert np.array_equal(c[['y1', 'y2']].values, _rows(df, 'comp')[['y1', 'y2']].values)
    assert set(s['row']) == set(c['row'])                       # 行番号は共有
    d = s['y1'].values - c[c.col == 1]['y1'].values
    assert np.all(np.abs(d - 10) <= 2)
    assert (s['note'] == 'y_shifted').all()


def test_ずれが無ければ触らない():
    img = _sheet(EDGES)
    df = _df(EDGES)
    out, warns = row_track.fix_offsets(img, df)
    assert out is df and warns == []


def test_半行を超えるずれは信用しない():
    img = _sheet(EDGES, name_shift=16)                          # 0.47 p (上限 0.45 p の外)
    df = _df(EDGES)
    out, warns = row_track.fix_offsets(img, df)
    s = _rows(out, 'sname')
    assert np.array_equal(s['y1'].values, _rows(df, 'sname')['y1'].values)
    assert any('sname' in w and '信用' in w for w in warns)


def test_学名と和名は別々のずれで動く():
    img = _sheet(EDGES, name_shift=10, ja_shift=-6)
    out, _ = row_track.fix_offsets(img, _df(EDGES))
    c = _rows(out, 'comp')
    c1 = c[c.col == 1]['y1'].values
    assert np.all(np.abs(_rows(out, 'sname')['y1'].values - c1 - 10) <= 2)
    assert np.all(np.abs(_rows(out, 'species_col')['y1'].values - c1 + 6) <= 2)


def test_下線つきの見出し行があっても推定は乱れない():
    img = _sheet(EDGES, name_shift=10, headings=(1, 7))
    out, _ = row_track.fix_offsets(img, _df(EDGES))
    c = _rows(out, 'comp')
    d = _rows(out, 'sname')['y1'].values - c[c.col == 1]['y1'].values
    assert np.all(np.abs(d - 10) <= 2)


def test_和名が1行おきに空いても動く():
    img = _sheet(EDGES, ja_shift=8, ja_rows=range(1, 13, 2))    # 12 行中 6 行
    out, _ = row_track.fix_offsets(img, _df(EDGES))
    c = _rows(out, 'comp')
    d = _rows(out, 'species_col')['y1'].values - c[c.col == 1]['y1'].values
    assert np.all(np.abs(d - 8) <= 2)


def test_単位が少ない列は触らない():
    img = _sheet(EDGES, ja_shift=8, ja_rows=(1, 5, 9))         # 12 行中 3 行
    df = _df(EDGES)
    out, warns = row_track.fix_offsets(img, df)
    assert np.array_equal(_rows(out, 'species_col')['y1'].values,
                          _rows(df, 'species_col')['y1'].values)
    assert any('species_col' in w and '少ない' in w for w in warns)


def test_傾き補正のあとに残るずれだけを測る():
    x_comp = list(range(200, 200 + 60 * 16 + 1, 60))            # 16 列，幅 960 px
    slope = 12 / 960
    img = _sheet(EDGES, name_shift=10, x_comp=x_comp, slope=slope, size=(1400, 700))
    df = _df(EDGES, x_comp=x_comp)
    df1, w1 = row_skew.fix_skew(img, df)
    assert w1 and '傾き' in w1[0]
    out, warns = row_track.fix_offsets(img, df1)
    s = _rows(out, 'sname')
    dark = row_track.ink.binarize(img)
    # 学名の字 (16 px) がセル (34 px) の中に収まる: セルの上辺・下辺に黒画素が無い
    for cell in s.itertuples(index=False):
        band = dark[:, int(cell.x1):int(cell.x2)]
        assert band[int(cell.y1)].sum() == 0 and band[int(cell.y2) - 1].sum() == 0


def test_cut_countは字の上を通る境の数():
    img = _sheet(EDGES)
    dark = row_track.ink.binarize(img)
    centers = np.array([(a + b) / 2.0 for a, b in zip(EDGES[:-1], EDGES[1:])])
    assert row_track.cut_count(dark, 0, 150, centers) == 12
    assert row_track.cut_count(dark, 0, 150, np.array(EDGES, dtype=float)) == 0


def test_beat_rowsは薄くて消えた行を拍子で補う():
    cys = [117.0 + 34 * i for i in range(12) if i != 5]
    r = row_track.beat_rows(cys, 34.0)
    assert r['n_est'] == 12 and abs(r['p'] - 34) <= 1 and r['cv'] < 0.05


def test_count_rowsは上の見出し行を学名の字で足す():
    img = _sheet(EDGES, headings=(1,), miss_dots=(6,))
    dark = row_track.ink.binarize(img)
    r = row_track.count_rows(dark, _df(EDGES), 34.0)
    assert r['n_est'] == 11 and r['above'] == 1 and r['n_full'] == 12


def test_check_beatsは3行以上の食い違いを知らせる():
    img = _sheet(EDGES)
    big = list(range(100, 100 + 27 * 15 + 1, 27))              # 15 行の格子 (紙面は 12 行)
    warns = row_track.check_beats(img, _df(big))
    assert warns and '食い違う' in warns[0] and '15' in warns[0] and '12' in warns[0]
    near = list(range(100, 100 + 31 * 13 + 1, 31))             # 13 行なら黙る
    assert row_track.check_beats(img, _df(near)) == []


def test_左端の列だけ2段の値でも中央値で数える():
    img = _sheet(EDGES, half_left=range(1, 13))
    dark = row_track.ink.binarize(img)
    r = row_track.count_rows(dark, _df(EDGES), 34.0)
    assert r['n_full'] == 12


def test_clean_rulesは長い横線と縦線だけ消す():
    img = Image.new('L', (300, 300), 255)
    px = img.load()
    _fill(px, 10, 100, 50, 52)            # 90 px の横線 (下線・罫線)
    _fill(px, 120, 140, 50, 52)           # 20 px の下線 (学名だけの行)
    _fill(px, 200, 202, 20, 200)          # 180 px の縦線 (縦罫線)
    _fill(px, 250, 252, 100, 120)         # 20 px の縦棒 (「1」)
    dark = row_track.ink.binarize(img)
    out = row_track.clean_rules(dark, 0, 300, 0, 300, 34.0)
    assert not out[50:52, 10:100].any() and out[50:52, 120:140].all()
    assert not out[20:200, 200:202].any() and out[100:120, 250:252].all()


# ---- 実データ (一時ディレクトリにあるときだけ) ------------------------------------

WORK = os.environ.get('COMPTEA_WORK', '')


def _real(name):
    d = os.path.join(WORK, name)
    if not WORK or not os.path.exists(os.path.join(d, 'located.csv')):
        pytest.skip('実データが無い (COMPTEA_WORK)')
    loc = pd.read_csv(os.path.join(d, 'located.csv'))
    det = pd.read_csv(os.path.join(d, 'detect.csv'))
    return Image.open(det['source_image'].iloc[0]), loc


def _dy(warns, cls):
    for w in warns:
        if f'列 {cls}' in w and 'ずらした' in w:
            return float(w.split('を ')[1].split(' px')[0])
    return None


@pytest.mark.slow
@pytest.mark.parametrize('name, cls, gain', [
    ('s01115_14_p1', 'sname', 0.5),          # 字を割る境 317 → 116 (2026-09-09)
    ('s01115_14_p1', 'species_col', 0.3),    # 286 → 187
    ('s01115_07_p1', 'sname', 0.3),          # 148 → 87
    ('s01115_07_p1', 'species_col', 0.5),    # 131 → 49
    ('s01115_04_p2', 'sname', 0.1),          # 83 → 71
])
def test_実データのタイプ打ちで字を割る境が減る(name, cls, gain):
    img, loc = _real(name)
    dark = row_track.ink.binarize(img)
    out, warns = row_track.fix_offsets(img, loc)
    assert _dy(warns, cls) is not None
    a, b = loc[loc.obj_name == cls], out[out.obj_name == cls]
    x1, x2 = int(a.x1.min()), int(a.x2.max())
    before = row_track.cut_count(dark, x1, x2, a.y1.values)
    after = row_track.cut_count(dark, x1, x2, b.y1.values)
    assert after <= before * (1 - gain)


@pytest.mark.slow
@pytest.mark.parametrize('name', ['s01114_kinki_017', 's01114_kinki_019', 's01114_kinki_010-1'])
def test_実データの活字は悪化しない(name):
    img, loc = _real(name)
    dark = row_track.ink.binarize(img)
    out, _ = row_track.fix_offsets(img, loc)
    for cls in row_track.SHIFT_CLASSES:
        a, b = loc[loc.obj_name == cls], out[out.obj_name == cls]
        if a.empty:
            continue
        x1, x2 = int(a.x1.min()), int(a.x2.max())
        assert (row_track.cut_count(dark, x1, x2, b.y1.values)
                <= row_track.cut_count(dark, x1, x2, a.y1.values))


@pytest.mark.slow
def test_実データで格子の余分な行を拍が見つける():
    img, loc = _real('s01115_13_p2')
    warns = row_track.check_beats(img, loc)
    assert warns and '66' in warns[0]
    img, loc = _real('s01115_04_p2')
    assert row_track.check_beats(img, loc) == []
