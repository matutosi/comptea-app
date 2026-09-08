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


def test_種名側が下にずれて打たれた紙面でも境は共有し組成の点を割らない():
    # 組成部の「・」は行の中央，種名の字は 10 px 下に打たれている(14_p1 型)．
    # 境は全クラスで共有し，位相は組成部の谷に置く(2026-09-08 ユーザ指示で共有に決めた．
    # 種名側との折衷は測って取り下げた)．格子は「・」を割っている状態から始める
    edges = list(range(100, 100 + 34 * 12 + 1, 34))
    grid_edges = [float(e + 17) for e in edges]          # 「・」(e+14..e+20)を割る位置
    comp = _grid(grid_edges, [300, 340, 380, 420], 'comp')
    sname = _grid(grid_edges, [0, 200], 'sname')
    df = pd.concat([sname, comp], ignore_index=True)
    img = _page(edges, name_shift=10)
    out, warns = row_heights.fix_row_heights(img, df)
    assert any('まとめて' in w for w in warns)
    c, s = _rows(out, 'comp'), _rows(out, 'sname')
    assert np.array_equal(np.sort(c['y1'].unique()), np.sort(s['y1'].unique()))  # 共有
    dark = row_heights.ink.binarize(img)
    prof_c = row_heights._profile(dark, 300, 420)
    for y in np.sort(c['y1'].unique())[1:]:
        assert prof_c[int(y)] == 0                           # 「・」は割らない


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


def _det(x1, y1, x2, y2, obj_name):
    return dict(obj_name=obj_name, confidence=0.9, x1=x1, y1=y1, x2=x2, y2=y2,
                source_image='x', model='m')


def test_下端は種名の箱まで組成の字がある行だけ足す():
    # 印字は 12 行．格子は上 8 行で止まっている(col の箱が途中で止まった 07_p2 型)．
    # 表の下には「出現1回の種」の見出しが種名側だけにある
    print_edges = list(range(100, 100 + 34 * 12 + 1, 34))
    short = print_edges[:9]
    df = pd.concat([_grid(short, [0, 150], 'sname'),
                    _grid(short, [160, 290], 'species_col'),
                    _grid(short, [300, 340, 380, 420], 'comp')], ignore_index=True)
    img = _page(print_edges, size=(500, 800))             # 学名は x 10-150
    px = img.load()
    for ya, yb in zip(print_edges[:-1], print_edges[1:]):  # 和名は x 170-280
        c = (ya + yb) // 2
        for x in range(170, 280):
            for y in range(c - 8, c + 8):
                px[x, y] = 0
    y_head = print_edges[-1] + 20                       # 見出しは学名の列だけ
    for x in range(10, 150):
        for y in range(y_head, y_head + 16):
            px[x, y] = 0
    det = pd.DataFrame([
        _det(300, 100, 420, print_edges[8], 'col'),          # 途中で止まった col
        _det(0, 100, 150, y_head + 30, 'sname'),             # 種名の箱は見出しまで
        _det(160, 100, 290, print_edges[-1], 'species_col'),
    ])
    out, warns = row_heights.fix_row_heights(img, df, df_det=det)
    assert any('上下端' in w for w in warns)
    c = _rows(out, 'comp')
    assert c['row'].nunique() == 12                      # 4 行足された
    assert (np.abs(np.sort(c['y1'].unique()) - np.array(print_edges[:-1])) <= 4).all()
    assert c['y2'].max() <= y_head                       # 見出しの行は足さない
    s = _rows(out, 'sname')
    assert s['row'].nunique() == 12                      # 種名も同じ行で組み直る


def test_組成の点が薄くて消えた最終行も学名と和名があれば残す():
    # 14_p1 型: 最終行の「・」が二値化で消える．学名・和名は有る
    print_edges = list(range(100, 100 + 34 * 10 + 1, 34))
    df = pd.concat([_grid(print_edges, [0, 150], 'sname'),
                    _grid(print_edges, [160, 290], 'species_col'),
                    _grid(print_edges, [300, 340, 380, 420], 'comp')], ignore_index=True)
    img = _page(print_edges, size=(500, 800))
    px = img.load()
    for ya, yb in zip(print_edges[:-1], print_edges[1:]):
        c = (ya + yb) // 2
        for x in range(170, 280):
            for y in range(c - 8, c + 8):
                px[x, y] = 0
    c = (print_edges[-2] + print_edges[-1]) // 2          # 最終行の「・」を消す
    for x in range(300, 420):
        for y in range(c - 4, c + 5):
            px[x, y] = 255
    det = pd.DataFrame([_det(300, 100, 420, print_edges[-1], 'col'),
                        _det(0, 100, 150, print_edges[-1], 'sname')])
    out, warns = row_heights.fix_row_heights(img, df, df_det=det)
    assert _rows(out, 'comp')['row'].nunique() == 10     # 落とされない


def test_地点が2列で隙間の無い表でも下端を足す():
    # 07_p2 型: 列の境が 1 本しか無く，種名と組成のあいだに空白の帯も無い．
    # body_extent_ink はこの形で格子の下端に固定していた(26 表が例外で落ちた回もある)
    print_edges = list(range(100, 100 + 34 * 12 + 1, 34))
    short = print_edges[:9]
    df = pd.concat([_grid(short, [0, 300], 'sname'),
                    _grid(short, [300, 340, 380], 'comp')], ignore_index=True)
    img = Image.new('L', (500, 800), 255)
    px = img.load()
    for ya, yb in zip(print_edges[:-1], print_edges[1:]):
        c = (ya + yb) // 2
        for x in (320, 360):
            for y in range(c - 3, c + 4):
                px[x, y] = 0
        for x in range(10, 296):                 # 種名は組成の直前まで(隙間なし)
            for y in range(c - 8, c + 8):
                px[x, y] = 0
    det = pd.DataFrame([_det(300, 100, 380, print_edges[8], 'col'),
                        _det(0, 100, 300, print_edges[-1] + 4, 'sname')])
    out, warns = row_heights.fix_row_heights(img, df, df_det=det)
    assert any('上下端' in w for w in warns)
    assert _rows(out, 'comp')['row'].nunique() == 12


def test_直下の流し込みは行として足さない():
    # 04_p2 型: 種名の箱が流し込みまで伸び，流し込みは幅いっぱいに字が流れる
    print_edges = list(range(100, 100 + 34 * 8 + 1, 34))
    df = pd.concat([_grid(print_edges, [0, 200], 'sname'),
                    _grid(print_edges, [300, 340, 380, 420], 'comp')], ignore_index=True)
    img = _page(print_edges, size=(500, 800))
    px = img.load()
    y0 = print_edges[-1] + 2
    for k in range(6):                              # 流し込み 6 行(全幅に字)
        for x in range(10, 420, 3):
            for y in range(y0 + 34 * k + 8, y0 + 34 * k + 26):
                px[x, y] = 0
    det = pd.DataFrame([_det(300, 100, 420, print_edges[-1], 'col'),
                        _det(0, 100, 200, y0 + 34 * 6, 'sname')])
    out, warns = row_heights.fix_row_heights(img, df, df_det=det)
    assert _rows(out, 'comp')['row'].nunique() == 8         # 1 行も足されない


def test_組成の字が無い末尾の行は落とす():
    print_edges = list(range(100, 100 + 34 * 10 + 1, 34))
    extra = print_edges + [print_edges[-1] + 34]         # 余分な 1 行(組成は空)
    df = pd.concat([_grid(extra, [0, 200], 'sname'),
                    _grid(extra, [300, 340, 380, 420], 'comp')], ignore_index=True)
    img = _page(print_edges, size=(500, 800))
    det = pd.DataFrame([_det(300, 100, 420, extra[-1], 'col')])
    out, warns = row_heights.fix_row_heights(img, df, df_det=det)
    assert any('落とした' in w for w in warns)
    assert _rows(out, 'comp')['row'].nunique() == 10


def test_検出を渡さなければ端は触らない():
    print_edges = list(range(100, 100 + 34 * 12 + 1, 34))
    short = print_edges[:9]
    df = _grid(short, [300, 340, 380, 420], 'comp')
    out, warns = row_heights.fix_row_heights(_page(print_edges, size=(500, 800)), df)
    assert out is df
