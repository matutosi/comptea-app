"""和名と階層の境を 1 本にする (col_edges.fix_name_layer_edge)

階層は和名の**右**にあり (81 段すべて)，両者の x が重なるのは 8 段だけ，しかも 3〜21 px
(2026-09-09 に実データで確認．ユーザ提案)．いまは「和名の右端」と「階層の左端」が
別々に決まるので，字を割る行が 685 ある．1 本の境にして，いまの境の ±1.5 行の窓で
**字を割る行がいちばん少ない x** に置くと 33 になる (悪化する段は無い)．
"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import col_edges


EDGES = list(range(100, 100 + 34 * 12 + 1, 34))
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


def _grid(x_edges, obj_name, col0=1, edges=EDGES):
    rows = []
    for c, (xa, xb) in enumerate(zip(x_edges[:-1], x_edges[1:]), col0):
        for r, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
            rows.append(dict(x1=float(xa), x2=float(xb), y1=float(ya), y2=float(yb),
                             obj_name=obj_name, note='', block=1, row=r, col=c))
    return pd.DataFrame(rows)


def _df(ja=(300, 500), layer=(500, 560), comp=(600, 660, 720)):
    return pd.concat([
        _grid([0, 290], 'sname'),
        _grid(list(ja), 'species_col'),
        _grid(list(layer), 'layer'),
        _grid(list(comp), 'comp', col0=2),
    ], ignore_index=True)


def _sheet(ja_text=(310, 540), sym=(548, 580), size=(900, 600)):
    """和名が右へはみ出し，階層の記号がその右にある紙面"""
    img = Image.new('L', size, 255)
    px = img.load()
    for ya, yb in zip(EDGES[:-1], EDGES[1:]):
        c = int((ya + yb) // 2)
        _text(px, 10, 280, c - 8, c + 8)                 # 学名
        _text(px, ja_text[0], ja_text[1], c - 8, c + 8)  # 和名
        _fill(px, sym[0], sym[1], c - 8, c + 8)          # 階層の記号
        for xa, xb in zip((600, 660), (660, 720)):
            xc = (xa + xb) // 2
            _fill(px, xc - 2, xc + 3, c - 3, c + 4)
    return img


def _edge(df):
    ja = df[df.obj_name == 'species_col']
    lay = df[df.obj_name == 'layer']
    return float(ja.x2.max()), float(lay.x1.min())


def test_和名と階層の境が1本になる():
    img = _sheet()
    df = _df()
    out, warns = col_edges.fix_name_layer_edge(img, df)
    ja_r, lay_l = _edge(out)
    assert ja_r == lay_l                       # 共有の境
    assert warns and any('和名と階層の境' in w for w in warns)


def test_境は字を割らない位置に来る():
    img = _sheet(ja_text=(310, 540), sym=(548, 580))
    out, _w = col_edges.fix_name_layer_edge(img, _df())
    ja_r, _l = _edge(out)
    assert 540 <= ja_r <= 548                  # 和名の右端と記号の左端のあいだ


def test_窓の外へは動かさない():
    # 遠くにもっと良い位置があっても，いまの境の ±1.5 行までしか動かさない
    img = _sheet(ja_text=(310, 400), sym=(548, 580))
    df = _df(ja=(300, 500), layer=(500, 560))
    out, _w = col_edges.fix_name_layer_edge(img, df)
    ja_r, _l = _edge(out)
    assert abs(ja_r - 500) <= 1.5 * PITCH + 1


def test_階層が無ければ和名の右端を字まで広げる():
    img = _sheet(ja_text=(310, 560), sym=(0, 0))
    df = pd.concat([_grid([0, 290], 'sname'), _grid([300, 500], 'species_col'),
                    _grid([600, 660, 720], 'comp', col0=2)], ignore_index=True)
    out, warns = col_edges.fix_name_layer_edge(img, df)
    ja_r = float(out[out.obj_name == 'species_col'].x2.max())
    assert 555 <= ja_r <= 600                  # 字の右端まで．組成部は越えない
    assert warns


def test_組成部は越えない():
    img = _sheet(ja_text=(310, 700), sym=(0, 0))   # 和名が組成部まで届く紙面
    df = pd.concat([_grid([0, 290], 'sname'), _grid([300, 500], 'species_col'),
                    _grid([600, 660, 720], 'comp', col0=2)], ignore_index=True)
    out, _w = col_edges.fix_name_layer_edge(img, df)
    ja_r = float(out[out.obj_name == 'species_col'].x2.max())
    assert ja_r <= 600


def test_行と他の列は変わらない():
    img = _sheet()
    df = _df()
    out, _w = col_edges.fix_name_layer_edge(img, df)
    assert len(out) == len(df) and out.row.nunique() == df.row.nunique()
    for cls in ('sname', 'comp'):
        a = df[df.obj_name == cls].sort_index()[['x1', 'x2', 'y1', 'y2']].values
        b = out[out.obj_name == cls].sort_index()[['x1', 'x2', 'y1', 'y2']].values
        assert np.array_equal(a, b)


def test_和名が無ければ何もしない():
    img = _sheet()
    df = pd.concat([_grid([0, 290], 'sname'), _grid([600, 660, 720], 'comp', col0=2)],
                   ignore_index=True)
    out, warns = col_edges.fix_name_layer_edge(img, df)
    assert out is df and warns == []


def _sheet_heading(size=(900, 600)):
    """見本 (examples/sample.jpg) と同じ形の紙面

    - 1 行目は種群の見出しで，和名から記号の列まで字が続く
    - 記号は「S・K」(S と ・K のあいだに隙間) か「K」だけ
    - 記号の右と組成部のあいだは空いている
    """
    img = Image.new('L', size, 255)
    px = img.load()
    for r, (ya, yb) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        c = int((ya + yb) // 2)
        if r == 0:
            _text(px, 310, 470, c - 8, c + 8, gap=2)      # 見出し
            continue
        _text(px, 310, 400, c - 8, c + 8)                 # 和名
        if r % 3 == 1:
            _fill(px, 440, 452, c - 8, c + 8)             # S
            _fill(px, 463, 480, c - 8, c + 8)             # ・K
        else:
            _fill(px, 463, 480, c - 8, c + 8)             # K
    return img


def test_見出しの行が横切っても記号を和名に入れない():
    """2026-09-18．見出しの行のせいで和名と記号のあいだに空白の帯ができず，
    記号の右の空きが「いちばん広い帯」として選ばれ，記号がまるごと和名の列に
    入っていた (見本で階層の読みが 25 → 0)"""
    img = _sheet_heading()
    out, warns = col_edges.fix_name_layer_edge(img, _df(ja=(300, 450), layer=(450, 560)))
    ja_r, lay_l = _edge(out)
    assert ja_r == lay_l
    assert 400 <= ja_r <= 440, warns           # 和名の末尾と S のあいだ
