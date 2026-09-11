"""右の段の下端を左の段にそろえる (blocks.align_block_bottoms)

種名のリストを折り返した組み方で，右の段の箱が最後の行まで届かずに行が落ちていた
(kinki_060 の「シナダレススメガヤ +・2」．2026-09-10 ユーザ指摘)．
**右の段は左の段の下端まで行を足す．空振り (OCR が全部空白) でもよい** (ユーザ指示)．
"""
import pandas as pd

from comptea import blocks


PITCH = 40.0
Y0 = 100.0


def _block(n_rows, block, x0, row0=1, y0=Y0):
    rows = []
    for i in range(n_rows):
        y1, y2 = y0 + PITCH * i, y0 + PITCH * (i + 1)
        for col, (name, (a, b)) in enumerate((('sname', (x0, x0 + 300)),
                                              ('species_col', (x0 + 300, x0 + 500)),
                                              ('comp', (x0 + 500, x0 + 560))), 1):
            rows.append(dict(x1=float(a), x2=float(b), y1=y1, y2=y2, obj_name=name,
                             note='', block=block, row=row0 + i, col=col))
    rows.append(dict(x1=float(x0), x2=float(x0 + 560), y1=y0 - 50, y2=y0,
                     obj_name='header', note='', block=block, row=0, col=1))
    return pd.DataFrame(rows)


def _two(n1, n2, y0_2=Y0):
    return pd.concat([_block(n1, 1, 0), _block(n2, 2, 600, row0=n1 + 1, y0=y0_2)],
                     ignore_index=True)


def test_右の段が短ければ左の段の行を写して足す():
    out, n = blocks.align_block_bottoms(_two(10, 7))
    assert n == 3
    b2 = out[(out.block == 2) & (out.obj_name == 'comp')]
    assert len(b2) == 10 and b2.y2.max() == Y0 + PITCH * 10
    assert len(out[(out.block == 2) & (out.obj_name == 'sname')]) == 10
    added = out[(out.block == 2) & (out.y1 >= Y0 + PITCH * 7)]
    assert (added.note == 'interpolated').all()
    assert (added.x1 >= 600).all()                      # 右の段の列の位置のまま
    rows = sorted(b2.row)
    assert rows == list(range(rows[0], rows[0] + 10))    # 行番号は続き番号 (表頭の行も数える)
    assert len(out[out.block == 1]) == len(_block(10, 1, 0))   # 左の段は触らない


def test_足す行は左の段の行の高さに合わせる():
    # 右の段の行が左より 10 px 下にずれている．写す行は左の段の帯で，最初だけ
    # 右の段の最後の行に続ける
    out, n = blocks.align_block_bottoms(_two(10, 7, y0_2=Y0 + 10))
    assert n == 3                                       # 左の段の行 8〜10 (中心が下端 390 より下)
    added = (out[(out.block == 2) & (out.obj_name == 'comp') & (out.note == 'interpolated')]
             .sort_values('y1'))
    assert added.y1.iloc[0] == Y0 + 10 + PITCH * 7      # 右の段の最後の行に続く
    assert added.y2.iloc[-1] == Y0 + PITCH * 10         # 左の段の下端まで


def test_右の段が長ければ触らない():
    out, n = blocks.align_block_bottoms(_two(7, 10))
    assert n == 0 and len(out) == len(_two(7, 10))


def test_半行未満の差は足さない():
    out, n = blocks.align_block_bottoms(_two(10, 10, y0_2=Y0 - 15))
    assert n == 0


def test_段が1つなら何もしない():
    out, n = blocks.align_block_bottoms(_block(10, 1, 0))
    assert n == 0 and len(out) == len(_block(10, 1, 0))


def test_列の境をまたぐ行が下に固まっていれば流し込み():
    """`flow_top_by_edges` (15_p5 型): 本体の値はセルの中，文章は幅いっぱいに流れる"""
    import numpy as np
    from comptea import body_rows as br
    dark = np.zeros((400, 600), dtype=bool)
    edges = [200, 300, 400, 500]
    for r in range(8):                             # 本体 8 行: セルの中だけ
        y = 20 + r * 30
        for x in (120, 220, 320, 420):
            dark[y + 8:y + 20, x:x + 40] = True
    for r in range(8, 11):                         # 下 3 行: 文章 (境をまたぐ)
        y = 20 + r * 30
        dark[y + 8:y + 20, 60:560] = True
    assert br.flow_top_by_edges(dark, edges, 20, 20 + 11 * 30, 30) == 20 + 8 * 30


def test_値がセルに収まっていれば切らない():
    import numpy as np
    from comptea import body_rows as br
    dark = np.zeros((400, 600), dtype=bool)
    for r in range(11):
        y = 20 + r * 30
        for x in (120, 220, 320, 420):
            dark[y + 8:y + 20, x:x + 40] = True
    y2 = 20 + 11 * 30
    assert br.flow_top_by_edges(dark, [200, 300, 400, 500], 20, y2, 30) == float(y2)
