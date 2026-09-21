"""細い切れ端の列を隣へ併合する (layer_col._merge_slivers)"""
import numpy as np
import pandas as pd

from comptea import layer_col


def _comp(xs, rows=3, h=20):
    cells = []
    for c, (a, b) in enumerate(zip(xs[:-1], xs[1:]), 1):
        for r in range(1, rows + 1):
            cells.append(dict(obj_name='comp', col=c, row=r, block=1,
                              x1=float(a), x2=float(b),
                              y1=float(r * h), y2=float(r * h + h)))
    return pd.DataFrame(cells)


def _cover(df):
    """列が覆っている x の範囲の和"""
    comp = df[df.obj_name == 'comp'].drop_duplicates('col')
    return sorted((float(a), float(b)) for a, b in zip(comp.x1, comp.x2))


def test_続けて細い列があっても範囲を失わない():
    """2026-09-18．前の切れ端が消えたあと，次の切れ端がその消えた列を併合先に
    選ぶと，代入先が無いまま自分だけ消え，その範囲 (見本の並びでは 30-40) が無くなっていた"""
    # 幅 20・10・10・60・60・60 (中央値 40)．10 px の 2 本が切れ端．1 本目は左へ併合され，
    # 2 本目の左隣はその消えた列になる
    xs = [0, 20, 30, 40, 100, 160, 220]
    df = _comp(xs)
    dark = np.ones((200, 300), dtype=bool)   # どの列にも字がある (端でも捨てない)
    out, _w = layer_col._merge_slivers(df, dark, body_min=0.0)
    got = _cover(out)
    assert got[0][0] == 0 and got[-1][1] == 220
    # 列どうしに隙間が無い (どこかの範囲が消えていない)
    for (a1, b1), (a2, b2) in zip(got[:-1], got[1:]):
        assert b1 == a2


# --- 組み直しても印を消さない (2026-09-18) -------------------------------

def test_列を組み直しても印を引き継ぐ():
    from comptea import col_edges
    src = pd.DataFrame([dict(row=1, x1=0.0, x2=50.0, note='interpolated'),
                        dict(row=1, x1=50.0, x2=100.0, note=float('nan'))])
    notes = col_edges._notes_by_row(src)
    assert col_edges._carried_note(notes, 1, 5.0, 45.0) == 'interpolated'
    assert col_edges._carried_note(notes, 1, 55.0, 95.0) == ''
    assert col_edges._carried_note(notes, 2, 0.0, 50.0) == ''


def test_行を追っても印に足す():
    from comptea import row_track
    df = pd.DataFrame([dict(row=1, y1=0.0, y2=10.0, note='snapped'),
                       dict(row=2, y1=10.0, y2=20.0, note=float('nan'))])
    out = row_track._apply_edges(df, df['row'] > 0, [0.0, 11.0, 21.0], 'y_fitted')
    assert list(out['note']) == ['snapped;y_fitted', 'y_fitted']
