"""表頭の帯は値の行ごとに切り，項目名の無い帯は前の項目へ合成する (2026-09-10 方針転換)

- header_lines.bands_from_value_lines: 値の行と行のあいだを境にする (1 行 1 帯)
- plot_table.plot_table: 項目名が空の帯を前の項目の続きとして値を「/」でつなぐ
"""
import numpy as np
import pandas as pd

from comptea import header_lines as hl
from comptea import plot_table as pt


def test_値の行のあいだが境になる():
    vs = [(112, 100, 125), (152, 140, 165), (192, 180, 205)]
    edges = hl.bands_from_value_lines(vs, 90, 230)
    assert list(edges) == [90.0, 132.5, 172.5, 230.0]


def _cells(rows):
    """rows: (y1, 項目名, [地点ごとの値])"""
    out = []
    for y1, name, vals in rows:
        out.append(dict(obj_name='header_item_ja', x1=0.0, y1=float(y1), corrected=name))
        for i, v in enumerate(vals):
            out.append(dict(obj_name='header_value', x1=100.0 + 50 * i, y1=float(y1),
                            corrected=v))
    return pd.DataFrame(out)


def test_項目名の無い帯は前の項目の続きになる():
    # 調査年月日の値が 3 行 ('83 / 6 / 8)．2・3 行目の帯は項目名が空
    df = _cells([(100, '通し番号', ['1', '2']),
                 (140, '調査年月日', ["'83", "'83"]),
                 (180, '', ['6', '7']),
                 (220, '', ['8', '15']),
                 (260, '海抜高', ['360', '450'])])
    res = pt.plot_table(df)
    assert list(res['plot']) == [1, 2]
    assert str(res.loc[0, 'altitude']) == '360'
    joined = str(res.loc[0, 'date'])
    assert '83' in joined and '6' in joined and '8' in joined
    assert any('合成' in w for w in res.attrs['warnings'])


def test_項目名が読めない帯は続きにしない():
    df = _cells([(100, '通し番号', ['1', '2']),
                 (140, 'ｘｘｘ', ['9', '9']),          # 判別できない項目名
                 (180, '海抜高', ['360', '450'])])
    res = pt.plot_table(df)
    assert 'altitude' in res.columns and 'plot_no' in res.columns
    assert any('判別できない' in w for w in res.attrs['warnings'])


def test_項目名の行と値の行のずれの中央値():
    # 項目名が値より 15 px 下に組まれている (22_p3 の型)
    values = [(100, 90, 110), (140, 130, 150), (180, 170, 190), (220, 210, 230)]
    items = [(115, 105, 125), (156, 146, 166), (194, 184, 204)]
    assert 14 <= hl.name_offset(items, values, 40.0) <= 16
    assert hl.name_offset(items[:2], values, 40.0) == 0.0        # 組が 3 つ未満
    assert hl.name_offset([], values, 40.0) == 0.0
