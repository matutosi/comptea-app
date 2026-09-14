"""1 枚に載る別々の表を見分ける段 (table_split.py) と，組んではいけない
ものを外す段 (filters.py)

2026-09-15 に足した．それまで `table_split` は公開関数 5 つとも，
`filters` は 7 つ中 6 つが的で呼ばれていなかった (工程の正の経路なのに)．

**どちらも「表を 1 つにまとめない」ための段**で，間違えると別々の表の地点が
1 つの番号体系に混ざる (取り返しがつかない)．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import filters, table_split


def _det(rows):
    """検出の表 (`detect.csv` と同じ形の最小版)"""
    return pd.DataFrame(
        [{'source_image': 'a.png', 'obj_name': o, 'confidence': c,
          'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
         for o, x1, y1, x2, y2, c in rows])


def _table(x0, *, head=True, name='sname'):
    """x0 を左端とする 1 つの表ぶんの検出"""
    rows = [(name, x0, 300, x0 + 200, 2000, 0.9)]
    if head:
        rows.append(('header', x0, 100, x0 + 900, 280, 0.9))
    rows += [('col', x0 + 220 + i * 100, 100, x0 + 300 + i * 100, 2000, 0.8)
             for i in range(6)]
    rows += [('row', x0, 320 + i * 60, x0 + 900, 370 + i * 60, 0.8)
             for i in range(20)]
    return rows


# --- 左右に並んだ表 --------------------------------------------------------

def test_表が1つなら分けない():
    out, warns = table_split.split_side_by_side(_det(_table(100)))
    assert len(out) == 1 and warns == []


def test_種名の列も表頭も2つなら分ける():
    """s01115_02 の型 (Tab.8 と Tab.9 が左右に並ぶ．仕切りは 23 px)"""
    out, warns = table_split.split_side_by_side(
        _det(_table(100) + _table(1200)))
    assert len(out) == 2
    assert any('左右に 2 つ並んでいる' in w for w in warns)
    # 左の表の組成部を割らない (切れ目は**次の表の種名の列の左端**)
    assert float(out[0]['x2'].max()) >= 1000


def test_表頭が1つなら折り返しとみなす():
    """**1 つの表を左右 2 段に折り返した紙面**は分けない (33 枚中 11 枚)"""
    out, warns = table_split.split_side_by_side(
        _det(_table(100) + _table(1200, head=False)))
    assert len(out) == 1
    assert any('折り返した紙面' in w for w in warns)


def test_切れ目は種名の列の左端():
    cuts = table_split.side_by_side_cuts(_det(_table(100) + _table(1200)))
    assert len(cuts) == 1
    assert 1150 <= cuts[0] <= 1250


# --- 表頭の中の種名の列 ----------------------------------------------------

def test_表頭の中の種名の列は捨てる():
    """凡例や表題の字を種名の列と見た誤検出．残すと段が余分に増える"""
    rows = _table(100) + [('sname', 150, 120, 400, 260, 0.5)]
    out, n = table_split.drop_stray_name_cols(_det(rows))
    assert n == 1
    assert len(out[out.obj_name == 'sname']) == 1


def test_本物の種名の列は残す():
    out, n = table_split.drop_stray_name_cols(_det(_table(100)))
    assert n == 0 and len(out) == len(_det(_table(100)))


# --- 表頭の外へ出た項目行 --------------------------------------------------

def test_表頭の外の項目行は捨てる():
    """残すと組成部の行が表頭の値として二重に切られる (18_p2)"""
    rows = _table(100) + [('plot_row', 100, 120, 1000, 160, 0.9),
                          ('plot_row', 100, 1500, 1000, 1550, 0.9)]
    out, warns = table_split.drop_stray_plot_rows(_det(rows), heads=False)
    assert len(out[out.obj_name == 'plot_row']) == 1
    assert warns


def test_項目行が表頭の中だけなら触らない():
    rows = _table(100) + [('plot_row', 100, 120, 1000, 160, 0.9)]
    out, warns = table_split.drop_stray_plot_rows(_det(rows), heads=False)
    assert len(out[out.obj_name == 'plot_row']) == 1
    assert warns == []


# --- 組んではいけないもの (filters) ----------------------------------------

def test_表頭も種名の列も無ければ切れ端():
    rows = [('table', 0, 0, 500, 500, 0.9), ('row', 0, 10, 500, 60, 0.8),
            ('col', 0, 0, 100, 500, 0.8)]
    assert filters.looks_like_fragment(_det(rows)) is True


def test_種名の列があれば切れ端ではない():
    assert filters.looks_like_fragment(_det(_table(100))) is False


def test_裏の取れない表頭は捨てる():
    """項目行にも項目名の列にも重ならない `header` は，種群を覆った誤検出"""
    rows = (_table(100)
            + [('plot_row', 100, 120, 1000, 160, 0.9)]      # 本物の裏づけ
            + [('header', 100, 900, 1000, 1400, 0.4)])      # 種群を覆った箱
    out, n = filters.drop_unsupported_headers(_det(rows))
    assert n == 1
    assert len(out[out.obj_name == 'header']) == 1


def test_裏の取れた表頭が無ければ触らない():
    """**1 地点の表の表頭は流し込み**で，項目行として検出されない"""
    rows = _table(100) + [('header', 100, 900, 1000, 1400, 0.4)]
    out, n = filters.drop_unsupported_headers(_det(rows))
    assert n == 0
    assert len(out[out.obj_name == 'header']) == 2


def test_名前とクラスで絞る():
    df = _det(_table(100))
    got = filters.filter_results(df, 'a.png', 'row')
    assert len(got) == 20
    assert filters.filter_results(df, 'b.png', 'row').empty


def test_重なった検出は確信度の低い方を捨てる():
    rows = [('row', 0, 100, 500, 160, 0.9), ('row', 0, 105, 500, 158, 0.4)]
    out = filters.remove_dup_range(_det(rows), 'row', 0.8)
    assert len(out) == 1
    assert float(out['confidence'].iloc[0]) == 0.9


def test_離れた検出は両方残す():
    rows = [('row', 0, 100, 500, 160, 0.9), ('row', 0, 300, 500, 360, 0.4)]
    out = filters.remove_dup_range(_det(rows), 'row', 0.8)
    assert len(out) == 2
