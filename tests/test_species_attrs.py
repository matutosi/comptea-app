"""行ごとの種名と階層を取り出す段 (comp_table.py)

2026-09-15 に足した．`species_attrs`・`split_bad_layers`・
`fill_sname_from_jname`・`to_wide` が的で呼ばれていなかった．

**縦持ちの表を受け取る側は，`layer` が `T1 T2 S K H M` しか取らないものとして
扱える**ようにしてある — その約束をここで守る．
"""
import numpy as np
import pandas as pd
import pytest

from comptea import comp_table


def _ocr(rows):
    """段階 2 の読みの表 (obj_name・row・col・corrected・status)"""
    return pd.DataFrame(
        [{'obj_name': o, 'row': r, 'col': c, 'corrected': v, 'status': s}
         for o, r, c, v, s in rows])


# --- 行ごとの属性 ----------------------------------------------------------

def test_和名と学名と階層を行ごとに取り出す():
    df = _ocr([('species_col', 0, 0, 'アカマツ', 'OK'),
               ('sname', 0, 0, 'Pinus densiflora', 'OK'),
               ('layer', 0, 0, 'T1', 'OK')])
    got = comp_table.species_attrs(df)
    assert list(got['row_no']) == [0]
    assert got['j_name'].iloc[0] == 'アカマツ'
    assert got['s_name'].iloc[0] == 'Pinus densiflora'
    assert got['layer'].iloc[0] == 'T1'


def test_読みが無ければ空():
    got = comp_table.species_attrs(_ocr([('comp', 0, 0, '2・2', 'OK')]))
    assert got.empty


def test_同じ行に2つあれば左を採る():
    df = _ocr([('species_col', 0, 1, '右', 'OK'),
               ('species_col', 0, 0, '左', 'OK')])
    got = comp_table.species_attrs(df)
    assert got['j_name'].iloc[0] == '左'


def test_要確認の階層は_layer_raw_へ回す():
    """`layer` には**読めたものだけ**を入れる (2026-09-07)"""
    df = _ocr([('layer', 0, 0, 'So', 'Need Check'),
               ('layer', 1, 0, 'S', 'OK')])
    got = comp_table.species_attrs(df).set_index('row_no')
    assert got.loc[1, 'layer'] == 'S'
    assert pd.isna(got.loc[0, 'layer'])
    assert got.loc[0, 'layer_raw'] == 'So'


# --- 階層として読めない値を分ける ------------------------------------------

def test_階層らしくない値は_layer_raw_へ移す():
    attrs = pd.DataFrame([{'row_no': 0, 'layer': 'S'},
                          {'row_no': 1, 'layer': 'S;O'}])
    got = comp_table.split_bad_layers(attrs)
    assert got['layer'].iloc[0] == 'S'
    assert pd.isna(got['layer'].iloc[1])
    assert got['layer_raw'].iloc[1] == 'S;O'


def test_階層の列が無ければ何もしない():
    attrs = pd.DataFrame([{'row_no': 0, 'j_name': 'アカマツ'}])
    got = comp_table.split_bad_layers(attrs)
    assert list(got.columns) == ['row_no', 'j_name']


def test_空の階層は移さない():
    attrs = pd.DataFrame([{'row_no': 0, 'layer': ''},
                          {'row_no': 1, 'layer': None}])
    got = comp_table.split_bad_layers(attrs)
    assert 'layer_raw' not in got.columns or got['layer_raw'].isna().all()


# --- 学名を和名から補う ----------------------------------------------------

def test_学名が空の行だけ補う():
    """**空の行だけ**．印字されている学名は触らない (2026-09-01 決定)

    引ける学名はいまの分類の名前なので，古い資料の印字とは食い違うことが
    あり，置き換えると「印字されていたもの」が分からなくなる．

    返るのは (attrs, 補った row_no の集合, 決められなかった件数)．
    """
    attrs = pd.DataFrame([{'row_no': 0, 'j_name': 'アカマツ', 's_name': ''},
                          {'row_no': 1, 'j_name': 'アカマツ',
                           's_name': 'Pinus thunbergii'}])
    got, filled, _n = comp_table.fill_sname_from_jname(attrs)
    assert filled == {0}
    assert got['s_name'].iloc[0]                    # 補われた
    assert got['s_name'].iloc[1] == 'Pinus thunbergii'


def test_辞書に無い和名は補えない():
    attrs = pd.DataFrame([{'row_no': 0, 'j_name': 'ズイナ群集', 's_name': ''}])
    got, filled, _n = comp_table.fill_sname_from_jname(attrs)
    assert filled == set()
    assert not str(got['s_name'].iloc[0] or '').strip()


def test_列が無ければ何もしない():
    attrs = pd.DataFrame([{'row_no': 0, 'layer': 'S'}])
    got, filled, n = comp_table.fill_sname_from_jname(attrs)
    assert filled == set() and n == 0
    assert list(got.columns) == ['row_no', 'layer']


# --- 見た目に戻す ----------------------------------------------------------

def _long(rows):
    return pd.DataFrame(
        [{'row_no': r, 'j_name': j, 's_name': '', 'layer': l, 'plot': p,
          'cover': c, 'sociability': s}
         for r, j, l, p, c, s in rows])


def test_地点を列に戻す():
    got = comp_table.to_wide(_long([(0, 'アカマツ', 'T1', 1, '3', '3'),
                                    (0, 'アカマツ', 'T1', 2, '2', '1')]))
    assert 1 in got.columns and 2 in got.columns
    assert got[1].iloc[0] == '3・3'                 # 被度・群度


def test_群度が1なら被度だけ():
    got = comp_table.to_wide(_long([(0, 'アカマツ', 'T1', 1, '+', '1')]))
    assert got[1].iloc[0] == '+'


def test_空の縦持ちはそのまま():
    empty = pd.DataFrame()
    assert comp_table.to_wide(empty).empty
