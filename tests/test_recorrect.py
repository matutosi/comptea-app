"""読み直しの読みを残し，補正だけを当て直す (pipeline/read.py)

2026-09-15 に足した．それまで `retry_cells` は `corrected` と `status` しか
更新せず，**NDLOCR-Lite が読んだ文字列を捨てていた**．そのため補正の規則を
良くしても，読み直しの読みには当て直せなかった (もう一度 OCR を回すしかなく，
1 表 4 分かかる)．
"""
import pandas as pd
import pytest

from comptea.pipeline import read as read_mod


class _Crops:
    """`read_crops` を持つ偽の読み手"""

    def __init__(self, texts):
        self.texts = texts

    def available(self):
        return True

    def read_crops(self, img, boxes, pad=0):
        return list(self.texts)[:len(boxes)]


def _df(rows):
    return pd.DataFrame(
        [{'obj_name': o, 'text': t, 'corrected': c, 'status': s, 'note': '',
          'x1': 0.0, 'y1': 0.0, 'x2': 10.0, 'y2': 10.0}
         for o, t, c, s in rows])


# --- 読みを残す ------------------------------------------------------------

def test_読めた読みを残す():
    df = _df([('comp', 'z-u', '', 'Need Check')])
    got = read_mod.retry_cells(None, df, _Crops(['2・2']))
    assert got['text_ndl'].iloc[0] == '2・2'
    assert got['status'].iloc[0] == 'OK'


def test_補正で落ちた読みも残す():
    """**あとで補正を良くしたときに当て直せる**ようにするため"""
    df = _df([('comp', 'z-u', '', 'Need Check')])
    got = read_mod.retry_cells(None, df, _Crops(['のののの']))
    assert got['text_ndl'].iloc[0] == 'のののの'
    assert got['status'].iloc[0] == 'Need Check'      # 直っていない


def test_読めなければ空のまま():
    df = _df([('comp', 'z-u', '', 'Need Check')])
    got = read_mod.retry_cells(None, df, _Crops(['']))
    assert got['text_ndl'].iloc[0] == ''


# --- 補正だけを当て直す ----------------------------------------------------

def test_保存した読みに補正を当て直す():
    df = _df([('comp', 'z-u', '', 'Need Check')])
    df['text_ndl'] = '2・2'
    got, n = read_mod.recorrect_cells(df)
    assert n == 1
    assert got['status'].iloc[0] == 'OK'
    assert got['corrected'].iloc[0]
    assert 'ndl' in got['note'].iloc[0]


def test_読めているセルには触らない():
    """OK のセルを補正の版で揺らさない"""
    df = _df([('comp', '2・2', '2;2', 'OK')])
    df['text_ndl'] = 'のののの'
    got, n = read_mod.recorrect_cells(df)
    assert n == 0
    assert got['corrected'].iloc[0] == '2;2'


def test_読みが無ければ何もしない():
    df = _df([('comp', 'z-u', '', 'Need Check')])
    df['text_ndl'] = ''
    got, n = read_mod.recorrect_cells(df)
    assert n == 0
    assert got['status'].iloc[0] == 'Need Check'


def test_列が無ければ何もしない():
    """古い作業ディレクトリ (読みを残す前のもの) でも落ちない"""
    df = _df([('comp', 'z-u', '', 'Need Check')])
    got, n = read_mod.recorrect_cells(df)
    assert n == 0
    assert len(got) == 1


def test_組成以外は触らない():
    df = _df([('species_col', 'アカマツ', '', 'Need Check')])
    df['text_ndl'] = '2・2'
    got, n = read_mod.recorrect_cells(df)
    assert n == 0
    assert got['status'].iloc[0] == 'Need Check'
