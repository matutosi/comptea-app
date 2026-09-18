"""目視に回すセルの選り分け (pipeline/read.py の suspect_empty・reasons_for)

組成部の空セルは非出現 (`・`) が普通なので全部は回さない．回しすぎると本当に
見たいものが埋もれ，回し漏らすと読み落としが残る．どちらも黙って起きる．
"""
import pandas as pd

from comptea.pipeline import read


def _cells(texts):
    return pd.DataFrame([dict(cell_id=i, obj_name='comp', text=t)
                         for i, t in enumerate(texts, 1)])


def test_空セルが少なければ全部を回す():
    df = _cells(['', None, '+', ''])
    ratios = {1: 0.001, 2: 0.5, 3: 0.2, 4: 0.001}
    assert read.suspect_empty(df, ratios) == {1, 2, 4}


def test_空セルが多ければ濃いものだけ回す():
    df = _cells([''] * 6)
    ratios = {1: 0.002, 2: 0.002, 3: 0.003, 4: 0.002, 5: 0.05, 6: 0.004}
    # 中央値 0.0025 の 3 倍 (下限 0.01) を超えるものだけ
    assert read.suspect_empty(df, ratios) == {5}


def test_読めたセルは空に数えない():
    df = _cells(['+', '1・1', '', '', '', ''])
    ratios = {i: 0.9 for i in range(1, 7)}
    assert read.suspect_empty(df, ratios) <= {3, 4, 5, 6}


def _row(**kw):
    base = dict(cell_id=1, obj_name='comp', text='+', status='OK', note=None,
                suggest=None)
    base.update(kw)
    return base


def test_組成の空セルは濃いときだけ回す():
    assert read.reasons_for(_row(text=''), suspect=set()) == []
    assert read.reasons_for(_row(text=''), suspect={1}) == ['empty']


def test_和名の空セルは常に回す():
    assert read.reasons_for(_row(obj_name='species_col', text=''), set()) == ['empty']


def test_理由を並べる():
    got = read.reasons_for(_row(status='Need Check', note='interpolated',
                                suggest='ヤマブドウ'), set())
    assert got == ['short_name', 'need_check', 'note']


def test_AIに読ませるときは全セルを回す():
    assert read.reasons_for(_row(), set(), reader='ai') == ['unread']
    assert read.reasons_for(_row(), set(), reader='both') == ['crosscheck']
    assert read.reasons_for(_row(), set(), reader='easyocr') == []
