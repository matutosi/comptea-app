"""学名・和名・階層の列が重なったとき，読みの中身で落とす (2026-09-18 ユーザ方針)

x 軸はほとんど重ならないが，一部で重なり，隣の列の字がセルの端に入る．
和名はカタカナ (「◯◯属の1種」「◯◯sp」も)，学名はアルファベット，
階層は B・T・S・H・K・M などなので，読みで見分けられる．
"""
import pandas as pd
import pytest

from comptea import comp_table, correct_text
from comptea.pipeline import read


@pytest.mark.parametrize('text, name, hint', [
    ('ヤマブドウ S・K', 'ヤマブドウ', 'S;K'),
    ('ヤマブドウS・K', 'ヤマブドウ', 'S;K'),
    ('マタタビ 5', 'マタタビ', 'S'),             # S を 5 と読む
    ('イタドリ K', 'イタドリ', 'K'),
    ('Vitis ヤマブドウ', 'ヤマブドウ', None),     # 学名の切れ端
    ('ヤマブドウ', 'ヤマブドウ', None),
    ('スゲ属sp', 'スゲ属sp', None),               # sp を階層と取らない
    ('K', 'K', None),                             # 名前が無ければ外さない
])
def test_和名から隣の列の字を外す(text, name, hint):
    assert correct_text.split_overlap('species_col', text) == (name, hint)


def test_外した記号が階層として読めなければ残す():
    assert correct_text.split_overlap('species_col', 'キブシ S・が') == ('キブシ S・が', None)


@pytest.mark.parametrize('obj, text, want', [
    ('sname', 'Vitis coignetiae ヤマ', 'Vitis coignetiae'),
    ('sname', 'ヤマブドウ', 'ヤマブドウ'),         # 落とすと空なら元のまま
    ('layer', 'ウ S', 'S'),
    ('layer', 'Vitis S', 'S'),
    ('layer', 'S・K', 'S・K'),                    # 中黒は区切りなので残す
])
def test_学名と階層から隣の列の字を落とす(obj, text, want):
    assert correct_text.split_overlap(obj, text)[0] == want


def test_落としても階層の読み落としは目視に回す():
    """`S・` (K の読み落とし) を `S` として通さない歯止めは残る"""
    assert correct_text.correct_cell('layer', 'ウ S・')['status'] == 'Need Check'


def test_補正で和名の末尾の階層を覚える():
    df = pd.DataFrame({'obj_name': ['species_col', 'layer'],
                       'text': ['ヤマブドウ S・K', None]})
    got = read.apply_corrections(df)
    assert got.loc[0, 'corrected'] == 'ヤマブドウ'
    assert got.loc[0, 'layer_hint'] == 'S;K'


def test_階層が空の行にだけ和名から外した記号を入れる():
    df = pd.DataFrame([
        dict(obj_name='species_col', row=1, col=2, corrected='ヤマブドウ', layer_hint='S;K'),
        dict(obj_name='layer', row=1, col=3, corrected=None, layer_hint=None),
        dict(obj_name='species_col', row=2, col=2, corrected='キブシ', layer_hint='K'),
        dict(obj_name='layer', row=2, col=3, corrected='S', layer_hint=None),
    ])
    got = comp_table.species_attrs(df).set_index('row_no')
    assert got.loc[1, 'layer'] == 'S;K'
    assert got.loc[2, 'layer'] == 'S'              # 読めている階層は変えない
