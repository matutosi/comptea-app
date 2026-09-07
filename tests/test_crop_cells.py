"""目視に回すセルの切り出し

一覧画像を**クラスごとに分ける**と，読む側が字種を決め打ちできる
(組成部なら `5 4 3 2 1 + r ・` だけ)．字種を絞ると読みが良くなるのは
`ocr.retry_empty_comp()` で実証済み(2026-09-08 に `--by-class` を足した)．
"""
import importlib.util
import pathlib
import sys

import pytest

import conftest

SRC = pathlib.Path(conftest.ROOT) / 'cli' / 'crop_cells.py'


@pytest.fixture(scope='module')
def mod():
    spec = importlib.util.spec_from_file_location('crop_cells_under_test', SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def tiles():
    """(cell_id, 画像, ラベル, クラス)．画像は使わないので None でよい"""
    return [(1, None, 'comp r1c1', 'comp'),
            (2, None, 'species_col r1c0', 'species_col'),
            (3, None, 'comp r2c1', 'comp'),
            (4, None, 'layer r2c0', 'layer')]


def test_既定は1つのまとまり(mod):
    got = mod._sheet_groups(tiles(), by_class=False)
    assert [(c, [t[0] for t in ts]) for c, ts in got] == [('', [1, 2, 3, 4])]


def test_クラスごとに分ける(mod):
    got = mod._sheet_groups(tiles(), by_class=True)
    assert [(c, [t[0] for t in ts]) for c, ts in got] == [
        ('comp', [1, 3]),
        ('species_col', [2]),
        ('layer', [4]),
    ]


def test_クラスの並びは最初に出てきた順(mod):
    """cell_id の順で読めるように，出現順を保つ"""
    # 逆順に渡すと layer(4) -> comp(3) -> species_col(2) の順に初めて出てくる
    got = mod._sheet_groups(tiles()[::-1], by_class=True)
    assert [c for c, _ in got] == ['layer', 'comp', 'species_col']
    assert [t[0] for t in dict(got)['comp']] == [3, 1]


def test_空でも落ちない(mod):
    assert mod._sheet_groups([], by_class=True) == []
    assert mod._sheet_groups([], by_class=False) == [('', [])]
