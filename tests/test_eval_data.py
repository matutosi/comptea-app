"""物差しが使うデータの置き場(環境変数 COMPTEA_DATA)

ラベルを付けたスキャンと正解表は**このリポジトリの外**にある．
これまでは「置き場を作業ディレクトリにして呼ぶ」約束だったが，
置き場が動くと呼び方も変わるので，環境変数で指せるようにした(2026-09-08)．
"""
import importlib.util
import os
import pathlib
import sys

import pytest

import conftest

SRC = pathlib.Path(conftest.ROOT) / 'eval' / '_data.py'


@pytest.fixture(scope='module')
def mod():
    spec = importlib.util.spec_from_file_location('eval_data_under_test', SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def here(tmp_path, monkeypatch):
    """いまいる場所を tmp に移し，環境変数を消してから試す"""
    monkeypatch.delenv('COMPTEA_DATA', raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_設定が無ければ何もしない(mod, here):
    assert mod.use_data_dir() == pathlib.Path.cwd()
    assert pathlib.Path.cwd() == here


def test_設定してあればそこへ移る(mod, here, monkeypatch):
    data = here / 'data'
    (data / 'labelme_data').mkdir(parents=True)
    monkeypatch.setenv('COMPTEA_DATA', str(data))
    got = mod.use_data_dir()
    assert got == data.resolve()
    assert pathlib.Path.cwd().resolve() == data.resolve()


def test_正解表だけでも置き場とみなす(mod, here, monkeypatch):
    data = here / 'data'
    (data / 'truth').mkdir(parents=True)
    monkeypatch.setenv('COMPTEA_DATA', str(data))
    assert mod.use_data_dir() == data.resolve()


def test_場所が無ければ止める(mod, here, monkeypatch):
    monkeypatch.setenv('COMPTEA_DATA', str(here / 'no_such'))
    with pytest.raises(SystemExit) as e:
        mod.use_data_dir()
    assert 'COMPTEA_DATA' in str(e.value)


def test_データが見あたらなければ止める(mod, here, monkeypatch):
    """**黙って別の場所で動かさない**．既定値は相対の場所なので，
    間違った場所へ移ると「0 件」で終わって気づけない
    """
    empty = here / 'empty'
    empty.mkdir()
    monkeypatch.setenv('COMPTEA_DATA', str(empty))
    with pytest.raises(SystemExit) as e:
        mod.use_data_dir()
    assert 'labelme_data' in str(e.value)


def test_物差しは全部この部品を呼ぶ():
    """1 本でも呼び忘れると，その物差しだけ置き場が効かない"""
    root = pathlib.Path(conftest.ROOT) / 'eval'
    for p in sorted(root.glob('*.py')):
        if p.name == '_data.py':
            continue
        s = p.read_text(encoding='utf-8')
        assert 'from _data import use_data_dir' in s, p.name
        assert 'use_data_dir()' in s.replace(
            'from _data import use_data_dir', ''), p.name
