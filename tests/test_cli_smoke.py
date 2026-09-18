"""cli/ の入口が読み込めて，引数の説明まで通る (同じプロセスで確かめる)

cli/ は薄い包みなので中身の試験は comptea/ の側にあるが，入口そのものは
一度も動かしていなかった (網羅 0%)．古い関数名を呼ぶ・import で落ちる，
といった壊れ方を安く捕まえる．subprocess で回すより速く，網羅にも数えられる．
"""
import importlib.util
import pathlib
import sys

import pytest

import conftest

CLI = sorted((pathlib.Path(conftest.ROOT) / 'cli').glob('*.py'))


def _load(path):
    spec = importlib.util.spec_from_file_location(f'cli_{path.stem}_smoke', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('path', CLI, ids=[p.stem for p in CLI])
def test_入口が読み込める(path):
    m = _load(path)
    # 薄い包み (run_ocr など) は段のモジュールを持つだけ．その main を確かめる
    stage = [v for v in vars(m).values()
             if getattr(v, '__name__', '').startswith('comptea.pipeline.')]
    assert (hasattr(m, 'main') or hasattr(m, 'parse_args')
            or any(hasattr(s, 'main') for s in stage))


@pytest.mark.parametrize('path', [p for p in CLI if 'def parse_args' in
                                  p.read_text(encoding='utf-8')],
                         ids=lambda p: p.stem)
def test_引数の説明が出る(path, monkeypatch, capsys):
    m = _load(path)
    monkeypatch.setattr(sys, 'argv', [path.name, '--help'])
    with pytest.raises(SystemExit) as e:
        try:
            m.parse_args()
        except TypeError:                       # parse_args(argv) の形
            m.parse_args(['--help'])
    assert e.value.code == 0
    assert 'usage' in capsys.readouterr().out.lower()
