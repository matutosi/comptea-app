"""モジュールをまたぐ参照が，実際に解決するか

**関数を別のモジュールへ移したのに，呼び出し側が古い place を指したまま**でも，
Python は動かすまで気づかない．2026-09-07 に `locate._edge_ink` を `axes.py` へ
移したとき，`col_edges.py` の 1 行が残り，**153 表のうち 50 表が落ちた**
(通しを 3 時間回して初めて分かった)．

ここでは**字面で**確かめる．`import` した仲間のモジュールに対する
`<モジュール>.<名前>` が，そのモジュールに本当にあるか見るだけなので一瞬で済む．
"""
import ast
import importlib
import pathlib

import pytest

import conftest

ROOT = pathlib.Path(conftest.ROOT)
FILES = sorted(list((ROOT / 'comptea').glob('*.py'))
               + list((ROOT / 'comptea' / 'pipeline').glob('*.py'))
               + list((ROOT / 'cli').glob('*.py'))
               + list((ROOT / 'eval').glob('*.py')))
CORE = {p.stem for p in (ROOT / 'comptea').glob('*.py')} - {'__init__'}


def sibling_imports(tree):
    """このファイルが読んでいる中核のモジュール {名前: モジュール名}"""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module in ('comptea', '.', None):
            for a in n.names:
                if a.name in CORE:
                    out[a.asname or a.name] = a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith('comptea.') and a.name.split('.')[-1] in CORE:
                    out[a.asname or a.name.split('.')[-1]] = a.name.split('.')[-1]
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_ほかのモジュールを指す名前が実在する(path):
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    names = sibling_imports(tree)
    if not names:
        pytest.skip('中核のモジュールを読んでいない')
    missing = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)):
            continue
        mod = names.get(n.value.id)
        if mod is None:
            continue
        m = importlib.import_module(f'comptea.{mod}')
        if not hasattr(m, n.attr):
            missing.append(f'{path.name}:{n.lineno} {n.value.id}.{n.attr}'
                           f' は comptea/{mod}.py に無い')
    assert not missing, '\n'.join(missing)
