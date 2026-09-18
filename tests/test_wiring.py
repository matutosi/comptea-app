"""モジュールをまたぐ参照が，実際に解決するか

**関数を別のモジュールへ移したのに，呼び出し側が古い置き場を指したまま**でも，
Python は動かすまで気づかない．2026-09-07 に `locate._edge_ink` を `axes.py` へ
移したとき，`col_edges.py` の 1 行が残り，**153 表のうち 50 表が落ちた**
(通しを 3 時間回して初めて分かった．テストも pyflakes も見本 1 枚も通っていた)．

ここでは**字面で**確かめる．`import` した仲間のモジュールに対する
`<モジュール>.<名前>` が，そのモジュールに本当にあるか見るだけなので一瞬で済む．

**重い依存が入っていない場でも回せる**ようにしてある(CI は easyocr も
ultralytics も入れない)．読めないモジュールは飛ばし，その旨を出す．
"""
import ast
import importlib
import pathlib

import pytest

import conftest

ROOT = pathlib.Path(conftest.ROOT)
ALL_FILES = sorted(list((ROOT / 'comptea').glob('*.py'))
                   + list((ROOT / 'comptea' / 'pipeline').glob('*.py'))
                   + list((ROOT / 'cli').glob('*.py'))
                   + list((ROOT / 'eval').glob('*.py')))
# 中核のモジュールを `comptea.` の下の点つきの名前で持つ ('ink'・'pipeline.read' など)
CORE = ({p.stem for p in (ROOT / 'comptea').glob('*.py')}
        | {f'pipeline.{p.stem}' for p in (ROOT / 'comptea' / 'pipeline').glob('*.py')}
        | {'pipeline'}) - {'__init__', 'pipeline.__init__'}


def _package(path):
    """このファイルの属するパッケージ (相対 import の基点)．comptea の外なら None"""
    rel = path.relative_to(ROOT).with_suffix('').parts
    if rel[0] != 'comptea':
        return None
    return '.'.join(rel[1:-1])                  # '' (comptea 直下) か 'pipeline'


def sibling_imports(tree, path=None):
    """このファイルが読んでいる中核のモジュール {呼ぶときの名前: モジュール名}

    `from comptea import ink`・`import comptea.ink`・`from . import ink`・
    `from comptea.pipeline import common as _common`・`from .. import note` を拾う
    (2026-09-18．相対の import と pipeline の下を拾っておらず，
    `_common.xxx` のような呼び出しが調べられていなかった)
    """
    pkg = _package(path) if path is not None else ''
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level:                          # 相対: 基点から level-1 段上がる
                if pkg is None:
                    continue
                parts = pkg.split('.') if pkg else []
                parts = parts[:len(parts) - (n.level - 1)]
                base = '.'.join(parts + ([n.module] if n.module else []))
            elif n.module == 'comptea':
                base = ''
            elif n.module and n.module.startswith('comptea.'):
                base = n.module[len('comptea.'):]
            else:
                continue
            for a in n.names:
                mod = f'{base}.{a.name}' if base else a.name
                if mod in CORE:
                    out[a.asname or a.name] = mod
        elif isinstance(n, ast.Import):
            for a in n.names:
                mod = a.name[len('comptea.'):] if a.name.startswith('comptea.') else None
                if mod in CORE:
                    out[a.asname or mod.split('.')[-1]] = mod
    return out


# 中核のモジュールを 1 つも読まないファイルは，集める段で外す
# (以前は 29 件が毎回「確かめる対象が無い」skip として数えられていた)
FILES = [p for p in ALL_FILES
         if sibling_imports(ast.parse(p.read_text(encoding='utf-8')), p)]


def test_相対のimportとpipelineの下も拾う():
    tree = ast.parse('from . import common as _common\n'
                     'from .. import note\n'
                     'from comptea.pipeline import read\n'
                     'from comptea import pipeline\n')
    got = sibling_imports(tree, ROOT / 'comptea' / 'pipeline' / 'grid.py')
    assert got == {'_common': 'pipeline.common', 'note': 'note',
                   'read': 'pipeline.read', 'pipeline': 'pipeline'}


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_ほかのモジュールを指す名前が実在する(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = sibling_imports(tree, path)

    mods, missing, skipped = {}, [], {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)):
            continue
        mod = names.get(n.value.id)
        if mod is None or mod in skipped:
            continue
        if mod not in mods:
            try:
                mods[mod] = importlib.import_module(f'comptea.{mod}')
            except ImportError as e:            # 重い依存が入っていない場
                skipped[mod] = str(e)
                continue
        if not hasattr(mods[mod], n.attr):
            missing.append(f'{path.name}:{n.lineno} {n.value.id}.{n.attr}'
                           f' は comptea/{mod}.py に無い')

    assert not missing, '\n'.join(missing)
    if skipped and not mods:
        pytest.skip('読めないモジュールがある: '
                    + ', '.join(f'{k}({v})' for k, v in sorted(skipped.items())))
