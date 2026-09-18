"""公開リポジトリに，手元の実際のパスを書かない (2026-09-14 ユーザ決定)

外の道具 (yomitoku・NDLOCR-Lite) の置き場は**環境変数で指す**．
既定値に手元のパスを書くと，公開したときにその人の環境が漏れるうえ，
他の環境では意味のない値になる．

うっかり戻したときに気づけるよう，**コードの字面**で確かめる
(`tests/test_wiring.py` と同じ考え方)．
"""
import glob
import io
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `C:\...` `D:/...` のような絶対パスと，ユーザの置き場
ABS = re.compile(r"""['"]\s*[A-Za-z]:[\\/]""")
HOME = re.compile(r'/home/[a-z]|/Users/[A-Za-z]|Documents and Settings')

FILES = sorted(glob.glob(os.path.join(ROOT, 'comptea', '*.py'))
               + glob.glob(os.path.join(ROOT, 'comptea', 'pipeline', '*.py'))
               + glob.glob(os.path.join(ROOT, 'cli', '*.py')))


@pytest.mark.parametrize('path', FILES, ids=[os.path.basename(p) for p in FILES])
def test_手元の絶対パスを書かない(path):
    bad = []
    for i, line in enumerate(io.open(path, encoding='utf-8'), 1):
        s = line.split('#')[0]                   # 注釈の中の例は許す
        if ABS.search(s) or HOME.search(s):
            bad.append(f'{i}: {line.strip()[:80]}')
    assert not bad, ('手元のパスが書かれている (環境変数で指す): '
                     + ' / '.join(bad))


# 文書・テスト・評価の道具も公開される．字面のドライブ名で確かめる
# (2026-09-17．範囲が comptea/・cli/ だけで，文書とテストの docstring に漏れていた)
DRIVE = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]+[A-Za-z]')
# テストの中の架空の置き場は許す
FAKE = re.compile(r'[A-Za-z]:[\\/]+(no[\\/]such|Users[\\/]+x[\\/]|w[\\/])')

OTHERS = sorted(
    p for pat in ('*.md', 'docs/**/*.md', 'eval/**/*.md', 'eval/*.py',
                  'apps/**/*.py', 'tests/*.py', '.claude/skills/**/*.md')
    for p in glob.glob(os.path.join(ROOT, pat), recursive=True)
    if 'worktrees' not in os.path.relpath(p, ROOT)
    and os.path.basename(p) != os.path.basename(__file__))


@pytest.mark.parametrize('path', OTHERS,
                         ids=[os.path.relpath(p, ROOT) for p in OTHERS])
def test_文書とテストにも手元のパスを書かない(path):
    bad = []
    for i, line in enumerate(io.open(path, encoding='utf-8'), 1):
        s = FAKE.sub('', line)
        if DRIVE.search(s) or HOME.search(s):
            bad.append(f'{i}: {line.strip()[:80]}')
    assert not bad, '手元のパスが書かれている: ' + ' / '.join(bad)


def test_外の道具の既定は空():
    """既定値に置き場を書かない．`COMPTEA_*` で指す"""
    from comptea import ndl, yomi

    assert yomi.DEFAULT_PYS == ()
    assert ndl.DEFAULT_DIRS == ()
    assert yomi.ENV_PY == 'COMPTEA_YOMI_PY'
    assert ndl.ENV == 'COMPTEA_NDLOCR'


# 見本の zip の中の表にも，作った人の置き場が残る (2026-09-18．`model` 列に
# 重みの絶対パス，`source_image` に手元の画像の場所が 1,100 行あった)
ZIPS = sorted(glob.glob(os.path.join(ROOT, 'examples', '*.zip')))


@pytest.mark.parametrize('path', ZIPS, ids=[os.path.basename(p) for p in ZIPS])
def test_見本のzipの中にも手元のパスを書かない(path):
    import zipfile

    bad = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            text = z.read(name).decode('utf-8', errors='replace')
            for i, line in enumerate(text.splitlines(), 1):
                if DRIVE.search(line) or HOME.search(line):
                    bad.append(f'{name}:{i}: {line.strip()[:60]}')
    assert not bad, '手元のパスが入っている: ' + ' / '.join(bad[:5])
