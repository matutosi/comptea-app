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


def test_外の道具の既定は空():
    """既定値に置き場を書かない．`COMPTEA_*` で指す"""
    from comptea import ndl, yomi

    assert yomi.DEFAULT_PYS == ()
    assert ndl.DEFAULT_DIRS == ()
    assert yomi.ENV_PY == 'COMPTEA_YOMI_PY'
    assert ndl.ENV == 'COMPTEA_NDLOCR'
