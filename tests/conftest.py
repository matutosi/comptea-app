"""テストの下ごしらえ

中核のモジュールは `comptea/` に平らに置いてあり，互いを `import locate` の
形で読む．種名の辞書も `j_name.txt` のような**相対パス**で開く．
そこで `sys.path` を通し，**作業ディレクトリを `comptea/` に移す**
(CUI もアプリも `cwd=comptea` で呼んでいるので，同じ条件になる)．
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "comptea")
APPS = os.path.join(ROOT, "apps")
SAMPLE = os.path.join(ROOT, "examples", "sample.jpg")

# `apps/_shared.py` はアプリが `import _shared` で読むので，同じ形で通す
for p in (CORE, APPS, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session", autouse=True)
def _in_core():
    """辞書を相対パスで開けるように，`comptea/` で動かす"""
    old = os.getcwd()
    os.chdir(CORE)
    yield
    os.chdir(old)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="重いテスト(検出・読み取りの通し)も走らせる")


def pytest_collection_modifyitems(config, items):
    """`slow` の印が付いたものは，`--runslow` のときだけ走らせる

    検出は重みの読み込みだけで数十秒かかるので，既定では飛ばす．
    """
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="重いので既定では飛ばす(--runslow で走る)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
