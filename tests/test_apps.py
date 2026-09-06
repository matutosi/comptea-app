"""Streamlit のアプリ 4 つ

**画面まで走らせて確かめる**(`streamlit.testing.v1.AppTest`)．
構文だけ見ても足りない: Cloud で出た `LocalFile` の不具合は，
「Streamlit は書き換えたファイルを走らせ直すが，`import` した先はそのまま」
という筋で，走らせて初めて出た(2026-09-06)．
"""
import os

import pytest

from streamlit.testing.v1 import AppTest

import conftest

APPS = ["1_split", "2_grid", "3_read", "4_table"]


def _path(key):
    return os.path.join(conftest.ROOT, "apps", key, "streamlit_app.py")


@pytest.mark.parametrize("key", APPS)
def test_画面が出るところまで走る(key):
    """見本を使う既定のまま，例外を出さずに描き切れること

    3・4 は既定で見本のデータを読むので，この 1 本で見本の読み込みも通る．
    """
    at = AppTest.from_file(_path(key), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith(key[0])


@pytest.mark.parametrize("key", APPS)
def test_全体像とほかの工程への案内が出る(key):
    """4 つは別々の URL になるので，どのページにも全体像を置く(2026-09-06)"""
    at = AppTest.from_file(_path(key), default_timeout=120)
    at.run()
    assert any("組成表" in m.value for m in at.info)
    side = [m.value for m in at.sidebar.markdown]
    assert len(side) >= len(APPS)
    now = [s for s in side if s.startswith("**▶")]
    assert len(now) == 1                       # 現在地はちょうど 1 つ


@pytest.mark.parametrize("key", APPS)
def test_入口ごとに依存が置いてある(key):
    """Streamlit Cloud は**入口と同じディレクトリの** requirements.txt を使う

    工程を分けたのは，重い依存を必要なアプリだけに閉じ込めるため．
    """
    req = os.path.join(conftest.ROOT, "apps", key, "requirements.txt")
    text = open(req, encoding="utf-8").read()
    assert "streamlit" in text


def test_重い依存は必要なアプリだけに置く():
    """1 は画像しか触らないので torch を入れない(無料枠のメモリに収まらない)"""
    def req(key):
        return open(os.path.join(conftest.ROOT, "apps", key, "requirements.txt"),
                    encoding="utf-8").read()

    assert "torch" not in req("1_split")
    assert "torch" not in req("4_table")
    assert "torch" in req("2_grid")
    assert "easyocr" in req("3_read")


@pytest.mark.parametrize("key", APPS)
def test_共通の部品を読み直している(key):
    """`importlib.reload(_shared)` が無いと，更新した直後に古いものが残る"""
    src = open(_path(key), encoding="utf-8").read()
    assert "importlib.reload(_shared)" in src
