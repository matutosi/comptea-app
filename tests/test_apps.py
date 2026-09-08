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


def test_見出しは工程の一覧と揃う():
    """見出しは `_shared.APPS` の 1 か所で決める(2026-09-07)

    共通の型に寄せたとき，見出しがタブの短い名前(「1. 切り分け」)に
    変わってしまった．一覧と同じ文言が出ることを押さえる．
    """
    import _shared

    for num, title, key, _ in _shared.APPS:
        at = AppTest.from_file(_path(key), default_timeout=120)
        at.run()
        assert at.title[0].value == f"{num}. {title}"


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


def test_再実行しても結果が消えない():
    """受け取りのボタンを押すと再実行が起きる(2026-09-06 に見つけた不具合)

    重い処理を `if st.button(...)` の中だけに置いていたので，押した直後に
    画面から結果が消えていた．いちばん軽い 4 で，代表して確かめる．
    """
    at = AppTest.from_file(_path("4_table"), default_timeout=300).run()
    at.button[0].click().run()
    assert len(at.dataframe) == 2
    before = [m.value for m in at.metric]
    assert before

    at.checkbox[0].set_value(True).run()       # 別のウィジェットを触る = 再実行
    assert [m.value for m in at.metric] == before
    assert len(at.dataframe) == 2


def test_切り分けは1度だけ走る():
    """1 はボタンが無いので，前は再実行のたびに切り直していた

    `file_uploader` は `AppTest` から入れられないので，差し替えて渡す．
    """
    script = f'''
import sys
sys.path.insert(0, r"{os.path.join(conftest.ROOT, 'apps')}")
import streamlit as st
import _shared
st.file_uploader = lambda *a, **k: _shared.LocalFile(r"{conftest.SAMPLE}")
path = r"{_path('1_split')}"
exec(compile(open(path, encoding="utf-8").read(), path, "exec"),
     {{"__name__": "__main__", "__file__": path}})
'''
    at = AppTest.from_string(script, default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "1_split" in at.session_state          # 結果を憶えている
    kept = at.session_state["1_split"]["value"]
    assert kept["total"] >= 1 and kept["zip"]

    at.run()                                      # 再実行しても作り直さない
    assert at.session_state["1_split"]["value"] is kept


# --- 扱える大きさ -------------------------------------------------------

def test_折り込みは切り分けを通り検出には大きすぎると案内する():
    """**長辺で切ってはいけない**(2026-09-07 ユーザ報告)

    A0 の折り込み(9344 x 12873 px)は長辺 8000 px を超えるが，
    それを表ごとに切るのがこの道具の役目だった．測ると切り分けの山は
    520 MB で無料枠に収まる．検出は縮尺(imgsz)でメモリが決まる．
    """
    import _shared
    from PIL import Image

    sheet = Image.new("L", (9344, 12873))
    assert _shared.too_big_to_split(sheet) is None
    msg = _shared.too_big(sheet)
    assert msg and "1. 折り込みを表ごとに切る" in msg

    page = Image.new("L", (2286, 3293))          # 本のページ
    assert _shared.too_big(page) is None
    assert _shared.too_big_to_split(page) is None

    huge = Image.new("L", (20000, 20000))        # 400 Mpx は切り分けも無理
    assert _shared.too_big_to_split(huge)


def test_Secretsが無くても画面に誤りを出さない(monkeypatch, tmp_path):
    """`st.secrets` は，ファイルが無いと**画面に誤りを出す**

    工程の一覧を作るたびに触るので，1 画面に 8 個並んでいた(2026-09-07)．
    """
    import _shared

    monkeypatch.setattr(_shared, "_secrets_file", lambda: None)
    assert _shared.app_url("2_grid") is None


def test_1枚に表が2つ以上あるとき別々の置き場から集める(tmp_path):
    """`wd` 自身は空になり，`wd_t1`・`wd_t1_s1`・`wd_t2` に格子ができる

    `wd` だけを見ていたので，そういう紙面は「格子が作れなかった」ことに
    なっていた(2026-09-07 に折り込み 23 枚を通して見つけた)．
    """
    import _shared

    wd = tmp_path / "out"
    wd.mkdir()
    src = tmp_path / "sheet.png"                               # 受け取った紙面
    src.write_bytes(b"\x89PNG\r\n")
    part = tmp_path / "out_t1_s1.png"                          # 切り出した部分画像
    part.write_bytes(b"\x89PNG\r\n")
    for name, ref in (("out_t1_s1", part), ("out_t1_s2", src), ("out_t2", src)):
        d = tmp_path / name
        d.mkdir()
        (d / "located.csv").write_text(f"source_image,x\n{ref},1\n",
                                       encoding="utf-8")
    (tmp_path / "out_t9").mkdir()                              # 格子が無い置き場

    got = _shared.grid_tables(str(wd), str(src))
    assert [n for n, _, _ in got] == ["out_t1_s1", "out_t1_s2", "out_t2"]
    # 格子が受け取った紙面とは別の画像を指していれば，その画像も渡す
    assert got[0][2] and got[0][2].endswith("out_t1_s1.png")
    assert got[1][2] is None and got[2][2] is None


def test_表が1つならそのまま渡す(tmp_path):
    import _shared

    wd = tmp_path / "out"
    wd.mkdir()
    (wd / "located.csv").write_text("x\n1\n", encoding="utf-8")
    got = _shared.grid_tables(str(wd))
    assert len(got) == 1 and got[0][0] == "out" and got[0][2] is None


def test_傾きを直した紙面はその画像も渡す(tmp_path):
    """格子の座標は `located.csv` が指す画像のもの(2026-09-07)

    CUI はそれをそのまま使うが，GUI は 3 で受け取った画像に当て直して
    いたので，傾きを直した紙面(最大 1.5°)で座標がずれていた．
    """
    import _shared

    wd = tmp_path / "out"
    wd.mkdir()
    src = tmp_path / "page.png"
    src.write_bytes(b"\x89PNG\r\n")
    fixed = tmp_path / "out_deskew.png"
    fixed.write_bytes(b"\x89PNG\r\n")
    (wd / "located.csv").write_text(f"source_image,x\n{fixed},1\n",
                                    encoding="utf-8")
    got = _shared.grid_tables(str(wd), str(src))
    assert got[0][2] and got[0][2].endswith("out_deskew.png")


def test_続きのページを集める(tmp_path):
    """枝番が付いていれば，表が無くても異常ではない(2026-09-08)

    GUI は `located.csv` だけを見ていたので，続きのページを
    「格子が作れませんでした」と報告していた．
    """
    import _shared

    wd = tmp_path / "s01114_kinki_081-2"
    wd.mkdir()
    (wd / "continuation.txt").write_text(
        "table\ts01114_kinki_081\npart\t2\nimage\tx.jpg\n", encoding="utf-8")
    (wd / "once_block.png").write_bytes(b"\x89PNG\r\n")
    (wd / "note.txt").write_text("調査地: …", encoding="utf-8")
    got = _shared.continuation(str(wd))
    assert got["table"] == "s01114_kinki_081"
    assert got["part"] == "2"
    assert got["once"] and got["note"] is None      # note_block.png は無い
    assert got["zip"]


def test_続きのページでなければ何も返さない(tmp_path):
    import _shared

    wd = tmp_path / "out"
    wd.mkdir()
    (wd / "located.csv").write_text("x\n1\n", encoding="utf-8")
    assert _shared.continuation(str(wd)) is None
