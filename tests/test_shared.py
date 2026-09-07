"""アプリと CUI の受け渡しの部品

工程を分けたので，**間の受け渡しが壊れると通しで何も出ない**．
Cloud でだけ出た不具合が 2 件あり，どちらもここに歯止めを置く．
"""
import io
import os
import zipfile

import pandas as pd
import pytest

import conftest

import _shared
from comptea import detect


# --- 見本と重み ---------------------------------------------------------

@pytest.mark.parametrize("path", ["SAMPLE", "SAMPLE_GRID", "SAMPLE_READ",
                                  "WEIGHTS"])
def test_見本と重みが置いてある(path):
    """2・3・4 は前の工程を通さずに試せる(見本が入口ごとに要る)"""
    assert os.path.isfile(getattr(_shared, path))


# --- 見本を「受け取ったファイル」として渡す ------------------------------

def test_見本はアップロードと同じ顔をする():
    """`LocalFile` は，読み込みの処理を分けずに済ませるためのもの

    Cloud で `AttributeError` が出たのはここ．`zipfile` と `read_csv` に
    そのまま渡せることを見る(2026-09-06)．
    """
    f = _shared.LocalFile(_shared.SAMPLE_READ)
    assert f.name.endswith(".zip")
    assert len(f.getbuffer()) > 0
    with zipfile.ZipFile(f) as z:
        assert z.namelist()


def test_getbufferは何度呼んでも同じ():
    f = _shared.LocalFile(_shared.SAMPLE_GRID)
    assert f.getbuffer() == f.getbuffer()


# --- zip での受け渡し ---------------------------------------------------

def test_zipにまとめて展開して戻る(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ok", encoding="utf-8")
    blob = _shared.zip_files(str(tmp_path), ["a.csv", "b.txt", "無い.csv"])
    assert blob

    import io

    up = io.BytesIO(blob)
    up.name = "grid.zip"
    out = tmp_path / "out"
    out.mkdir()
    names = _shared.unzip_into(up, str(out))
    assert sorted(names) == ["a.csv", "b.txt"]
    assert (out / "a.csv").is_file()


def test_1枚だけでも受け取れる(tmp_path):
    """CSV を 1 枚だけ渡す道も残してある"""
    import io

    up = io.BytesIO(b"x\n1\n")
    up.name = "ocred.csv"
    assert _shared.unzip_into(up, str(tmp_path)) == ["ocred.csv"]


def test_無いものだけならNoneを返す(tmp_path):
    assert _shared.zip_files(str(tmp_path), ["無い.csv"]) is None


# --- 大きすぎる画像 -----------------------------------------------------

def test_大きすぎる画像は手元のCUIへ誘導する():
    """上限は**検出の縮尺**で決まる(2026-09-07 に測って決め直した)"""
    from PIL import Image

    assert _shared.too_big(Image.new("L", (100, 100))) is None
    msg = _shared.too_big(Image.new("L", (12000, 3000)))
    assert msg and "折り込み" in msg


# --- 検出の返り値 -------------------------------------------------------

class _Polars:
    """`to_df()` が polars を返す新しい ultralytics の代わり"""

    def to_dicts(self):
        return [{"name": "row", "confidence": 0.9}]


def test_検出の返りをpandasにそろえる():
    """**ultralytics の版で中身が変わる**(2026-09-06 に Cloud で発覚)

    新しい版は polars を返し，`'Series' object has no attribute 'apply'`
    で落ちていた．
    """
    got = detect.to_pandas(_Polars())
    assert isinstance(got, pd.DataFrame) and got.at[0, "name"] == "row"
    same = pd.DataFrame([{"name": "col"}])
    assert detect.to_pandas(same) is same


# --- 結果を憶えておく ---------------------------------------------------

def test_入力が同じなら憶えた結果を返す(monkeypatch):
    """**Streamlit はどのウィジェットを触っても全体を走らせ直す**

    重い処理をボタンの中だけに置くと，次の再実行で結果が消える(2026-09-06)．
    """
    import streamlit as st

    state = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)
    assert _shared.cached("k", ("a", 1)) is None
    _shared.remember("k", ("a", 1), {"n": 3})
    assert _shared.cached("k", ("a", 1)) == {"n": 3}
    # 入力が変われば捨てる
    assert _shared.cached("k", ("b", 2)) is None


def test_入力の印は名前と大きさで決まる():
    a = _shared.LocalFile(_shared.SAMPLE_GRID)
    b = _shared.LocalFile(_shared.SAMPLE_READ)
    assert _shared.upload_sig(a) == _shared.upload_sig(a)
    assert _shared.upload_sig(a) != _shared.upload_sig(b)
    assert _shared.upload_sig(None) == (None,)
    # 印を取っても，中身は先頭から読める(`zipfile` に渡せる)
    _shared.upload_sig(a)
    with zipfile.ZipFile(a) as z:
        assert z.namelist()


# --- 段を同じプロセスで呼ぶ ---------------------------------------------

def test_段を同じプロセスから呼べる(tmp_path):
    """アプリはこれを呼ぶ(2026-09-07)

    前は `cli/*.py` を subprocess で起こしており，そのたびに torch や
    easyocr を読み込んでいた．CPU 版の torch なら検出 432 MB・読み取り
    507 MB で，無料枠(1 GB ほど)に収まると測って決めた．
    """
    import zipfile

    from comptea import pipeline

    with zipfile.ZipFile(os.path.join(conftest.ROOT, "examples",
                                      "sample_read.zip")) as z:
        z.extractall(tmp_path)
    code, out = pipeline.run("table", [str(tmp_path)])
    assert code == 0, out
    assert "[地点数] 本体 6 地点" in out
    assert (tmp_path / "comp_table_long.csv").is_file()


def test_できないときは終了コードと理由が返る(tmp_path):
    """段は「できない理由」を SystemExit で返す．例外にせず拾う"""
    from comptea import pipeline

    code, out = pipeline.run("table", [str(tmp_path / "無い")])
    assert code != 0
    assert out.strip()


# --- 表の見せ方 ---------------------------------------------------------

def test_出す行数を選べる(monkeypatch):
    """長い表をそのまま出すと画面が重い(縦持ちは数千行になる)"""
    import streamlit as st

    monkeypatch.setattr(st, "selectbox", lambda *a, **k: 100, raising=False)
    assert _shared.how_many("出す行数", "k") == 100
    monkeypatch.setattr(st, "selectbox", lambda *a, **k: 0, raising=False)
    assert _shared.how_many("出す行数", "k") is None      # 0 は「すべて」


def test_セルの実物を切り出して並べる():
    """**読んだ字の隣に実物を置く**．読みだけでは違いに気づけない"""
    import pandas as pd

    df = pd.DataFrame([{"x1": 100, "y1": 200, "x2": 300, "y2": 260},
                       {"x1": 100, "y1": 260, "x2": 300, "y2": 320}])
    got = _shared.cell_images(df, conftest.SAMPLE)
    assert list(got.columns)[0] == "画像"
    assert all(str(v).startswith("data:image/png;base64,") for v in got["画像"])


def test_箱が無ければ画像の列を作らない():
    import pandas as pd

    df = pd.DataFrame([{"text": "S"}])
    assert "画像" not in _shared.cell_images(df, conftest.SAMPLE).columns


def test_zipの中の1枚を差し替える(tmp_path):
    """直した値を，次の工程へそのまま渡すため"""
    (tmp_path / "ocred.csv").write_text("x\n1\n", encoding="utf-8", newline="")
    (tmp_path / "located.csv").write_text("y\n2\n", encoding="utf-8", newline="")
    blob = _shared.zip_files(str(tmp_path), ["ocred.csv", "located.csv"])
    new = _shared.replace_in_zip(blob, "ocred.csv", "x\n9\n")
    with zipfile.ZipFile(io.BytesIO(new)) as z:
        assert sorted(z.namelist()) == ["located.csv", "ocred.csv"]
        assert z.read("ocred.csv").decode("utf-8-sig") == "x\n9\n"
        assert z.read("located.csv").decode("utf-8") == "y\n2\n"


def test_中核の古い写しを捨てる(monkeypatch, tmp_path):
    """公開先では，`import` した先が古いまま残ることがある

    `split_sheet.cut_table` を足した直後に「無い」と言われた
    (2026-09-07 ユーザ報告)．ファイルの更新時刻が変わったときだけ捨てる．
    """
    import pathlib
    import sys

    import streamlit as st

    state = {}
    monkeypatch.setattr(st, "session_state", state, raising=False)
    from comptea import split_sheet                    # noqa: F401

    assert _shared.reload_core() == 0                  # 初回は捨てない
    assert "comptea.split_sheet" in sys.modules

    pathlib.Path(_shared.CORE, "split_sheet.py").touch()
    assert _shared.reload_core() > 0                   # 新しくなったら捨てる
    assert "comptea.split_sheet" not in sys.modules
    assert _shared.reload_core() == 0                  # 続けて呼んでも捨てない

    from comptea import split_sheet as again           # 読み直せる
    assert hasattr(again, "cut_table")
