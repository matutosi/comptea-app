"""アプリと CUI の受け渡しの部品

工程を分けたので，**間の受け渡しが壊れると通しで何も出ない**．
Cloud でだけ出た不具合が 2 件あり，どちらもここに歯止めを置く．
"""
import os
import zipfile

import pandas as pd
import pytest

import conftest

import _shared
import detect


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
    from PIL import Image

    assert _shared.too_big(Image.new("L", (100, 100))) is None
    msg = _shared.too_big(Image.new("L", (_shared.MAX_SIDE + 1, 10)))
    assert msg and "CUI" in msg


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
