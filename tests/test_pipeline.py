"""見本 1 枚を通す煙テスト

工程どうしの繋ぎ目は，部品ごとの試験では見えない
(「部品ごとには動くのに，通しでは何も出ない」型の壊れ方をする)．
`examples/` の見本だけで完結するので，実データが無くても走る．

検出と読み取りは重いので `slow` の印を付ける(`--runslow` で走る)．
"""
import os
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

import conftest

CORE = conftest.CORE
ROOT = conftest.ROOT


def _run(script, *args):
    """CUI の入口を，中核の場所で走らせる"""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "cli", script), *args],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=CORE)
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    return (r.stdout or "") + (r.stderr or "")


# --- 段階1 格子 ---------------------------------------------------------

@pytest.mark.slow
def test_見本から格子ができる(tmp_path):
    wd = tmp_path / "out"
    _run("run_pipeline.py", os.path.join(ROOT, "examples", "sample.jpg"),
         "--workdir", str(wd))
    assert (wd / "located.csv").is_file()
    assert (wd / "overlay.png").is_file()     # 目で確かめる絵が要る

    loc = pd.read_csv(wd / "located.csv")
    body = loc[loc["obj_name"] == "comp"]
    # 見本は 6 地点・27 行の表．検出のゆらぎを見込んで幅を持たせる
    assert 20 <= body["row"].nunique() <= 35
    assert 5 <= body["col"].nunique() <= 8
    assert len(body) >= 100


# --- 段階3 組み上げ -----------------------------------------------------

@pytest.fixture
def read_dir(tmp_path):
    """見本の読み取り結果を展開した作業ディレクトリ"""
    with zipfile.ZipFile(os.path.join(ROOT, "examples", "sample_read.zip")) as z:
        z.extractall(tmp_path)
    return tmp_path


def test_見本を縦持ちに組める(read_dir):
    out = _run("build_table.py", str(read_dir))
    assert "[地点数] 本体 6 地点" in out

    long = pd.read_csv(read_dir / "comp_table_long.csv")
    for col in ("plot", "row_no", "j_name", "layer", "cover", "status", "source"):
        assert col in long.columns
    assert long["plot"].max() == 6
    assert (long["status"] == "OK").sum() > 40
    # **和名が 1 つも無い表**は，格子の段では気づけない(2026-09-05)
    assert long["j_name"].notna().sum() > 0


def test_表頭も表になる(read_dir):
    _run("build_table.py", str(read_dir))
    plot = pd.read_csv(read_dir / "plot_table.csv")
    assert len(plot) == 6                     # 1 行 = 1 地点
    assert "altitude" in plot.columns


def test_検査の結果が残る(read_dir):
    _run("build_table.py", str(read_dir))
    text = (read_dir / "checks.txt").read_text(encoding="utf-8")
    assert "[重複]" in text and "[被度]" in text


# --- 段階2 読み取り -----------------------------------------------------

@pytest.mark.slow
def test_見本のセルを読める(tmp_path):
    """EasyOCR の重みを落としてくるので，とりわけ重い"""
    with zipfile.ZipFile(os.path.join(ROOT, "examples", "sample_grid.zip")) as z:
        z.extractall(tmp_path)
    import shutil

    shutil.copy(os.path.join(ROOT, "examples", "sample.jpg"),
                tmp_path / "sample.jpg")
    _run("run_ocr.py", str(tmp_path))
    ocred = pd.read_csv(tmp_path / "ocred.csv")
    assert "corrected" in ocred.columns
    assert ocred["corrected"].notna().sum() > 50
