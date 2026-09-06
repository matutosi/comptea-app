"""縦持ちに組む段の判断

括弧付きのセルの読み分け(常在度か，単独地点の被度か)と，
列の境で割れた値の繕い．どちらも**黙って別の値になる**型なので歯止めを置く．
"""
import pandas as pd
import pytest

import comp_table as cta


# --- セルを被度と群度に分ける -------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("4;2", ("4", "2", "OK")),
    ("5", ("5", None, "OK")),
    # `+` は `+・1`，`r` は `r・1` の省略形なので群度 1 を補う
    ("+", ("+", "1", "OK")),
    ("r", ("r", "1", "OK")),
    ("", (None, None, "absent")),
    ("・", (None, None, "absent")),
    (None, (None, None, "absent")),
])
def test_被度と群度に分ける(text, want):
    assert cta.split_comp(text) == want


def test_括弧付きのセルは列の種類で意味が変わる():
    """**1 つの表に両方が混ざる**(2026-09-05 に実物で確認)

    群落の要約列では常在度(括弧の中は被度の範囲)，単独地点の列では
    被度・群度．セルだけを見て意味は決められない．
    """
    assert cta.split_comp("IV(+-3)", kind="constancy") == ("+-3", None, "OK")
    assert cta.split_comp("2(3-4)", kind="plot") == ("2", "3-4", "OK")
    # 決まらないときは常在度として扱う
    assert cta.split_comp("IV(+-3)") == ("+-3", None, "OK")


# --- 列の多数決 ---------------------------------------------------------

def _comp(values_by_col):
    rows = []
    for col, values in values_by_col.items():
        for i, v in enumerate(values):
            rows.append({"col": col, "row": i, "comp_raw": v})
    return pd.DataFrame(rows)


def test_列ごとに常在度か被度かを決める():
    comp = _comp({
        1: ["IV(+-3)", "III(+-1)", "II(+-2)", "V(1-3)"],   # 群落の要約列
        2: ["1(+)", "2(3-4)", "3(+-1)", "2(2)"],            # 単独地点の列
    })
    kind = cta.column_head_kinds(comp)
    assert set(kind[comp["col"] == 1]) == {"constancy"}
    assert set(kind[comp["col"] == 2]) == {"plot"}


def test_票が足りない列は決めない():
    """括弧付きのセルが少ない列で，取り違えて全部を書き換えないための歯止め"""
    comp = _comp({1: ["IV(+-3)", "5;5", "・"]})
    assert cta.column_head_kinds(comp).isna().all()


def test_常在度の割合():
    """頭が**ローマ数字のものだけ**を数える(算用数字の頭は単独地点の被度)"""
    assert cta.constancy_share(["IV(+-3)", "III(+)", "5;5", "1(+)"]) == 0.5
    assert cta.constancy_share([]) == 0.0


# --- 割れた値の繕い -----------------------------------------------------

def _split_case(a, b):
    return pd.DataFrame([
        {"block": 1, "row": 1, "plot": 1, "comp_raw": a},
        {"block": 1, "row": 1, "plot": 2, "comp_raw": b},
    ])


def test_隣へはみ出した被度を分け直す():
    """`example.jpg` の行18 は `[1・] [11・1]` と組まれている

    切り出しは幾何で決まるので，そのまま切ると `1` と `1;1;1` になる．
    **移した結果，両方が値として読めるときだけ**直す．
    """
    got, n = cta.repair_split_values(_split_case("1", "1;1;1"))
    assert n == 1
    assert list(got["comp_raw"]) == ["1;1", "1;1"]
    assert all("moved" in s for s in got["note"])


def test_割れた常在度を繋ぎ直す():
    """`III(+-1)` は 8 文字あり，列の境で割れやすい(2026-09-04)"""
    got, n = cta.repair_split_values(_split_case("III(", "+-1)"))
    assert n == 1
    assert sorted(got["comp_raw"]) == ["", "III(+-1)"]


def test_片方でも値として読めるなら繋がない():
    """`1(+)` と `2(+)` という**別々の値**を `II(+-1)` に繋いでいた"""
    got, n = cta.repair_split_values(_split_case("1(+)", "2(+)"))
    assert n == 0
    assert list(got["comp_raw"]) == ["1(+)", "2(+)"]


def test_片方が括弧だけの断片なら繋がない():
    """`4(1` + `}` は，印字 `IV(1-3)` なのに `IV(1)` になっていた

    OCR が落とした分を見落としたまま，値らしい形になる(2026-09-04)．
    """
    got, n = cta.repair_split_values(_split_case("4(1", "}"))
    assert n == 0


def test_読めるものは壊さない():
    got, n = cta.repair_split_values(_split_case("5;5", "4;2"))
    assert n == 0
    assert list(got["comp_raw"]) == ["5;5", "4;2"]
