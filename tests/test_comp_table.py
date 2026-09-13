"""縦持ちに組む段の判断

括弧付きのセルの読み分け(常在度か，単独地点の被度か)と，
列の境で割れた値の繕い．どちらも**黙って別の値になる**型なので歯止めを置く．
"""
import pandas as pd
import pytest

from comptea import comp_table as cta


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


# --- 流し込みの行に印を付ける ----------------------------------------------
#
# 2026-09-13 に全 147 表を通して分かった．17_p1 の要確認 591 件のうち
# **278 件 (47%) が，格子の下端が「出現 1 回の種」の文章に食い込んだ 9 行**
# だった．**欠落を避けるために残している行**なので落とさず，印を付けて
# 後段で外せるようにする (2026-09-11 の方針どおり)．
#
# 判定は `row_kinds.looks_flow_ja` と同じ考え: **本物の行は和名より右に
# ラテン語が無い**．画像でなく，読み終えた文字列に当てる．

def test_和名にラテン語が並ぶ行は流し込み():
    df = pd.DataFrame([
        {'row_no': 1, 'j_name': 'ススキ', 's_name': 'Miscanthus sinensis',
         'note': None},
        {'row_no': 2, 'j_name': 'In 5: Tsuga diversifolia ツガ K-t, Vlola',
         's_name': None, 'note': None},
    ])
    got = cta.mark_flow(df)
    assert 'flow' not in str(got.loc[0, 'note'] or '')
    assert 'flow' in str(got.loc[1, 'note'])


def test_見出しの語でも流し込みとみなす():
    df = pd.DataFrame([{'row_no': 1, 'j_name': '出現1回の種 Außerdem je einmal',
                        's_name': None, 'note': None}])
    assert 'flow' in str(cta.mark_flow(df).loc[0, 'note'])


def test_本物の行には付けない():
    df = pd.DataFrame([
        {'row_no': 1, 'j_name': 'コナラ', 's_name': 'Quercus serrata',
         'note': None},
        {'row_no': 2, 'j_name': 'ヤマザクラ', 's_name': 'Prunus jamasakura',
         'note': 'y_fitted'},
    ])
    got = cta.mark_flow(df)
    assert all('flow' not in str(v) for v in got['note'])


def test_同じ行のセルには同じ印():
    df = pd.DataFrame([
        {'row_no': 7, 'j_name': 'In 5: Tsuga diversifolia ツガ', 's_name': None,
         'note': None},
        {'row_no': 7, 'j_name': None, 's_name': None, 'note': 'y_fitted'},
    ])
    got = cta.mark_flow(df)
    assert all('flow' in str(v) for v in got['note'])


def test_元の印は残す():
    df = pd.DataFrame([{'row_no': 1, 'j_name': 'In 5: Tsuga diversifolia ツガ',
                        's_name': None, 'note': 'y_fitted'}])
    got = cta.mark_flow(df)
    assert 'y_fitted' in str(got.loc[0, 'note'])
