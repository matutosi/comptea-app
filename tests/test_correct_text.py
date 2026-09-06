"""被度・常在度・階層・表頭の値の補正

**実物を見て決めた判断を，そのまま歯止めにする**．
どれも「黙って別の値になる」型の誤りで，通してみても気づけない．
経緯は docs/lessons.md を見る．
"""
import pytest

import correct_text as ct


# --- 被度・群度 ---------------------------------------------------------

@pytest.mark.parametrize("text, corrected", [
    ("5・5", "5;5"),
    ("2・2", "2;2"),
    ("+・2", "+;2"),
    ("r・1", "r;1"),
    ("5", "5"),
    ("+", "+"),
    # **読む人が約束どおり `;` で区切っても壊れない**(2026-09-03)．
    # 区切りを落とし忘れていたころは `5;5` が `5;;;5` になっていた
    ("5;5", "5;5"),
    ("十・2", "+;2"),
])
def test_被度は区切りを揃えて通る(text, corrected):
    got = ct.correct_comp(text)
    assert got == {"corrected": corrected, "status": "OK"}


@pytest.mark.parametrize("text", ["・", "", "  "])
def test_非出現のセルは空で返る(text):
    assert ct.correct_comp(text)["status"] is None


@pytest.mark.parametrize("text", ["36+23", "ムっム", "8"])
def test_値にならないものは目視へ回す(text):
    assert ct.correct_comp(text)["status"] == "Need Check"


# --- 常在度 -------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("IV(+-3)", "IV(+-3)"),
    ("III(+-1)", "III(+-1)"),
    # 閉じ括弧は落ちやすいので，無くても認める
    ("III(+-1", "III(+-1)"),
    # 括弧の中の記号が 2 つ並んだら，間の `-` が落ちたもの
    ("III(+2)", "III(+-2)"),
    # 閉じ括弧を字と読んだ形
    ("III(+-15", "III(+-1)"),
    # `l` `|` `T` はローマ数字の I の誤読．`[` `【` も**頭では** I
    ("lll(+-1)", "III(+-1)"),
    ("[II(+)", "III(+)"),
    # 頭に数字が 2 つ並ぶのはローマ数字の誤読(印字の頭が算用数字なら 1 桁)
    ("11(+-2)", "II(+-2)"),
    # I に満たない散発の出現は `+`・`r` が頭に来る(実物で確認)
    ("+(+)", "+(+)"),
    ("r(+)", "r(+)"),
    # 開き括弧を `し` と読んだ形
    ("4し+-3)", "4(+-3)"),
])
def test_常在度の形を整える(text, want):
    assert ct.correct_constancy(text) == want


def test_単独地点の被度は算用数字のまま返す():
    """**頭の算用数字をローマ数字に読み替えてはいけない**(2026-09-05)

    同じ表に `1(+)`(被度1・群度+)と `I(+)`(常在度I)が**どちらも印字**されて
    いる．どちらの列かは列ごとに決まるので，ここでは印字どおり返す．
    """
    assert ct.correct_constancy("1(+)") == "1(+)"
    assert ct.correct_constancy("2(3-4)") == "2(3-4)"


def test_括弧の中のIを直すのは範囲の後ろだけ():
    """前側まで直すと，印字 `III(+-4)` が黙って `III(1-4)` になる

    `+` を `l` と読むことがあり，`1` にすると別の値になったまま通る．
    前側が読めないセルは値にせず，目視へ回す(2026-09-05 に目視で見つけた)．
    """
    assert ct.correct_constancy("+(+-I") == "+(+-1)"      # 後ろ側は直す
    assert ct.correct_constancy("III(l-4)") is None       # 前側は直さない


@pytest.mark.parametrize("text", ["VI(+-3)", "X(+)", "6(+)", "III()", "III(9)"])
def test_常在度にならない形はNone(text):
    assert ct.correct_constancy(text) is None


def test_常在度は被度としても値になる():
    got = ct.correct_comp("IV(+-3)")
    assert got == {"corrected": "IV(+-3)", "status": "OK"}


# --- 階層 ---------------------------------------------------------------

@pytest.mark.parametrize("text, corrected, status", [
    ("K", "K", "OK"),
    ("B1", "B1", "OK"),
    ("S2", "S2", "OK"),
    ("5", "S", "OK"),               # S を 5 と読む
    ("S・K", "S;K", "multi"),        # 階層が複数．行は分けず目視を促す
    ("S;K", "S;K", "multi"),        # 約束どおり `;` で区切っても壊れない
    ("あ", "あ", "Need Check"),
    ("", "", None),
])
def test_階層(text, corrected, status):
    assert ct.correct_layer(text) == {"corrected": corrected, "status": status}


# --- 表頭の値 -----------------------------------------------------------

@pytest.mark.parametrize("key, text, corrected", [
    ("plot_no", "l0", "10"),
    ("area", "100m². Höhe ü. Meer", "100"),     # 後ろのドイツ語を巻き込まない
    ("altitude", "640m", "640"),
    ("aspect", "N巳", "NE"),
    ("date", "'83 5 21", "1983-05-21"),
    ("field_no", "S0-216", "SO-216"),           # 数字に先頭のゼロは付かない
])
def test_表頭の値を項目ごとに直す(key, text, corrected):
    got = ct.correct_header_value(key, text)
    assert got["corrected"] == corrected
    assert got["status"] == "OK"


@pytest.mark.parametrize("key, text", [
    ("slope", "409"),        # 40° の `°` を 9 と読んだ形
    ("veg_cover", "509"),    # 50% の `%` を 9 と読んだ形
])
def test_上限を超える値は目視へ回す(key, text):
    assert ct.correct_header_value(key, text)["status"] == "Need Check"


def test_先頭のゼロは小数点の読み落としとみなす():
    """`0 6` は 0.6 と 6 のどちらか決められないので，直さず目視へ回す"""
    assert ct.correct_number("06")["status"] == "Need Check"


# --- 振り分け -----------------------------------------------------------

def test_文章の領域は素通しする():
    """`None` にすると，部品ごとには動くのに通しでは何も出ない壊れ方をする"""
    got = ct.correct_cell("header", "Feld-Nr. 調査番号")
    assert got["corrected"] == "Feld-Nr. 調査番号"


def test_読めなかったセルは落ちない():
    assert ct.correct_cell("comp", None) == {"corrected": None, "status": None}
    assert ct.correct_cell("species_col", float("nan"))["status"] is None
