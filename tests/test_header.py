"""表頭の項目名と，文章形式の表頭・1回出現種の解析

項目名の 1 文字の誤読で項目が丸ごと落ちていた(2026-09-03)．
**部分文字列で寄せてはいけない**という判断が要で，そこを歯止めにする．
"""
import pytest

from comptea import parse_text
from comptea import plot_table as pt


# --- 項目名を出力の列名に寄せる -----------------------------------------

@pytest.mark.parametrize("text, want", [
    ("通し番号", "plot_no"),
    ("調査面積", "area"),
    ("海抜高", "altitude"),
    ("方位", "aspect"),
    ("傾斜", "slope"),
    ("出現種数", "n_species"),
    # 実物で確かめた誤読 6 件(2026-09-03)．どれも 1 文字の誤り
    ("通し務号", "plot_no"),
    ("調査面称", "area"),
    ("出現穂数", "n_species"),
    ("低木腐の植被率", "低木層_cover"),
    ("草本厨の高さ", "草本層_height"),
    ("挙本國の植被率", "草本層_cover"),
])
def test_項目名を寄せる(text, want):
    assert pt.match_item(text) == want


def test_全体の距離で比べる():
    """**部分文字列で寄せてはいけない**(2026-09-03)

    `調査面称` は先頭 3 文字 `調査面` が `調査日` と距離 1 なので，
    窓を滑らせると `調査年月日` に当たる．正しくは `調査面積`．
    """
    assert pt.match_item("調査面称") == "area"


def test_階層の項目は語彙より先に見る():
    """`低木厨植被率` は `植被率` を含む

    先に語彙を見ると，階層を持たない `veg_cover` に落ちる(2026-09-03)．
    """
    assert pt.match_item("低木厨植被率") == "低木層_cover"
    assert pt.match_item("植被率") == "veg_cover"      # 階層が無ければ植生全体


def test_階層を決められないときは植生全体に落とさない():
    """`挙本國の植被率` を `veg_cover` にすると，階層が消えたまま黙って表に入る

    頭が読めていれば階層はそのまま活かし(`ｘｙｚ層` は階層として残す)，
    階層の名前として読めないときは判別不能を返す．
    """
    assert pt.match_item("ｘｙｚ層の植被率") == "xyz層_cover"   # NFKC で半角になる
    assert pt.match_item("ｘｙｚの植被率") is None


def test_階層の呼び方は取り違えない():
    """`低木第1層` と `低木第2層` は距離 1．緩めると取り違える"""
    assert pt.match_item("低木第1層の高さ") == "低木第1層_height"
    assert pt.match_item("低木第2層の高さ") == "低木第2層_height"


# --- 文章形式の表頭 -----------------------------------------------------

def test_文章形式の表頭から項目を取り出す():
    """値の終わりは「次の項目名が始まるところ」

    句読点で切ると `100m². Höhe` のように値の中の点で切れる．
    """
    text = ("Feld-Nr. 調査番号：SO-216, Größe d. Probefläche 調査面積：100m². "
            "Höhe ü. Meer 海抜高：640m. Exposition 方位：W.")
    got = parse_text.parse_header_text(text, pt.ITEM_ALIASES)
    assert got["field_no"].startswith("SO-216")
    assert got["area"].startswith("100m")
    assert "640m" in got["altitude"]


# --- 1回出現種の流し込み -------------------------------------------------

def test_1回出現種を1件ずつ取り出す():
    text = "in 1: Bidens pilosa コセンダングサ +・2, Cirsium maritimum ハマアザミ +"
    got = parse_text.parse_once_species(text)
    assert [g["j_name"] for g in got] == ["コセンダングサ", "ハマアザミ"]
    assert got[0]["plot"] == 1
    assert got[0]["comp_raw"] == "+;2"


def test_和名の無い断片は直前の種の別の階層():
    """`ヒメドコロ S—+, K—+` のように階層が 2 つ続く"""
    got = parse_text.parse_once_species("in 3: Dioscorea tokoro ヒメドコロ S-+, K-+")
    assert len(got) == 2
    assert {g["j_name"] for g in got} == {"ヒメドコロ"}
    assert {g["layer"] for g in got} == {"S", "K"}


def test_地点の印が無ければ空():
    assert parse_text.parse_once_species("種名がただ並ぶだけの文") == []
