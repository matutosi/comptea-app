"""種名の辞書との突き合わせ

辞書は `comptea/j_name.txt`・`s_name.txt`(維管束植物和名チェックリスト)．
公開しているファイルなので，実データが無くても走る．
"""
import pytest

from comptea import correct_text as ct


def test_辞書にある名前はそのまま通る():
    assert ct.correct_name("ススキ", target="j_name")["status"] == "OK"


def test_濁点だけ違う候補を先に採る():
    """古い印刷は「クロヅル」を「クロツル」と組む

    先頭からの一致だけで選ぶと，同じ距離 1 の「クロツグ」(別属の別種)に
    化ける．距離では分けられない(2026-09-01)．
    """
    got = ct.correct_name("クロツル", target="j_name")
    assert "クロヅル" in got["corrected"]
    assert "クロツグ" not in got["corrected"]


def test_離れた候補は採らない():
    """辞書に無い名前(異名・旧学名)を，近いだけの別種にすり替えない

    `Actinidia arguta` が距離 3 の `Actinidia rufa` になっていた(2026-09-01)．
    元の印字を残して目視へ回す．
    """
    got = ct.correct_name("Zzzxx qqqvvv", target="s_name")
    assert got == {"corrected": "Zzzxx qqqvvv", "status": "Need Check"}


def test_一文字の読みは完全一致だけ採る():
    """1 文字どうしは必ず距離 1 になる(2026-09-06)

    辞書にある 1 文字の名前が総なめで候補になり(`口` → `イ;桂;樟`)，
    同じ文字列が何行にも並んで重複の検査が鳴っていた．
    実物は `モミ`・`カヤ` で，OCR が 1 文字しか拾えていなかった．
    """
    got = ct.correct_name("口", target="j_name")
    assert got["status"] == "Need Check"
    assert ";" not in got["corrected"]


@pytest.mark.parametrize("text", [
    "ミカン科の一種",     # 実データ(tab4_sz1856)
    "アザミ属の1種",      # kinki_061 の1回出現種
    "スゲ属の種",         # 数を書かない形
    "sp.",
])
def test_種まで決まっていない記載は辞書に無くて当たり前(text):
    """辞書 27,127 件に `属`・`科` を含む和名は 1 件も無い(2026-09-08 に数えた)

    距離に回すと，近いだけの別種に化けるか，永久に目視へ残る．
    印字として正しい値なので，そのまま通す．
    """
    got = ct.correct_name(text, target="j_name")
    assert got == {"corrected": text, "status": "OK"}


def test_属や科を含んでいても普通の和名は辞書で見る():
    """`ヒノキ` のような普通の名前まで素通しにしない"""
    assert ct.correct_name("ヒノキ", target="j_name")["status"] == "OK"
    got = ct.correct_name("アザミ属", target="j_name")
    assert got["status"] != "OK" or got["corrected"] != "アザミ属"


def test_和名に混じった空白は落とす():
    """辞書 27,127 件に空白を含む和名は 0 件(2026-09-08 に数えた)

    `サカ キ` が距離 1 で `サカキ` に当たっていた．掃除すれば完全一致になる．
    """
    got = ct.correct_name("サカ キ", target="j_name")
    assert got == {"corrected": "サカキ", "status": "OK"}


def test_和名に混じったひらがなは完全一致のときだけ落とす():
    """辞書にも `西湖の葭` が 1 件あるので，無条件には落とせない

    `のツタ` は落とすと `ツタ` に完全一致する(実データ．kinki_047)．
    """
    got = ct.correct_name("のツタ", target="j_name")
    assert got == {"corrected": "ツタ", "status": "OK"}


def test_ひらがなを落としても辞書に無ければ落とさない():
    """落とした結果が辞書に無いなら，印字を残して距離の判断へ回す"""
    got = ct.correct_name("のあいうツツ", target="j_name")
    assert "のあいう" in got["corrected"] or got["status"] == "Need Check"


def test_学名の空白は語の区切りなので詰めない():
    got = ct.correct_name("Actinidia  arguta", target="s_name")
    assert got["corrected"] == "Actinidia arguta"
    assert got["status"] == "OK"


def test_短い読みは濁点以外の違いでは採らない():
    """`マソウ` が `ロソウ` になっていた(2026-09-07．実データ)

    辞書 27,027 件の和名を 1 文字だけ壊して測ると，別種に化ける率は
    4 字 8.3% に対して 5 字 1.6% で，**4 字と 5 字のあいだで一桁落ちる**．
    印字はそのまま残し，候補は `suggest` に入れて目視へ回す．
    """
    got = ct.correct_name("マソウ", target="j_name")
    assert got["corrected"] == "マソウ"
    assert got["status"] == "Need Check"
    assert got.get("suggest")


def test_短くても濁点だけの違いなら採る():
    """濁点・半濁点だけの違いは，長さによらず化ける率が 1% 前後

    実データで 4 字以下が直っていた件は，ほとんどがこれだった
    (`イヌピワ` → `イヌビワ`，`ミソシダ` → `ミゾシダ`)．
    """
    got = ct.correct_name("イヌピワ", target="j_name")
    assert got["corrected"] == "イヌビワ"
    assert got["status"] == "suggested"
    assert "suggest" not in got


def test_長い読みはこれまでどおり採る():
    got = ct.correct_name("ノリウッギ", target="j_name")
    assert got["corrected"] == "ノリウツギ"
    assert got["status"] == "suggested"


def test_候補が多すぎるときは並べない():
    """短い語は距離 3 以内に何百と当たり，連ねると表を壊す"""
    got = ct.correct_name("随伴種", target="j_name")
    assert got["status"] == "Need Check"
    assert len(got["corrected"]) < 20


def test_辞書の見出しの頭に一致すれば正しい名前とみなす():
    """辞書は種内分類群まで載せており，種名そのものが無いことがある

    `Actinidia arguta f. megalocarpa` はあるのに `Actinidia arguta` が無い．
    組成表は種名で書かれることが多い(2026-09-01)．
    """
    assert ct.correct_name("Actinidia arguta")["status"] == "OK"


def test_known_namesは辞書にある名前だけ真を返す():
    """種群の見出しや，候補を連ねたものは種ではない(2026-09-06)"""
    got = ct.known_names(["ススキ", "亜群集区分種", "クグ;クコ", None])
    assert got == [True, False, False, False]


def test_空文字はNoneを返す():
    assert ct.correct_name("") is None
