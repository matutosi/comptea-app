"""黒画素から測る道具

実データが要らないよう，**印字を模した小さな画像**を組み立てて確かめる．
`head_strokes()` は，OCR が読み違えても正しい形の値になってしまう
`II` → `I` を見つけるためのもので，読みだけでは気づけない(2026-09-06)．
"""
import numpy as np
import pytest
from PIL import Image

import ink


def _blank(h=40, w=120):
    return np.zeros((h, w), dtype=bool)


def _bar(dark, x, w, y=10, h=18):
    """縦棒を1本置く(ローマ数字の `I` や括弧の代わり)"""
    dark[y:y + h, x:x + w] = True


# --- 白黒 2 値の画像 -----------------------------------------------------

def test_白黒2値の画像でも黒画素が取れる():
    """しきい値の比較を `<` にすると，0 と 255 しか無い画像で 0 件になる

    折り込みのスキャンで分かった(2026-09-02)．暗い側の組は `g <= t`．
    """
    a = np.full((10, 10), 255, dtype=np.uint8)
    a[2:5, 2:5] = 0
    dark = ink.binarize(Image.fromarray(a))
    assert dark.sum() == 9


# --- 頭の画数 -----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_括弧の左の字の数を数える(n):
    dark = _blank()
    x = 5
    for _ in range(n):                       # 頭のローマ数字(幅 8・高さ 18)
        _bar(dark, x, 8)
        x += 14                              # 字と字のあいだは 5-7 px
    _bar(dark, x, 5, y=7, h=24)              # 括弧は背が高くて細い
    assert ink.head_strokes(dark, (0, 0, 120, 40)) == n


def test_serifで割れたIも1つと数える():
    """serif 付きの `I` は上下の横棒に穴があき，3 本に割れて見える

    3 px までのすき間は同じ字としてつなぐ(字と字の間は 5-7 px)．
    """
    dark = _blank()
    _bar(dark, 5, 3)                          # 縦棒
    _bar(dark, 10, 3)                         # 2 px のすき間を空けて
    _bar(dark, 15, 3)
    _bar(dark, 30, 4, y=7, h=24)              # 括弧
    assert ink.head_strokes(dark, (0, 0, 120, 40)) == 1


def test_括弧が無ければNone():
    dark = _blank()
    _bar(dark, 5, 8)
    _bar(dark, 19, 8)
    assert ink.head_strokes(dark, (0, 0, 120, 40)) is None


def test_頭の字の形がそろわなければ数えない():
    """`(` が細くない字体では，括弧や汚れを数に入れてしまう

    形がそろっていることを条件に足して，11_p1 の誤りが 0 になった(2026-09-06)．
    """
    dark = _blank()
    _bar(dark, 5, 8)                          # 幅 8
    _bar(dark, 19, 16)                        # 幅 16 = 形がそろわない
    _bar(dark, 40, 5, y=7, h=24)
    assert ink.head_strokes(dark, (0, 0, 120, 40)) is None


def test_空の領域はNone():
    assert ink.head_strokes(_blank(), (0, 0, 0, 0)) is None


# --- 罫線 ---------------------------------------------------------------

def test_横罫線は縦の罫線もろとも除いて割合を測る():
    """幅 4-5 px の縦線は横にも 4-5 px 連なるので，`thick_ratio` を通る

    14_p5 は階層の列の中身が罫線だけで，29 セルが全滅していた(2026-09-03)．
    """
    dark = _blank(h=40, w=40)
    dark[:, 10:14] = True                     # 高さいっぱいの縦罫線
    assert ink.thick_ratio(dark, 0, 40, 0, 40) > 0.05
    assert ink.text_ratio(dark, 0, 40, 0, 40) < 0.01


def test_破線の行を見つける():
    """破線は短い連なりが一定の間隔で横に並ぶ(字の行はばらばら)"""
    dark = _blank(h=20, w=400)
    for x in range(0, 400, 20):               # 長さ 10・間隔 10 の連なり
        dark[10, x:x + 10] = True
    got = ink.dashed_rows(dark)
    assert got[10] and got.sum() == 1


# --- セルの枠線消し -----------------------------------------------------

def test_セルを横切る横線だけ消す():
    """枠線がセルに入ると，EasyOCR は `+` を `4` と読む(2026-09-04)"""
    a = np.full((30, 60), 255, dtype=np.uint8)
    a[5, :] = 0                               # セルを横切る横線
    got = np.asarray(ink.erase_box_lines(Image.fromarray(a)).convert('L'))
    assert (got[5] == 255).all()


def test_縦線は消さない():
    """タイプ打ちの `1` はまっすぐな縦棒で，高さの 6 割を超える

    細くて長いかたまりを全部消す案は，単独の `+`・`1` まで消して改悪だった．
    """
    a = np.full((30, 60), 255, dtype=np.uint8)
    a[:, 30] = 0                              # セルの中ほどの縦棒
    got = np.asarray(ink.erase_box_lines(Image.fromarray(a)).convert('L'))
    assert (got[:, 30] == 0).all()
