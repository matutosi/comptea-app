"""表にならない切れ端を，箱の段階で落とす (split_sheet.MIN_HEIGHT)

2026-09-15 に足した．折込 23 枚で，面積の歯止め (`MIN_AREA` = 0.01) を
**通り抜ける低い切れ端**が 2 つ残っていた (16_p1 は 1782x656 で面積比 0.014，
21_p2 は 3041x527 で 0.013)．格子を組む段で落ちるので実害は無かったが，
箱の数が真値と 2 枚ぶん食い違っていた．

**実測で境を決めた**: 本物の表 68 箱の高さ比は 0.109〜0.987，切れ端は
0.041・0.059．0.08 はどちらからも離れている．
"""
import numpy as np
import pytest

from comptea import split_sheet


def _sheet(h=4000, w=4000, boxes=()):
    """指定した矩形にインクを置いた紙面 (True が黒)

    **紙面は大きくとる**．切り出しに付ける余白 (`MARGIN` = 40 px) は
    実物 (A0 は 12873 px) では無視できるが，小さな合成紙面では比を押し上げる．
    """
    dark = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        dark[y1:y2, x1:x2] = True
    return dark


def test_低すぎる箱は返さない():
    """紙面の高さの 8% に満たない箱は表とみなさない"""
    dark = _sheet(boxes=[(400, 400, 3600, 2000),    # 本物 (高さ比 0.40)
                         (400, 3600, 3600, 3700)])  # 切れ端 (0.025)
    got = split_sheet.find_tables(dark)
    assert len(got) == 1
    assert got[0][1] < 500                          # 残ったのは上の箱


def test_歯止めを外せば両方返す():
    """部分画像に切り出してやり直す経路 (`resplit_parts`) はこちらで呼ぶ"""
    dark = _sheet(boxes=[(400, 400, 3600, 2000), (400, 3600, 3600, 3700)])
    got = split_sheet.find_tables(dark, min_height=0)
    assert len(got) == 2


def test_本物の小さい表は残す():
    """いちばん小さい本物は高さ比 0.109 (s01115_08_p5)"""
    dark = _sheet(boxes=[(400, 400, 3600, 2400),
                         (400, 3000, 3600, 3400)])  # 高さ比 0.12
    got = split_sheet.find_tables(dark)
    assert len(got) == 2


def test_境は実測に合わせてある():
    assert split_sheet.MIN_HEIGHT == 0.08
    # 切れ端 0.041・0.059 と，本物の最小 0.109 のあいだ
    assert 0.059 < split_sheet.MIN_HEIGHT < 0.109


def test_面積の歯止めは今までどおり():
    """小さすぎる汚れは面積で落ちる (高さが足りていても)"""
    dark = _sheet(boxes=[(400, 400, 3600, 2400),
                         (400, 3000, 460, 3500)])   # 細長い汚れ
    got = split_sheet.find_tables(dark)
    assert len(got) == 1
