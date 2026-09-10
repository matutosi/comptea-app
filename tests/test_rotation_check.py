"""紙面が 90 度回して組まれていないかの検査 (split_sheet.check_rotation)

kinki_014 は本のページが横倒しで，種名が下から上へ読む向き (2026-09-10 ユーザ指摘 22・42)．
157 枚で回転と判定されるのはこの 1 枚だけ (横の帯 3 本・縦の帯 39 本．比 0.077)．
"""
import numpy as np
from PIL import Image

from comptea import split_sheet as ss


def _lines(size=(2000, 2600), horizontal=True, n=40):
    img = Image.new('L', size, 255)
    px = img.load()
    w, h = size
    for i in range(n):
        # 行に沿う向きは字がつながる (語間は行ごとにずれるので，帯にならない)
        if horizontal:
            y = 100 + i * 55
            for xx in range(200 + (i * 7) % 20, w - 200):
                for yy in range(y, y + 22):
                    px[xx, yy] = 0
        else:
            x = 100 + i * 45
            for yy in range(200 + (i * 7) % 20, h - 200):
                for xx in range(x, x + 22):
                    px[xx, yy] = 0
    return img


def test_正しい向きは警告しない():
    assert ss.check_rotation(_lines(horizontal=True)) == []


def test_横倒しは警告する():
    warns = ss.check_rotation(_lines(horizontal=False))
    assert warns and '90 度' in warns[0]


def test_小さい切れ端は判定しない():
    assert ss.check_rotation(_lines(size=(900, 900), horizontal=False, n=15)) == []
