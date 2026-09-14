"""段階 1 の入口 (pipeline/grid.py) と，読みの下ごしらえ (ocr.py)

2026-09-15 に足した．`grid` は公開 14 のうち 11，`ocr` は 11 のうち 10 が
的で呼ばれていなかった．**画像と検出器が要らない所**だけをここで固める．
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from comptea import ocr
from comptea.pipeline import grid


# --- imgsz は画像の大きさから決める ----------------------------------------

def test_数で渡されたらそのまま():
    assert grid.resolve_imgsz(None, 1280) == 1280
    assert grid.resolve_imgsz(None, '2560') == 2560


def test_autoは画像の長辺から決める():
    """長辺 3300 px ↔ imgsz 1280 (学習時の縮尺に合わせる)"""
    im = Image.new('L', (2480, 3300), 255)
    assert grid.resolve_imgsz(im, 'auto', quiet=True) == 1280


def test_大きな紙面ほど大きくなる():
    small = grid.resolve_imgsz(Image.new('L', (2480, 3300), 255), 'auto',
                               quiet=True)
    big = grid.resolve_imgsz(Image.new('L', (6000, 9000), 255), 'auto',
                             quiet=True)
    assert big > small


def test_上限で頭打ちになる():
    """**A0 級の折込は上限で止まる**．学習時より縮尺が小さく，行が取れにくい

    上限の値そのものは決め打ちにしない (`split_sheet` が持つ)．
    頭打ちかどうかは `imgsz_capped` が答える．
    """
    from comptea import split_sheet

    assert split_sheet.imgsz_capped(9344, 12873) is True
    assert split_sheet.imgsz_capped(2480, 3300) is False
    huge = grid.resolve_imgsz(Image.new('L', (9344, 12873), 255), 'auto',
                              quiet=True)
    assert huge % 32 == 0                     # 32 の倍数で返す


# --- まとめ ----------------------------------------------------------------

def _loc(n_rows=3, n_cols=2):
    rows = []
    for c in range(n_cols):
        for r in range(n_rows):
            rows.append({'obj_name': 'comp', 'block': 1, 'col': c, 'row': r,
                         'x1': 100.0 + c * 50, 'x2': 150.0 + c * 50,
                         'y1': 200.0 + r * 30, 'y2': 230.0 + r * 30,
                         'note': ''})
    rows.append({'obj_name': 'sname', 'block': 1, 'col': 0, 'row': 0,
                 'x1': 0.0, 'x2': 100.0, 'y1': 200.0, 'y2': 230.0, 'note': ''})
    return pd.DataFrame(rows)


class _Args:
    image = 'a.png'
    weights = None
    conf = 30
    conf_col = 20
    imgsz = 'auto'


def test_まとめに行と列の数が出る():
    got = grid.summarize(pd.DataFrame([{'obj_name': 'row', 'confidence': 0.9}]),
                         _loc(), _Args())
    assert '3' in got and '2' in got          # 3 行 2 列


def test_まとめは文字列を返す():
    got = grid.summarize(pd.DataFrame(columns=['obj_name', 'confidence']),
                         _loc(), _Args())
    assert isinstance(got, str) and got


# --- 格子の焼き込み --------------------------------------------------------

def test_行番号と列番号を焼き込む():
    """**画像は書き換えて返す** (overlay に使う)"""
    im = Image.new('RGB', (400, 400), (255, 255, 255))
    before = np.asarray(im).copy()
    got = grid.label_grid(im, _loc())
    assert got is im                          # 同じ画像に描く
    assert not np.array_equal(before, np.asarray(got))


def test_格子が空でも落ちない():
    im = Image.new('RGB', (100, 100), (255, 255, 255))
    empty = pd.DataFrame(columns=['obj_name', 'block', 'col', 'row',
                                  'x1', 'x2', 'y1', 'y2', 'note'])
    assert grid.label_grid(im, empty) is im


# --- 読みの下ごしらえ ------------------------------------------------------

def test_白い余白を落とす():
    im = Image.new('L', (200, 100), 255)
    im.paste(0, (80, 40, 120, 60))            # 真ん中に字
    got = ocr.trim_image(im)
    assert got.size[0] < im.size[0]           # 幅は縮む
    assert got.size[0] > 40                   # **余白は足し直す** (ギリギリは読めない)


def test_真っ白なら触らない():
    im = Image.new('L', (200, 100), 255)
    got = ocr.trim_image(im)
    assert got.size == im.size


def test_余白を足す():
    im = Image.new('L', (50, 20), 255)
    got = ocr.add_margin(im, 5, 6, 7, 8)
    assert got.size == (50 + 6 + 8, 20 + 5 + 7)


def test_二値化すると白と黒だけになる():
    """返るのは mode '1' の画像 (配列にすると True / False)"""
    im = Image.new('L', (20, 20), 128)
    im.paste(30, (0, 0, 10, 20))
    got = ocr.binarize_image(im)
    assert got.mode == '1'
    arr = np.asarray(got)
    assert set(np.unique(arr)) <= {True, False}
    # **ディザ (誤差拡散) がかかる**ので，画素ごとには決まらない．
    # 濃い側が黒くなっていることだけを見る
    assert arr[:, :10].mean() < arr[:, 10:].mean()
