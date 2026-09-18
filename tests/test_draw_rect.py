"""格子の絵 (overlay) を描く `draw_rect` の試験"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import draw_rect


def _page(path, size=(60, 40)):
    Image.new('RGB', size, 'white').save(path)
    return str(path)


def _rects(src, obj='comp', note=None):
    return {'source_image': src, 'x1': 5, 'y1': 5, 'x2': 30, 'y2': 20,
            'obj_name': obj, 'note': note}


def test_画像が複数なら全部を描く(tmp_path):
    """以前はループの中で return していて，最初の 1 枚しか描かなかった"""
    a, b = _page(tmp_path / 'a.png'), _page(tmp_path / 'b.png')
    got = draw_rect.draw_rects_df(pd.DataFrame([_rects(a), _rects(b)]))
    assert list(got) == [a, b]
    for img in got.values():
        assert (np.asarray(img) != 255).any()   # 何か描かれている


def test_開けない画像は飛ばす(tmp_path):
    a = _page(tmp_path / 'a.png')
    got = draw_rect.draw_rects_df(pd.DataFrame(
        [_rects(str(tmp_path / 'none.png')), _rects(a)]))
    assert list(got) == [a]


def test_注記は目立つ色を先に採る():
    assert draw_rect.note_color('interpolated;on_text') == 'red'
    assert draw_rect.note_color('snapped') == 'orange'
    assert draw_rect.note_color('row_fixed') is None


def test_注記が空やNaNなら色を付けない():
    assert draw_rect.note_color('') is None
    assert draw_rect.note_color(None) is None
    assert draw_rect.note_color(float('nan')) is None
