"""複数の読み手を工程に組み込む (`--reader multi`)

EasyOCR でセルごとに読んだうえで，**入っている読み手で領域を読み直し**，
クラスごとの順で質の通る読みを採る．どこから採ったかは `read_by` に残す．
"""
import pandas as pd
import pytest

from comptea.pipeline import read as read_mod


def _df(rows):
    return pd.DataFrame(
        [{'cell_id': c, 'obj_name': o, 'x1': a, 'y1': b, 'x2': x, 'y2': y,
          'source_image': 'a.png', 'text': t}
         for c, o, a, b, x, y, t in rows])


class Fake:
    """領域を読む偽の読み手．`found` をそのまま返す"""

    def __init__(self, found, ok=True):
        self.found = found
        self.ok = ok
        self.calls = 0

    def available(self):
        return self.ok

    def read_boxes(self, img, box=None):
        self.calls += 1
        return self.found


def test_選べる読み方に_multi_がある():
    args = read_mod.parse_args(['w', '--reader', 'multi'])
    assert args.reader == 'multi'


def test_入っている読み手だけを使う():
    a, b = Fake([], ok=True), Fake([], ok=False)
    assert read_mod.usable_readers({'a': a, 'b': b}) == {'a': a}


def test_領域の読みでセルを差し替える():
    """学名は yomitoku を先に見る (`read_region.ORDER`)"""
    df = _df([(1, 'sname', 0, 0, 100, 50, 'アカマツ'),
              (2, 'sname', 0, 50, 100, 100, '')])
    fake = Fake([((10, 10, 60, 40), 'Pinus densiflora'),
                 ((10, 60, 60, 90), 'Abies firma')])
    out = read_mod.blend(None, df, {'yomi': fake}, ok=lambda t: True)
    got = dict(zip(out['cell_id'], out['text']))
    assert got == {1: 'Pinus densiflora', 2: 'Abies firma'}
    by = dict(zip(out['cell_id'], out['read_by']))
    assert by[1] == 'yomi' and by[2] == 'yomi'


def test_通らない読みは元のまま():
    """質が通らなければ EasyOCR の読みを残す"""
    df = _df([(1, 'species_col', 0, 0, 100, 50, 'アカマツ')])
    fake = Fake([((10, 10, 60, 40), 'ヌケマシ')])
    out = read_mod.blend(None, df, {'yomi': fake},
                         ok=lambda t: t == 'アカマツ')
    assert out['text'].iloc[0] == 'アカマツ'
    assert out['read_by'].iloc[0] == 'easy'


def test_対象のクラスだけを読み直す():
    """組成のような細かい格子は，いまのところ対象にしない"""
    df = _df([(1, 'comp', 0, 0, 20, 20, '・')])
    fake = Fake([((5, 5, 15, 15), '+')])
    out = read_mod.blend(None, df, {'yomi': fake}, ok=lambda t: True)
    assert out['text'].iloc[0] == '・'
    assert fake.calls == 0


def test_読み手が無ければ何もしない():
    df = _df([(1, 'sname', 0, 0, 100, 50, 'アカマツ')])
    out = read_mod.blend(None, df, {}, ok=lambda t: True)
    assert out['text'].iloc[0] == 'アカマツ'
