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


# --- 読めなかった組成のセルを読み直す --------------------------------------
#
# 2026-09-14 の実測: **読めなかった組成のセルは NDLOCR-Lite がよく読む**
# (22_p2 で 77%・17_p1 で 53%・05_p2 で 26%)．EasyOCR は 66 個中 5 個，
# yomitoku は 0 個だった．**1 セル 1 画像でまとめて渡す**と 0.6 秒/セル
# (1 セルずつ呼ぶと 15 秒)．

class _Crops:
    """まとめ読みの口だけを持つ偽の読み手"""

    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def available(self):
        return True

    def read_crops(self, img, boxes, pad=0):
        self.calls += 1
        return list(self.texts)[:len(list(boxes))]


def test_読めなかったセルだけ読み直す():
    df = pd.DataFrame([
        {'cell_id': 1, 'obj_name': 'comp', 'x1': 0, 'y1': 0, 'x2': 9, 'y2': 9,
         'corrected': '1;a', 'status': 'Need Check', 'note': ''},
        {'cell_id': 2, 'obj_name': 'comp', 'x1': 10, 'y1': 0, 'x2': 19, 'y2': 9,
         'corrected': '1', 'status': 'OK', 'note': ''},
    ])
    r = _Crops(['2・2'])
    got = read_mod.retry_cells(None, df, r)
    assert r.calls == 1
    assert got.loc[0, 'corrected'] == '2;2'
    assert got.loc[0, 'status'] == 'OK'
    assert 'ndl' in str(got.loc[0, 'note'])
    assert got.loc[1, 'corrected'] == '1'      # 読めていたセルは触らない


def test_読み直しても駄目なら元のまま():
    df = pd.DataFrame([
        {'cell_id': 1, 'obj_name': 'comp', 'x1': 0, 'y1': 0, 'x2': 9, 'y2': 9,
         'corrected': '1;a', 'status': 'Need Check', 'note': ''},
    ])
    got = read_mod.retry_cells(None, df, _Crops(['のののの']))
    assert got.loc[0, 'corrected'] == '1;a'
    assert got.loc[0, 'status'] == 'Need Check'


def test_組成のセルだけ読み直す():
    df = pd.DataFrame([
        {'cell_id': 1, 'obj_name': 'sname', 'x1': 0, 'y1': 0, 'x2': 9, 'y2': 9,
         'corrected': 'Zzz', 'status': 'Need Check', 'note': ''},
    ])
    r = _Crops(['1'])
    got = read_mod.retry_cells(None, df, r)
    assert r.calls == 0
    assert got.loc[0, 'corrected'] == 'Zzz'


def test_読み手が無ければ何もしない():
    df = pd.DataFrame([
        {'cell_id': 1, 'obj_name': 'comp', 'x1': 0, 'y1': 0, 'x2': 9, 'y2': 9,
         'corrected': '1;a', 'status': 'Need Check', 'note': ''},
    ])
    got = read_mod.retry_cells(None, df, None)
    assert got.loc[0, 'corrected'] == '1;a'
