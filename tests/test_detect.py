"""検出の結果を表にする (comptea/detect.py)

ultralytics そのものは使わず，返りの形だけを真似た偽物で確かめる．
"""
import pandas as pd

from comptea import detect


def test_箱の列を4つの座標に分ける():
    df = pd.DataFrame({'name': ['comp', 'sname'], 'class': [0, 3],
                       'confidence': [0.9, 0.5],
                       'box': [{'x1': 10.4, 'y1': 20.6, 'x2': 30.0, 'y2': 40.49},
                               {'x1': 1.0, 'y1': 2.0, 'x2': 3.0, 'y2': 4.0}]})
    got = detect.split_box_column(df)
    assert list(got['obj_name']) == ['comp', 'sname']
    assert list(got['obj_class']) == [0, 3]
    assert got.loc[0, ['x1', 'y1', 'x2', 'y2']].tolist() == [10, 21, 30, 40]
    assert 'box' not in got.columns


def test_検出が0件なら決まった列の空の表():
    got = detect.split_box_column(pd.DataFrame())
    assert got.empty and list(got.columns) == detect.EMPTY_COLS


class _Boxes:
    def __init__(self, cls, conf):
        self.cls, self.conf = _L(cls), _L(conf)


class _L(list):
    def tolist(self):
        return list(self)


class _Res:
    names = {0: 'comp', 1: 'col'}

    def __init__(self, cls, conf):
        self.boxes = _Boxes(cls, conf)

    def __getitem__(self, keep):
        return _Res([self.boxes.cls[i] for i in keep],
                    [self.boxes.conf[i] for i in keep])


def test_クラスごとの閾値で絞る():
    res = [_Res([0, 0, 1, 1], [0.5, 0.2, 0.25, 0.15])]
    got = detect.filter_by_conf(res, 0.3, {'col': 0.2})
    assert got[0].boxes.cls == [0, 1]         # comp は 0.3，col は 0.2 で切る
    assert got[0].boxes.conf == [0.5, 0.25]


def test_クラス別の閾値が無ければそのまま():
    res = [_Res([0], [0.1])]
    assert detect.filter_by_conf(res, 0.3) is res
