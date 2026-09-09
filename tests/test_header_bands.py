"""表頭の帯の境は「次の項目名の直前」に置く (header_lines.bands_from_spans)

1 つの項目の値は 2〜3 行にわたることがある (「調査年月日」は '83 / 6 / 8 の 3 行)．
境を隣り合う行の中点に置くと，2 行目・3 行目が次の項目の帯に落ちる (2026-09-09 ユーザ指示)．

    -------------------
    通し番号   AA  AA
               01  02
    -------------------
    年月日     10  10
               10  15
    -------------------
"""
import numpy as np

from comptea import header_lines


PITCH = 30.0


def _box(y1, y2, x1=10, x2=200):
    return (float(x1), float(y1), float(x2), float(y2))


def test_値が2行にわたる項目でも境は次の項目名の直前():
    # 項目名は y=100 と y=190 (あいだの y=145 は値の 2 行目．項目名の側には無い)
    spans = header_lines.line_spans([_box(100, 120), _box(190, 210)], pitch=PITCH)
    edges = header_lines.bands_from_spans(spans, 90, 300, pitch=PITCH)
    assert len(edges) == 3
    assert 180 <= edges[1] <= 190            # 2 行目 (145) は前の帯に残る
    assert edges[0] <= 100 and edges[-1] == 300


def test_行の中心の中点には置かない():
    spans = header_lines.line_spans([_box(100, 120), _box(190, 210)], pitch=PITCH)
    edges = header_lines.bands_from_spans(spans, 90, 300, pitch=PITCH)
    mid = (110 + 200) / 2.0
    assert abs(edges[1] - mid) > 20


def test_前の行の字は切らない():
    # 行間が狭いとき，境が前の行の下端より上に来てはいけない
    spans = header_lines.line_spans([_box(100, 128), _box(130, 158)], pitch=PITCH)
    edges = header_lines.bands_from_spans(spans, 90, 300, pitch=PITCH)
    assert edges[1] >= 128


def test_境は昇順():
    spans = header_lines.line_spans(
        [_box(100, 120), _box(150, 170), _box(210, 230)], pitch=PITCH)
    edges = header_lines.bands_from_spans(spans, 90, 300, pitch=PITCH)
    assert np.all(np.diff(edges) >= 0)


def test_line_spansは行ごとの上端と下端を返す():
    spans = header_lines.line_spans(
        [_box(100, 120, 10, 90), _box(102, 118, 110, 200), _box(160, 180)], pitch=PITCH)
    assert len(spans) == 2
    assert spans[0][1] == 100 and spans[0][2] == 120     # 同じ行の 2 つの箱を束ねる
    assert spans[1][1] == 160


def test_高い箱は行の数に分ける():
    # 行間の狭い 2 行が 1 つの箱になった場合 (検出器の癖)
    spans = header_lines.line_spans([_box(100, 190)], pitch=PITCH)
    assert len(spans) == 3
    assert spans[0][1] == 100 and spans[-1][2] == 190


def test_行が1つなら帯を作らない():
    spans = header_lines.line_spans([_box(100, 120)], pitch=PITCH)
    assert header_lines.bands_from_spans(spans, 90, 300, pitch=PITCH) is None
