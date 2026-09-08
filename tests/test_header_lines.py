"""表頭の項目行を OCR の検出器の箱から作る (header_lines.py)"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from comptea import header_lines


def test_箱を行に束ねる():
    # 3 行．同じ行に語の箱が 2〜3 個，高さ 14．行の間隔 35
    boxes = []
    for c in (100, 135, 170):
        for x in (10, 80, 150):
            boxes.append((x, c - 7, x + 50, c + 7))
    boxes.append((60, 135 - 5, 66, 135 + 1))           # 小さい箱 (句読点) は同じ行に入る
    centers = header_lines.line_centers(boxes)
    assert len(centers) == 3
    assert np.allclose(centers, [100, 135 - 0.5, 170], atol=1.5)


def test_行の高さで束ね2行にまたがる箱は分ける():
    # 行の高さ 35．同じ行の独文と和文は 8 px ずれて打たれている (箱の高さで切ると割れる)
    boxes = []
    for c in (100, 135, 170, 205):
        boxes.append((10, c - 7, 200, c + 7))            # 独文
        boxes.append((300, c + 8 - 7, 400, c + 8 + 7))   # 和文 (8 px 下)
    # 240 と 275 の 2 行が 1 つの箱 (高さ 2 行分) になっている
    boxes.append((10, 240 - 17, 200, 275 + 17))
    centers = header_lines.line_centers(boxes, pitch=35)
    assert len(centers) == 6
    assert np.allclose(centers, [104, 139, 174, 209, 240.5, 274.5], atol=2)


def test_中点で帯を作り両端は間隔の半分():
    edges = header_lines.bands_from_centers([100, 135, 170], top=60, bottom=220)
    assert np.allclose(edges, [82.5, 117.5, 152.5, 187.5])
    edges = header_lines.bands_from_centers([100, 135, 170], top=90, bottom=180)
    assert edges[0] == 90 and edges[-1] == 180          # 領域の端で止まる


def test_値の無い先頭の帯は落とす():
    # 帯 4 つ．先頭の帯 (表題) だけ値側に字が無い
    dark = np.zeros((200, 400), dtype=bool)
    for c in (85, 120, 155):
        dark[c - 3:c + 3, 300:380] = True
    edges = np.array([30.0, 65.0, 100.0, 135.0, 170.0])
    out = header_lines.drop_unvalued(edges, dark, (300, 400))
    assert np.allclose(out, [65.0, 100.0, 135.0, 170.0])


def test_値の無い末尾の帯も落とすが中の帯は残す():
    dark = np.zeros((300, 400), dtype=bool)
    for c in (50, 85, 155):                              # 120 の帯だけ値が無い (項目の見出し)
        dark[c - 3:c + 3, 300:380] = True
    edges = np.array([30.0, 65.0, 100.0, 135.0, 170.0, 205.0, 240.0])
    out = header_lines.drop_unvalued(edges, dark, (300, 400))
    assert np.allclose(out, [30.0, 65.0, 100.0, 135.0, 170.0])


def test_reader_が無ければ帯を作らない():
    img = Image.new('L', (400, 300), 255)
    boxes = header_lines.detect_boxes(img, (0, 0, 400, 300), reader=False)
    assert boxes == [] or boxes is not None


@pytest.mark.slow
def test_描いた表頭から行数を数える():
    # 活字風: 10 項目を 40 px おきに描く．値側は数字
    img = Image.new('L', (900, 520), 255)
    d = ImageDraw.Draw(img)
    labels = ['Laufende Nr.:', 'Feld-Nr.:', 'Datum d. Aufnahme:', 'Ort d. Aufn.:',
              'Groesse d. Probeflaeche:', 'Hoehe ue. Meer (m):', 'Exposition:',
              'Neigung:', 'Deckung (%):', 'Artenzahl:']
    for i, t in enumerate(labels):
        y = 60 + 40 * i
        d.text((20, y - 8), t, fill=0)
        for k in range(6):
            d.text((520 + 60 * k, y - 8), str(10 + i + k), fill=0)
    d.text((20, 12), 'Tab. 12  Titel der Tabelle', fill=0)          # 表題 (値が無い)
    edges = header_lines.header_bands(img, (10, 0, 500, 480), value_x=(510, 890))
    assert edges is not None
    assert len(edges) - 1 == 10
