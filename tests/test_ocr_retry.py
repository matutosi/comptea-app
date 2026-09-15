"""読み手を偽物にして，読み直しの決まりを確かめる (ocr.py)

2026-09-15 に足した．`ocr.py` の公開 12 のうち 9 が的で呼ばれていなかった．
EasyOCR は重いので，**読み手だけを差し替える**．
`readtext()` は字を探してから読み，`recognize()` は与えた画像を 1 行として
読む — 読み直しが後者を使う理由は `retry_comp_cell` の説明にある．
"""
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from comptea import ocr


class FakeReader:
    """渡された字種と呼ばれ方を覚える読み手"""

    def __init__(self, texts=('+',), conf=0.9):
        self.texts = list(texts)
        self.conf = conf
        self.calls = []

    def _out(self, allowlist):
        self.calls.append(allowlist)
        return [([[0, 0], [9, 0], [9, 9], [0, 9]], t, self.conf)
                for t in self.texts]

    def readtext(self, arr, allowlist=None):
        return self._out(allowlist)

    def recognize(self, arr, allowlist=None):
        return self._out(allowlist)


def _cell(text='', size=(60, 40)):
    img = Image.new('RGB', size, 'white')
    if text:
        ImageDraw.Draw(img).text((10, 10), text, fill='black')
    return img


# --- 読み手を使い回す ------------------------------------------------------

def test_読み手は一度だけ作る(monkeypatch):
    """**import では作らない** (入っていない所でも取り込めるように)"""
    made = []

    class Mod:
        @staticmethod
        def Reader(langs):
            made.append(langs)
            return 'reader'

    monkeypatch.setattr(ocr, '_READER', None)
    monkeypatch.setitem(sys.modules, 'easyocr', Mod)
    assert ocr.get_reader() == 'reader'
    assert ocr.get_reader() == 'reader'
    assert len(made) == 1
    ocr._READER = None


# --- 1 セルを読む ----------------------------------------------------------

def test_箱の上下左右が逆でも読める():
    """格子の端では高さや幅が負になる箱ができる (s01115_18_p2)"""
    r = FakeReader()
    text, img = ocr.ocr_image(_cell('1'), (50, 30, 10, 5), reader=r)
    assert text and img.width > 0


def test_中身の無いセルは読ませない():
    """幅か高さが 0 の画像を渡すと EasyOCR の中で落ちる"""
    r = FakeReader()
    text, _img = ocr.ocr_image(_cell(''), (10, 10, 10, 30), reader=r)
    assert text == []
    assert r.calls == []                        # 読み手を呼んでいない


def test_字種を絞って読む():
    r = FakeReader(texts=['S'])
    ocr.ocr_image(_cell('S'), (0, 0, 60, 40), reader=r, allow=ocr.LAYER_ALLOW)
    assert r.calls[0] == ocr.LAYER_ALLOW


def test_絞って空なら絞らずに読み直す():
    """種群の見出しが列に掛かったセルは，絞ると空で返る"""
    r = FakeReader(texts=['  '])
    ocr.ocr_image(_cell('あ'), (0, 0, 60, 40), reader=r, allow=ocr.LAYER_ALLOW)
    assert r.calls == [ocr.LAYER_ALLOW, None]


# --- 組成部のセルの読み直し ------------------------------------------------

def test_値の形になったものだけ採る(monkeypatch):
    monkeypatch.setattr(ocr, 'get_reader', lambda: FakeReader(texts=['+']))
    assert ocr.retry_comp_cell(_cell('+'), (0, 0, 60, 40), thin=False) == '+'


def test_値にならない読みは捨てる(monkeypatch):
    monkeypatch.setattr(ocr, 'get_reader', lambda: FakeReader(texts=['xx']))
    assert ocr.retry_comp_cell(_cell('x'), (0, 0, 60, 40), thin=False) == ''


def test_薄いセルに数字は入れない(monkeypatch):
    """非出現の `・` を数字と読む誤りを止める"""
    monkeypatch.setattr(ocr, 'get_reader', lambda: FakeReader(texts=['3']))
    assert ocr.retry_comp_cell(_cell('.'), (0, 0, 60, 40), thin=True) == ''
    assert ocr.retry_comp_cell(_cell('.'), (0, 0, 60, 40), thin=False) == '3'


def test_狭い字種から順に試す(monkeypatch):
    r = FakeReader(texts=['+'])
    monkeypatch.setattr(ocr, 'get_reader', lambda: r)
    ocr.retry_comp_cell(_cell('+'), (0, 0, 60, 40), thin=False)
    assert r.calls == [ocr.COMP_ALLOW]          # 括弧付きまでは行かない


# --- 階層のセルの読み直し --------------------------------------------------

def test_階層は字数が合うときだけ採る(monkeypatch):
    """`S・K` で `S` だけ返ると，**K が黙って落ちる**"""
    monkeypatch.setattr(ocr, 'get_reader', lambda: FakeReader(texts=['S']))
    img = Image.new('RGB', (80, 40), 'white')
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, 20, 30), fill='black')     # 字が 2 つ写っている
    d.rectangle((50, 10, 60, 30), fill='black')
    assert ocr.retry_layer_cell(img, (0, 0, 80, 40)) == ''


def test_階層として読めなければ捨てる(monkeypatch):
    monkeypatch.setattr(ocr, 'get_reader', lambda: FakeReader(texts=['zz']))
    assert ocr.retry_layer_cell(_cell('z'), (0, 0, 60, 40)) == ''


def test_空の読みは採らない(monkeypatch):
    monkeypatch.setattr(ocr, 'get_reader', lambda: FakeReader(texts=['  ']))
    assert ocr.retry_layer_cell(_cell('S'), (0, 0, 60, 40)) == ''


# --- 表ぜんたいの読み直し --------------------------------------------------

def _layer_df(text):
    return pd.DataFrame([{'obj_name': 'layer', 'x1': 0, 'y1': 0,
                          'x2': 60, 'y2': 40, 'text': text, 'note': None}])


def test_読めた階層は読み直さない(monkeypatch):
    monkeypatch.setattr(ocr, 'retry_layer_cell',
                        lambda *a: pytest.fail('読み直してはいけない'))
    _df, n = ocr.retry_bad_layers(_layer_df('S'), _cell('S'))
    assert n == 0


def test_読めない階層だけ読み直して印を付ける(monkeypatch):
    monkeypatch.setattr(ocr, 'retry_layer_cell', lambda *a: 'K')
    df, n = ocr.retry_bad_layers(_layer_df('zz'), _cell('z'))
    assert n == 1
    assert df.at[0, 'text'] == 'K'
    assert 'retry' in df.at[0, 'note']           # 段階2の目視に回す


def test_列が無ければ何もしない():
    df, n = ocr.retry_bad_layers(pd.DataFrame({'text': ['S']}), _cell('S'))
    assert n == 0


# --- 常在度の頭を画数から直す ----------------------------------------------

def _roman_page(strokes):
    """セルごとに `strokes` 本の縦棒と括弧を置いた紙面 (True が黒)"""
    dark = np.zeros((40 * len(strokes), 80), dtype=bool)
    rows = []
    for k, n in enumerate(strokes):
        top = 40 * k
        x = 5
        for _ in range(n):
            dark[top + 10:top + 28, x:x + 8] = True
            x += 14
        dark[top + 6:top + 34, x + 4:x + 9] = True      # 開き括弧
        dark[top + 6:top + 34, 70:75] = True            # 閉じ括弧
        rows.append({'obj_name': 'comp', 'x1': 0, 'y1': top,
                     'x2': 80, 'y2': top + 40, 'text': 'I(+)',
                     'corrected': 'I(+)', 'note': None, 'cell_id': k})
    return dark, pd.DataFrame(rows)


def test_画数が読みより多ければ直す():
    dark, df = _roman_page([1] * 10 + [2, 3])
    got = ocr.fix_roman_heads(df, dark)
    assert list(got['corrected'][:10]) == ['I(+)'] * 10
    assert got.at[10, 'corrected'] == 'II(+)'
    assert got.at[11, 'corrected'] == 'III(+)'
    assert 'roman' in got.at[10, 'note']


def test_目安が少ない表では直さない():
    dark, df = _roman_page([1, 1, 2])
    got = ocr.fix_roman_heads(df, dark)
    assert list(got['corrected']) == ['I(+)'] * 3


def test_画数の数え方が合わない表では直さない():
    """読みと画数が合う割合が低い表は，数え方がその表に合っていない"""
    dark, df = _roman_page([2] * 12)
    got = ocr.fix_roman_heads(df, dark)
    assert list(got['corrected']) == ['I(+)'] * 12


def test_組成部が無ければ触らない():
    df = pd.DataFrame([{'obj_name': 'row', 'text': 'x'}])
    assert ocr.fix_roman_heads(df, np.zeros((10, 10), bool)).equals(df)


# --- 画像を文字列で持つ ----------------------------------------------------

def test_画像をデータURLにする():
    got = ocr.jpg2base64(_cell('1'))
    assert got.startswith('data:image/jpeg;base64,')
    assert len(got) > 100
