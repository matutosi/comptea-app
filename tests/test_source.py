"""格子を作ったのと同じ画像を開く (comptea/source.py)

**元画像と突き合わせる間違いは何度も起きている**ので，ここで歯止めをかける．
工程は傾きを直した画像や横倒しを起こした画像で検出するので，格子の座標は
元画像とずれる (2026-09-11: 20_p3 の表頭の境が「12 本字を割る」と出たが，
正しい画像で測ると 0 本だった)．
"""
import os

import pandas as pd
import pytest
from PIL import Image

from comptea import source


def _grid(path, x2=80, y2=60):
    return pd.DataFrame({'x1': [0.0], 'y1': [0.0], 'x2': [float(x2)],
                         'y2': [float(y2)], 'obj_name': ['comp'],
                         'source_image': [path]})


def _png(path, size=(100, 80)):
    Image.new('L', size, 255).save(path)
    return str(path)


def test_記録された画像を開く(tmp_path):
    p = _png(tmp_path / 'a_deskew.png')
    im = source.open_image(_grid(p))
    assert im.size == (100, 80)


def test_置き場が移っていても同じ名前を探す(tmp_path):
    """一時ディレクトリで作った格子を，別の場所から見ることがある"""
    work = tmp_path / 'work'
    work.mkdir()
    p = _png(work / 'a_deskew.png')
    df = _grid(r'C:\もう無い場所\a_deskew.png')
    assert source.find(df, str(work)) == p
    assert source.open_image(df, str(work)).size == (100, 80)


def test_見つからなければ黙って元画像を使わない(tmp_path):
    """**代用してはいけない**．座標がずれたまま突き合わせることになる"""
    df = _grid(r'C:\もう無い場所\a_deskew.png')
    with pytest.raises(FileNotFoundError):
        source.open_image(df, str(tmp_path))


def test_収まらなければ知らせる(tmp_path):
    """格子が画像からはみ出すのは，違う画像と突き合わせている印"""
    p = _png(tmp_path / 'a.png', size=(50, 40))
    df = _grid(p, x2=80, y2=60)
    im = Image.open(p)
    assert not source.fits(df, im.size)
    warns = source.check(df, im)
    assert warns and '収まっていない' in warns[0]


def test_収まっていれば何も言わない(tmp_path):
    p = _png(tmp_path / 'a.png')
    df = _grid(p)
    assert source.fits(df, Image.open(p).size)
    assert source.check(df, Image.open(p)) == []


def test_記録が無ければNone(tmp_path):
    df = _grid(None).drop(columns=['source_image'])
    assert source.recorded(df) is None
    assert source.find(df, str(tmp_path)) is None


def test_格子と画像をまとめて読む(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    p = _png(work / 'a_deskew.png')
    _grid(p).to_csv(work / 'located.csv', index=False)
    df, im = source.load(str(work))
    assert len(df) == 1 and im.size == (100, 80)
    assert source.fits(df, im.size)


def test_元画像を渡されても記録が優先される(tmp_path):
    """`fallback` は記録が見つからないときだけ使う"""
    p = _png(tmp_path / 'a_deskew.png', size=(100, 80))
    other = _png(tmp_path / 'a.png', size=(50, 40))
    im = source.open_image(_grid(p), fallback=other)
    assert im.size == (100, 80)
