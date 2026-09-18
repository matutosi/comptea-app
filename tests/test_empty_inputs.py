"""空・欠けた入力で落ちない，値を失わない (2026-09-18 の点検で見つけた型)

どれも「ふだんの見本では起きず，全表を通すと一部の表で止まる・消える」壊れ方．
"""
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import conftest
from comptea import comp_table, note, ocr


# --- note ---------------------------------------------------------------

@pytest.mark.parametrize('old', [None, float('nan'), '', ';'])
def test_空のnoteに足してもnanにならない(old):
    assert note.add(old, 'ndl') == 'ndl'


def test_noteは区切りを1つにして足す():
    assert note.add('snapped', 'retry') == 'snapped;retry'
    assert note.add('snapped;', 'retry') == 'snapped;retry'
    assert note.add('retry', 'retry') == 'retry'          # 同じ印は重ねない


def test_CSVから読んだ空のnoteに足す(tmp_path):
    """CSV の空欄は NaN で戻る．`str(v or '')` はこれを 'nan' にしていた"""
    p = tmp_path / 'x.csv'
    pd.DataFrame({'note': ['', 'snapped']}).to_csv(p, index=False)
    d = pd.read_csv(p)
    assert [note.add(v, 'ndl') for v in d['note']] == ['ndl', 'snapped;ndl']


# --- 縦持ち・横持ち -----------------------------------------------------

def _long(layer):
    return pd.DataFrame({
        'row_no': [1, 1, 2], 'j_name': ['ヤマブドウ', 'ヤマブドウ', None],
        's_name': [None, None, 'Vitis'], 'layer': [layer] * 3,
        'plot': [1, 2, 1], 'cover': ['1', '+', '2'], 'sociability': ['1', '', '2']})


@pytest.mark.parametrize('layer', [None, 'K'])
def test_横持ちは鍵が空の行も落とさない(layer):
    """pivot_table は鍵が NaN の行を捨てる (out/v4 で 0 行の表があった)"""
    wide = comp_table.to_wide(_long(layer))
    assert len(wide) == 2
    assert list(wide[1]) == ['1', '2・2']


def test_組成が無い表でも縦持ちの列はそろう():
    df = pd.DataFrame({'obj_name': ['sname'], 'col': [1], 'row': [1],
                       'text': ['Vitis'], 'corrected': ['Vitis']})
    res = comp_table.comp_table(df)
    assert len(res) == 0
    for col in ('note', 'source', 'constancy'):
        assert col in res.columns


# --- 画像 ---------------------------------------------------------------

@pytest.mark.parametrize('mode', ['RGBA', 'P', 'LA', 'I;16', 'L', 'RGB'])
def test_どの色の形でもbase64にできる(mode):
    img = Image.new('RGB', (8, 8), 'white').convert(mode)
    assert ocr.jpg2base64(img).startswith('data:image/jpeg;base64,')


# --- cli -----------------------------------------------------------------

def _load(name):
    src = pathlib.Path(conftest.ROOT) / 'cli' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'{name}_under_test', src)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_目視に回すセルが0件でも切り出しは落ちない(tmp_path):
    """旧版の段階2は 0 件のとき改行だけの review.tsv を書いていた"""
    crop = _load('crop_cells')
    df = pd.DataFrame({'cell_id': [1], 'obj_name': ['comp']})
    args = SimpleNamespace(ids=None, cls=None, what='review')
    for content in ('\r\n', 'cell_id\tobj_name\treason\n'):
        (tmp_path / 'review.tsv').write_text(content, encoding='utf-8')
        assert crop.select(df, args, tmp_path).empty


def test_注記だけのページでも1回出現種の段は落ちない(tmp_path, monkeypatch):
    once = _load('read_once_page')
    Image.fromarray(np.full((20, 20), 255, np.uint8)).save(tmp_path / 'p.png')
    monkeypatch.setattr(once.once_page, 'find_block', lambda image: {
        'box': None, 'span': None, 'lines': [], 'has_mark': False,
        'note': '調査地: 奈良県'})
    found = once.crop_block(str(tmp_path / 'p.png'), str(tmp_path))
    assert found['note']
    assert (tmp_path / 'note.txt').is_file()
    assert not (tmp_path / 'once_block.png').exists()
