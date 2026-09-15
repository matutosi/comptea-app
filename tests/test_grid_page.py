"""紙面ごとの下ごしらえ (pipeline/grid.py の回転・傾き直し・続きのページ)

2026-09-15 に足した．検出器は重いので `grid.detect` を差し替える．
ここで固めるのは**戻す決まり**である．傾きを直すと検出が減る紙面があり
(04_p2 は 122 行 → 34 行)，直した利益より大きい (2026-09-04)．
"""
import types

import pandas as pd
import pytest
from PIL import Image

from comptea.pipeline import grid


def _page(tmp_path, size=(400, 200), name='kinki_047.png'):
    p = tmp_path / name
    Image.new('RGB', size, 'white').save(p)
    return str(p)


def _det(n_col=5, n_mark=3, n_row=5):
    rows = []
    for i in range(n_col):
        rows.append({'obj_name': 'col', 'x1': 10 + 20 * i, 'y1': 50,
                     'x2': 28 + 20 * i, 'y2': 180, 'confidence': 0.9})
    for i in range(n_mark):
        rows.append({'obj_name': 'sname', 'x1': 0, 'y1': 50 + 10 * i,
                     'x2': 9, 'y2': 58 + 10 * i, 'confidence': 0.9})
    for i in range(n_row):
        rows.append({'obj_name': 'row', 'x1': 0, 'y1': 50 + 10 * i,
                     'x2': 200, 'y2': 58 + 10 * i, 'confidence': 0.9})
    return pd.DataFrame(rows)


def _args(tmp_path, image):
    return types.SimpleNamespace(image=image, workdir=str(tmp_path / 'work'),
                                 weights='none.pt', conf=25, imgsz=1280,
                                 no_resplit=False)


# --- 横倒しの紙面を回す ----------------------------------------------------

def test_時計回りに回して隣に置く(tmp_path):
    image = _page(tmp_path, size=(400, 200))
    out = grid.rotate_page(image, tmp_path / 'work' / 'kinki_047')
    assert Image.open(out).size == (200, 400)       # 縦横が入れ替わる
    assert out.endswith('_rot.png')


def test_反時計回りにも回せる(tmp_path):
    """kinki_014 は反時計回り"""
    image = _page(tmp_path, size=(400, 200))
    a = grid.rotate_page(image, tmp_path / 'w' / 'a', how='ROTATE_270')
    b = grid.rotate_page(image, tmp_path / 'w' / 'b', how='ROTATE_90', tag='rot2')
    assert b.endswith('_rot2.png')
    assert Image.open(a).transpose(Image.ROTATE_180) != Image.open(b)


# --- 傾きを直す ------------------------------------------------------------

def test_傾きが無ければ触らない(tmp_path, monkeypatch):
    image = _page(tmp_path)
    df = _det()
    monkeypatch.setattr('comptea.deskew.deskew_to', lambda *a: (image, 0.0))
    got_img, got_df, warn = grid.deskew_page(image, df, _args(tmp_path, image), {})
    assert got_img == image and warn == []
    assert len(got_df) == len(df)


def test_検出が少ない紙面では測らない(tmp_path, monkeypatch):
    """`col` も行の高さの元も 3 本に満たなければ，角度を出せない"""
    image = _page(tmp_path)
    monkeypatch.setattr('comptea.deskew.deskew_to',
                        lambda *a: pytest.fail('測ってはいけない'))
    got_img, _df, warn = grid.deskew_page(image, _det(n_col=2, n_row=2),
                                          _args(tmp_path, image), {})
    assert got_img == image and warn == []


def test_直して検出が保たれれば採る(tmp_path, monkeypatch):
    image = _page(tmp_path)
    fixed = _page(tmp_path, name='kinki_047_deskew.png')
    monkeypatch.setattr('comptea.deskew.deskew_to',
                        lambda img, out, *a: (fixed, -0.83))
    monkeypatch.setattr(grid, 'detect', lambda *a, **k: _det())
    got_img, _df, warn = grid.deskew_page(image, _det(), _args(tmp_path, image), {})
    assert got_img == fixed
    assert warn and '傾き' in warn[0]


def test_直して検出が減れば元に戻す(tmp_path, monkeypatch):
    """04_p2 は回すと 122 行 → 34 行になった．回転画像も残さない"""
    image = _page(tmp_path)
    saved = []

    def fake_deskew(img, out, *a):
        Image.new('RGB', (400, 200), 'white').save(out)   # 本物も同じ所に書く
        saved.append(out)
        return str(out), -0.83

    monkeypatch.setattr('comptea.deskew.deskew_to', fake_deskew)
    monkeypatch.setattr(grid, 'detect',
                        lambda *a, **k: _det(n_col=1, n_mark=0, n_row=1))
    got_img, got_df, warn = grid.deskew_page(image, _det(),
                                             _args(tmp_path, image), {})
    assert got_img == image
    assert len(got_df) == len(_det())              # 元の検出のまま
    assert warn and '直さなかった' in warn[0]
    assert not saved[0].is_file()                  # 使わない回転画像は消す


# --- 続きのページ ----------------------------------------------------------

def test_枝番が無ければ続きとして扱わない(tmp_path):
    """枝番の無いページで表が無いのは，これまでどおり異常"""
    image = _page(tmp_path)
    assert grid.save_continuation(image, tmp_path / 'work' / 'kinki_047') is False


def test_枝番があれば続きとして書き出す(tmp_path, monkeypatch):
    image = _page(tmp_path)
    monkeypatch.setattr('comptea.once_page.find_block', lambda img: None)
    base = tmp_path / 'work' / 'kinki_047-2'
    assert grid.save_continuation(image, base) is True
    got = (base / 'continuation.txt').read_text(encoding='utf-8')
    assert 'table\tkinki_047' in got
    assert 'part\t2' in got


def test_枝番は画像の名前で決める(tmp_path, monkeypatch):
    """GUI の置き場は一時ディレクトリなので，置き場の名前には枝番が付かない"""
    image = _page(tmp_path)
    monkeypatch.setattr('comptea.once_page.find_block', lambda img: None)
    base = tmp_path / 'out'
    assert grid.save_continuation(image, base, src_name='kinki_047-3') is True
    assert 'part\t3' in (base / 'continuation.txt').read_text(encoding='utf-8')


def test_流し込みの塊と注記を切り出す(tmp_path, monkeypatch):
    image = _page(tmp_path)
    monkeypatch.setattr('comptea.once_page.find_block', lambda img: {
        'box': (10, 10, 100, 60), 'note_box': (10, 70, 200, 120),
        'note': 'この表は前ページから続く', 'has_mark': True})
    base = tmp_path / 'work' / 'kinki_047-2'
    assert grid.save_continuation(image, base) is True
    assert (base / 'once_block.png').is_file()
    assert (base / 'note_block.png').is_file()
    assert (base / 'note.txt').read_text(encoding='utf-8').startswith('この表は')
