"""NDLOCR-Lite を読み手として使う (ndl.py)

外部プロセスを呼ぶ部分は実データ (slow) で，JSON の読み取りと座標の戻しは
組み立てた値で確かめる．
"""
import os

import pytest

from comptea import ndl

DOC = {'contents': [[
    {'boundingBox': [[10, 20], [10, 40], [60, 20], [60, 40]],
     'text': '通し番号', 'confidence': 0.92},
    {'boundingBox': [[10, 50], [10, 70], [70, 50], [70, 70]],
     'text': '調査番号', 'confidence': 0.86},
    {'boundingBox': [[10, 80], [10, 90], [20, 80], [20, 90]],
     'text': '   ', 'confidence': 0.1},          # 空は捨てる
    {'boundingBox': [[1, 1]], 'text': 'x'},      # 箱が足りないものも捨てる
]]}


def test_JSON_から箱と文字列を取る():
    got = ndl.parse(DOC)
    assert [t for _b, t, _c in got] == ['通し番号', '調査番号']
    assert got[0][0] == (10, 20, 60, 40)
    assert got[0][2] == pytest.approx(0.92)


def test_中身が無ければ空():
    assert ndl.parse({}) == []
    assert ndl.parse(None) == []


def test_置き場が無ければ読まない(monkeypatch):
    """入っていない環境では黙って空を返す (読みは補助)"""
    monkeypatch.setattr(ndl, 'find_dir', lambda: None)
    r = ndl.NdlReader(ndl_dir=None)
    assert r.available() is False
    assert r.read_boxes(None, (0, 0, 10, 10)) == []


def test_環境変数で置き場を指せる(tmp_path, monkeypatch):
    d = tmp_path / 'src'
    d.mkdir()
    (d / 'ocr.py').write_text('', encoding='utf-8')
    monkeypatch.setenv(ndl.ENV, str(tmp_path))
    assert ndl.find_dir() == str(d)


def test_読みの座標は元の画像に戻す(monkeypatch, tmp_path):
    """切り出して読むので，箱の左上を足して返す"""
    PIL = pytest.importorskip('PIL')
    from PIL import Image

    calls = {}

    def fake_run(cmd, **kw):
        dst = cmd[cmd.index('--output') + 1]
        import json
        with open(os.path.join(dst, 'a.json'), 'w', encoding='utf-8') as f:
            json.dump(DOC, f)
        calls['n'] = calls.get('n', 0) + 1

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(ndl.subprocess, 'run', fake_run)
    r = ndl.NdlReader(ndl_dir=str(tmp_path))
    got = r.read_boxes(Image.new('RGB', (400, 400), 'white'), (100, 200, 300, 380))
    assert calls['n'] == 1
    assert got[0] == ((110, 220, 160, 240), '通し番号')
