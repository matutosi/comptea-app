"""yomitoku を読み手として使う (yomi.py)

外部プロセスを呼ぶ部分は差し替えて確かめる．yomitoku 本体は要らない．
"""
import json
import os

import pytest

from comptea import yomi

GOT = [[[10, 20, 60, 40], '通し番号', 0.92],
       [[10, 50, 70, 70], '調査番号', 0.86],
       [[10, 80, 20, 90], '   ', 0.1]]          # 空は捨てる


def test_入っていなければ読まない(monkeypatch):
    monkeypatch.delenv(yomi.ENV_PY, raising=False)
    r = yomi.YomiReader()
    assert r.available() is False
    assert r.read_boxes(None, (0, 0, 10, 10)) == []


def test_環境変数で_python_を指せる(tmp_path, monkeypatch):
    p = tmp_path / 'python.exe'
    p.write_text('', encoding='utf-8')
    monkeypatch.setenv(yomi.ENV_PY, str(p))
    assert yomi.find_python() == str(p)


def test_読みの座標は元の画像に戻す(monkeypatch, tmp_path):
    PIL = pytest.importorskip('PIL')
    from PIL import Image

    calls = {}

    def fake_run(cmd, **kw):
        with open(cmd[3], 'w', encoding='utf-8') as f:
            json.dump(GOT, f)
        calls['n'] = calls.get('n', 0) + 1

        class R:
            returncode = 0
        return R()

    p = tmp_path / 'python.exe'
    p.write_text('', encoding='utf-8')
    monkeypatch.setattr(yomi.subprocess, 'run', fake_run)
    r = yomi.YomiReader(python=str(p))
    got = r.read_boxes(Image.new('RGB', (400, 400), 'white'),
                       (100, 200, 300, 380))
    assert calls['n'] == 1
    assert got == [((110, 220, 160, 240), '通し番号'),
                   ((110, 250, 170, 270), '調査番号')]


def test_読めなければ空(monkeypatch, tmp_path):
    def fake_run(cmd, **kw):
        class R:
            returncode = 1
        return R()                              # 出力を書かない

    p = tmp_path / 'python.exe'
    p.write_text('', encoding='utf-8')
    monkeypatch.setattr(yomi.subprocess, 'run', fake_run)
    PIL = pytest.importorskip('PIL')
    from PIL import Image
    r = yomi.YomiReader(python=str(p))
    assert r.read_boxes(Image.new('RGB', (50, 50), 'white'),
                        (0, 0, 50, 50)) == []
