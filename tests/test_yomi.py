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


# --- 段落で読む (レイアウト解析) -------------------------------------------
#
# 表の下の注記は**文章**なので，語ではなく段落で受け取る．
# yomitoku の `DocumentAnalyzer` が段落を返す．

PARAS = [[[10, 900, 800, 960], '調査地 Lage: Lfd. Nr.1: 神戸市山田町', 0.9],
         [[10, 970, 800, 1020], '', 0.1]]        # 空は捨てる


def test_段落を読む(monkeypatch, tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    def fake_run(cmd, **kw):
        with open(cmd[3], 'w', encoding='utf-8') as f:
            json.dump(PARAS, f)

        class R:
            returncode = 0
        return R()

    p = tmp_path / 'python.exe'
    p.write_text('', encoding='utf-8')
    monkeypatch.setattr(yomi.subprocess, 'run', fake_run)
    r = yomi.YomiReader(python=str(p))
    got = r.read_paragraphs(Image.new('RGB', (900, 1100), 'white'))
    assert len(got) == 1
    assert got[0]['text'].startswith('調査地')
    assert got[0]['box'] == (10, 900, 800, 960)


def test_段落も切り出しの座標を元に戻す(monkeypatch, tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    def fake_run(cmd, **kw):
        with open(cmd[3], 'w', encoding='utf-8') as f:
            json.dump([[[0, 0, 100, 50], '調査地 神戸市', 0.9]], f)

        class R:
            returncode = 0
        return R()

    p = tmp_path / 'python.exe'
    p.write_text('', encoding='utf-8')
    monkeypatch.setattr(yomi.subprocess, 'run', fake_run)
    r = yomi.YomiReader(python=str(p))
    got = r.read_paragraphs(Image.new('RGB', (400, 400), 'white'),
                            box=(50, 300, 350, 400))
    assert got[0]['box'] == (50, 300, 150, 350)


def test_入っていなければ段落も読まない(monkeypatch):
    monkeypatch.delenv(yomi.ENV_PY, raising=False)
    assert yomi.YomiReader().read_paragraphs(None) == []


# --- 装置 (CPU / GPU) ------------------------------------------------------

def test_装置は環境に合わせる(monkeypatch):
    """**GPU があるとは限らない**．既定を cuda に固定しない"""
    from comptea import device

    device.forget()
    monkeypatch.setenv(device.ENV, 'cpu')
    assert yomi.YomiReader(python='x').device == 'cpu'
    monkeypatch.setenv(device.ENV, 'cuda')
    assert yomi.YomiReader(python='x').device == 'cuda'
    device.forget()


def test_指した装置が優先(monkeypatch):
    from comptea import device

    monkeypatch.setenv(device.ENV, 'cuda')
    assert yomi.YomiReader(python='x', device='cpu').device == 'cpu'
