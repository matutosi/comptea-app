"""NDLOCR-Lite を読み手として使う (ndl.py)

外部プロセスを呼ぶ部分は実データ (slow) で，JSON の読み取りと座標の戻しは
組み立てた値で確かめる．
"""
import json
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


def test_装置は環境に合わせる(monkeypatch, tmp_path):
    """NDLOCR-Lite は ONNX なので cpu でも速い (GPU の無い環境の本命)"""
    from comptea import device

    device.forget()
    monkeypatch.setenv(device.ENV, 'cpu')
    assert ndl.NdlReader(ndl_dir=str(tmp_path), python='x').device == 'cpu'
    monkeypatch.setenv(device.ENV, 'cuda')
    assert ndl.NdlReader(ndl_dir=str(tmp_path), python='x').device == 'cuda'
    device.forget()


# --- セルをまとめて読む ----------------------------------------------------
#
# 2026-09-14 の実測: **読めなかった組成のセルは NDLOCR-Lite なら 66 個中
# 44 個 (67%) 読める** (EasyOCR は 5 個，yomitoku は 0 個)．
# ただし 1 セルずつ呼ぶと **1 個 15 秒** (別プロセスの起動と模型の読み込み)．
# NDLOCR-Lite は**ディレクトリを丸ごと読む**ので，
# セルを画像として並べれば **1 回のプロセスで全部読める**．
#
# 領域をまとめて読む形 (`read_region.read_cells`) は駄目だった
# (488 セル中 53 個しか割り当たらない)．縦や格子に並べた 1 枚も駄目
# (17_p1 で 0 個)．**1 セル 1 画像**が要る．

def test_セルをまとめて読む(monkeypatch, tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    seen = {}

    def fake_run(cmd, **kw):
        src = cmd[cmd.index('--sourcedir') + 1]
        dst = cmd[cmd.index('--output') + 1]
        names = sorted(os.listdir(src))
        seen['n'] = seen.get('n', 0) + 1
        seen['files'] = len(names)
        for i, n in enumerate(names):
            doc = {'contents': [[{'boundingBox': [[0, 0], [9, 0], [9, 9], [0, 9]],
                                  'text': f'{i}', 'confidence': 0.9}]]}
            with open(os.path.join(dst, n.rsplit('.', 1)[0] + '.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(doc, f)

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(ndl.subprocess, 'run', fake_run)
    r = ndl.NdlReader(ndl_dir=str(tmp_path))
    img = Image.new('RGB', (200, 200), 'white')
    got = r.read_crops(img, [(0, 0, 20, 20), (20, 20, 40, 40), (40, 40, 60, 60)])
    assert got == ['0', '1', '2']
    assert seen['n'] == 1                  # **1 回のプロセスで読む**
    assert seen['files'] == 3


def test_読めなかったセルは空(monkeypatch, tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    def fake_run(cmd, **kw):
        class R:
            returncode = 0
        return R()                          # 何も書かない

    monkeypatch.setattr(ndl.subprocess, 'run', fake_run)
    r = ndl.NdlReader(ndl_dir=str(tmp_path))
    got = r.read_crops(Image.new('RGB', (50, 50), 'white'), [(0, 0, 10, 10)])
    assert got == ['']


def test_入っていなければ空(monkeypatch):
    monkeypatch.setenv(ndl.ENV, 'D:/no/such/place')
    monkeypatch.setattr(ndl, 'DEFAULT_DIRS', ())
    assert ndl.NdlReader().read_crops(None, [(0, 0, 1, 1)]) == ['']


def test_箱が無ければ空():
    assert ndl.NdlReader(ndl_dir='x').read_crops(None, []) == []
