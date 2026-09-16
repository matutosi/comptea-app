"""結果に版と設定を書き残す (pipeline/common.py の run_info)

2026-09-16 に足した．保存した通しの結果が**コミットしていない作業ツリーの
状態** (読み直しの余白が広いまま) で回されていたのに気づかず，
「17_p1 が 22 件後退した」と取り違えて原因を追った．版と設定が結果に
残っていれば，比べる前に 1 行で気づける．
"""
import json
import subprocess

from comptea.pipeline import common


def test_段ごとに書き足す(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'code_version', lambda root=None: {'commit': 'abc1234', 'dirty': False})
    common.write_run_info(tmp_path, 'grid', {'imgsz': 1280})
    common.write_run_info(tmp_path, 'read', {'retry_pad': 3})
    info = json.loads((tmp_path / 'run_info.json').read_text(encoding='utf-8'))
    assert set(info) == {'grid', 'read'}
    assert info['grid']['settings'] == {'imgsz': 1280}
    assert info['read']['version']['commit'] == 'abc1234'


def test_同じ段は上書きする(tmp_path, monkeypatch):
    """段階 2 をやり直せば，段階 2 の記録だけが新しくなる"""
    monkeypatch.setattr(common, 'code_version', lambda root=None: None)
    common.write_run_info(tmp_path, 'read', {'retry_pad': 10})
    common.write_run_info(tmp_path, 'read', {'retry_pad': 3})
    info = json.loads((tmp_path / 'run_info.json').read_text(encoding='utf-8'))
    assert info['read']['settings'] == {'retry_pad': 3}


def test_未コミットの変更に印を付ける(tmp_path, monkeypatch):
    """今日の取り違えを防ぐための肝心の 1 文字"""
    monkeypatch.setattr(common, 'code_version',
                        lambda root=None: {'commit': 'abc1234', 'dirty': True})
    common.write_run_info(tmp_path, 'read', {'retry_pad': 10})
    line = common.run_info_line(tmp_path)
    assert 'read abc1234*' in line
    assert '未コミットの変更あり' in line
    assert 'retry_pad=10' in line


def test_変更が無ければ印を付けない(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'code_version',
                        lambda root=None: {'commit': 'abc1234', 'dirty': False})
    common.write_run_info(tmp_path, 'grid', {})
    line = common.run_info_line(tmp_path)
    assert 'grid abc1234' in line and '*' not in line


def test_記録が無ければ空(tmp_path):
    assert common.run_info_line(tmp_path) == ''


def test_壊れた記録でも落ちない(tmp_path):
    (tmp_path / 'run_info.json').write_text('{壊れている', encoding='utf-8')
    assert common.run_info_line(tmp_path) == ''
    common.write_run_info(tmp_path, 'table', {})       # 上書きして直す
    assert 'table' in json.loads((tmp_path / 'run_info.json').read_text(encoding='utf-8'))


def test_gitが無い環境でも落ちない(tmp_path, monkeypatch):
    """配布先やアプリでは git が無いことがある"""
    def boom(*a, **k):
        raise FileNotFoundError('git')
    monkeypatch.setattr(subprocess, 'run', boom)
    assert common.code_version(tmp_path) is None


def test_gitの置き場でなければNone(tmp_path):
    assert common.code_version(tmp_path) is None
