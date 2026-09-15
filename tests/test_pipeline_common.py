"""段の入口の共通部品 (pipeline/common.py)

2026-09-15 に足した．公開 6 つのうち 5 つが的で呼ばれていなかった．

**警告の選り分けはここが要**．1 枚あたり警告は中央値 5 件あり，全部を並べると
本当に見るべきものが埋もれる (2026-09-01 に全 88 枚を通して決めた)．
"""
import os
from pathlib import Path

import pytest

from comptea.pipeline import common


# --- 置き場 ----------------------------------------------------------------

def test_中核の場所を返す():
    d = common.package_dir()
    assert os.path.isdir(d)
    assert os.path.isfile(os.path.join(d, '__init__.py'))


def test_読めるようにする():
    """`setup` は渡した場所を**絶対パスに直して**返す

    工程は途中で作業ディレクトリを移ることがあるので，相対のままだと
    あとで開けなくなる．併せて中核を import できる状態にする．
    """
    got = list(common.setup(['a.png']))
    assert len(got) == 1
    assert os.path.isabs(got[0])
    assert got[0].endswith('a.png')
    import comptea                                   # noqa: F401


def test_中間物の置き場を決める():
    """指定が無ければ画像の名前から決める"""
    got = common.workdir('/tmp/s01115_01_p1.png', None, make=False)
    assert 's01115_01_p1' in str(got)


def test_置き場を指せる():
    got = common.workdir('/tmp/a.png', '/tmp/work/here', make=False)
    assert str(got).replace('\\', '/').endswith('work/here')


# --- 前の工程の出力が無いとき ----------------------------------------------

def test_無ければ何を実行するか言って止まる():
    with pytest.raises(SystemExit) as e:
        common.need_file(Path('/no/such/file.csv'), 'run_pipeline.py')
    assert 'run_pipeline.py' in str(e.value)


def test_あれば止まらない(tmp_path):
    f = tmp_path / 'located.csv'
    f.write_text('x', encoding='utf-8')
    common.need_file(f, 'run_pipeline.py')           # 例外は出ない


# --- 警告の選り分け --------------------------------------------------------

def test_要確認と参考に分ける():
    must, note = common.sort_warnings([
        "'row' が1件も検出されなかったため，行の位置を決められない",
        '傾きを直した (0.5 度)',
    ])
    assert len(must) == 1 and len(note) == 1
    assert '検出されなかった' in must[0]


def test_目印が無ければ参考():
    must, note = common.sort_warnings(['段1: 行を中央値の高さで並べ直した'])
    assert must == [] and len(note) == 1


def test_空でも落ちない():
    assert common.sort_warnings(None) == ([], [])
    assert common.sort_warnings([]) == ([], [])


def test_表示は要確認を先に出す(capsys):
    common.show_warnings([
        '傾きを直した',
        "'col' が1件も検出されず，組成部のセルを出力できない",
    ])
    out = capsys.readouterr().out
    assert '出力できない' in out
    assert out.index('出力できない') < out.index('傾きを直した')


def test_警告が無ければ何も出さない(capsys):
    common.show_warnings([])
    assert capsys.readouterr().out.strip() == ''
