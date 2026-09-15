"""表頭の値の補正 (correct_text.py)

2026-09-15 に足した．方位・調査年月日・調査番号・和名から学名を引く段が，
的で呼ばれていなかった．**表頭の値は地点の素性**なので，黙って化けると
あとから気づけない．
"""
import pytest

from comptea import correct_text


# --- 方位 ------------------------------------------------------------------

def test_方位の誤読を直す():
    """'N巳' → 'NE' (漢字に読まれた字を戻す)"""
    assert correct_text.correct_aspect('N巳')['corrected'] == 'NE'


def test_正しい方位はそのまま():
    for t in ('N', 'SW', 'NNE'):
        got = correct_text.correct_aspect(t)
        assert got['corrected'] == t
        assert got['status'] == 'OK'


def test_方位で無ければ要確認():
    got = correct_text.correct_aspect('あいう')
    assert got['status'] != 'OK'


# --- 調査年月日 ------------------------------------------------------------

def test_年月日を組み立てる():
    """'83 5 21' → '1983-05-21' (2 桁の年は 19xx)"""
    assert correct_text.correct_date('83 5 21')['corrected'] == '1983-05-21'


def test_区切りが違っても読む():
    for t in ("'83/5/21", '83.5.21', '83 5 21'):
        got = correct_text.correct_date(t)
        assert got['corrected'] == '1983-05-21', t


def test_日付で無ければ要確認():
    assert correct_text.correct_date('ほげ')['status'] != 'OK'


# --- 調査番号 --------------------------------------------------------------

def test_調査番号の定型に直す():
    """英字 2 文字 + 数字 (SO-216・KF 360・MS-82)"""
    for t in ('SO-216', 'SO 216', 'SO216'):
        got = correct_text.correct_field_no(t)
        assert got['status'] == 'OK', t
        assert '216' in got['corrected']


def test_調査番号で無ければ要確認():
    assert correct_text.correct_field_no('ののの')['status'] != 'OK'


# --- 和名から学名 ----------------------------------------------------------

def test_一つに決まる和名だけ引く():
    """**1 つに決められるときだけ**返す (属や種が分かれる和名は返さない)"""
    name, why = correct_text.sname_from_jname('アカマツ')
    assert why == 'ok' and name


def test_辞書に無い和名は引かない():
    name, why = correct_text.sname_from_jname('ズイナ群集')
    assert why != 'ok'


def test_空なら引かない():
    for t in ('', None):
        _name, why = correct_text.sname_from_jname(t)
        assert why != 'ok'
