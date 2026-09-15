"""読み直しを，変わらなくなるまで繰り返す (pipeline/read.py)

2026-09-15 に足した．保存済みの結果に**同じ読み直しをもう一度**当てると，
17_p1 でだけ 43 セル読めた (要確認 367 → 324)．画像・読み手・余白・束の
大きさ・装置・補正はすべて同じで，**なぜ 1 回で取りこぼしたのかは未解明**．
原因が分からなくても，**変化が無くなるまで回せば取りこぼさない**．
"""
import pandas as pd
import pytest

from comptea.pipeline import read as read_mod


class _Rounds:
    """**セルごとに**決まった読みを返す偽の読み手

    セルは `x1` で見分ける (`read_crops` は箱しか受け取らないため)．
    `rounds` は巡ごとの {x1: 読み}．巡が尽きたら最後のものを繰り返す．
    """

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = 0

    def available(self):
        return True

    def read_crops(self, img, boxes, pad=0):
        i = min(self.calls, len(self.rounds) - 1)
        self.calls += 1
        table = self.rounds[i]
        return [table.get(int(b[0]), '') for b in boxes]


def _df(n):
    """x1 が 0, 10, 20 … のセル (読み手が見分けるため)"""
    return pd.DataFrame(
        [{'obj_name': 'comp', 'text': 'z-u', 'corrected': '',
          'status': 'Need Check', 'note': '',
          'x1': float(i * 10), 'y1': 0.0, 'x2': float(i * 10 + 8), 'y2': 10.0}
         for i in range(n)])


def _left(df):
    return int(((df['obj_name'] == 'comp')
                & (df['status'] == 'Need Check')).sum())


def test_2巡目で読めたぶんも拾う():
    """1 巡目は 1 つ，2 巡目で残りが読める"""
    r = _Rounds([{0: '2・2'}, {10: '1・1', 20: '1・2'}])
    got, rounds = read_mod.retry_until_stable(None, _df(3), r)
    assert _left(got) == 0
    assert rounds >= 2


def test_変わらなければ止まる():
    """**同じ結果が返るなら 2 巡で止める** (無駄に回さない)"""
    r = _Rounds([{0: '2・2'}])                 # いつ読んでも同じ
    got, rounds = read_mod.retry_until_stable(None, _df(3), r)
    assert _left(got) == 2
    assert rounds == 2                         # 1 巡目で直り，2 巡目で変化なし


def test_上限で止まる():
    """毎巡 1 つずつ読めても，上限を超えては回さない"""
    r = _Rounds([{0: '2・2'}, {10: '1・1'}, {20: '1・2'}, {30: '2・1'}])
    got, rounds = read_mod.retry_until_stable(None, _df(4), r, rounds=2)
    assert rounds == 2
    assert _left(got) == 2


def test_要確認が無ければ読まない():
    df = _df(2)
    df['status'] = 'OK'
    r = _Rounds([{0: '2・2', 10: '2・2'}])
    got, rounds = read_mod.retry_until_stable(None, df, r)
    assert rounds == 0
    assert r.calls == 0                        # 読み手を呼ばない


def test_読み手が無ければ何もしない():
    got, rounds = read_mod.retry_until_stable(None, _df(2), None)
    assert _left(got) == 2
    assert rounds == 1                         # 1 巡目で変化なし → 止まる
