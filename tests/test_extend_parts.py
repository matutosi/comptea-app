"""行の上下端を伸ばす段 (row_heights.py の `extend_edges` から切り出した 3 段)

2026-09-14 に 178 行の `extend_edges` を分けたときに足した．
**判定 (`is_row`・`is_flow`・`judge`) を引数で受け取る形**になったので，
合成データだけで筋道を確かめられる．

ここは**欠落に直結する**段で，ユーザの指示は
「学名・和名・組成の欠落は絶対に避ける．多めに取ってから読んで除外する」．
"""
import numpy as np
import pytest

from comptea import row_heights as rh


MED = 30.0
EDGES = [100.0, 130.0, 160.0, 190.0]


def _yes(a, b):
    return True


def _no(a, b):
    return False


# --- 末尾を落とす ----------------------------------------------------------

def test_字のある行は落とさない():
    out, trimmed = rh._trim_tail(list(EDGES), MED, {}, _yes, _no, None)
    assert out == EDGES and trimmed == 0


def test_字の無い末尾は落とす():
    out, trimmed = rh._trim_tail(list(EDGES), MED, {}, _no, _no, None)
    assert trimmed == 2 and out == EDGES[:2]      # 2 本残るまで落とす


def test_流し込みの末尾も落とす():
    out, trimmed = rh._trim_tail(list(EDGES), MED, {}, _yes, _yes, None)
    assert trimmed == 2 and out == EDGES[:2]


def test_短すぎる末尾は無条件に落とす():
    """並べ直しが残す 0.5 行ぶんの余り (中央値の 3/4 未満)"""
    edges = EDGES + [EDGES[-1] + MED * 0.4]
    out, trimmed = rh._trim_tail(edges, MED, {}, _yes, _no, None)
    assert trimmed == 1 and out == EDGES


def test_読んで種の行なら戻す():
    """**形で字が無く見えても**，読みが species なら落とさない (16_p2 の型)

    種名の側に字の塊があること (`unit_centred`) も要る．
    """
    prof = {'sname': np.zeros(260)}
    # **帯ごとに塊が要る** (上から 1 帯ずつ戻し，違うものが出たら止めるため)
    prof['sname'][135:155] = 1.0                  # 帯 (130, 160) の中
    prof['sname'][165:185] = 1.0                  # 帯 (160, 190) の中
    out, trimmed = rh._trim_tail(list(EDGES), MED, prof, _no, _no,
                                 lambda a, b: 'species')
    assert out == EDGES                           # 2 本とも戻った
    assert trimmed == 0


def test_戻すのは続いているあいだだけ():
    """最初に違うものが出たら止める (見出しの下に並ぶ種名を拾わない: 13_p1)"""
    prof = {'sname': np.zeros(260)}
    prof['sname'][165:185] = 1.0                  # 下の帯にだけ塊
    out, trimmed = rh._trim_tail(list(EDGES), MED, prof, _no, _no,
                                 lambda a, b: 'species')
    assert out == EDGES[:2] and trimmed == 2


def test_読みが種でなければ戻さない():
    prof = {'sname': np.ones(260)}
    out, trimmed = rh._trim_tail(list(EDGES), MED, prof, _no, _no,
                                 lambda a, b: 'heading')
    assert out == EDGES[:2] and trimmed == 2


# --- 下へ足す --------------------------------------------------------------

def test_下端まで行の高さで足す():
    out, below = rh._add_below([100.0, 130.0], MED, 220.0, _yes, _no, None)
    assert below == 3
    assert out == [100.0, 130.0, 160.0, 190.0, 220.0]


def test_流し込みに入ったら止める():
    out, below = rh._add_below([100.0, 130.0], MED, 220.0, _yes, _yes, None)
    assert below == 0 and out == [100.0, 130.0]


def test_字が無ければ止める():
    out, below = rh._add_below([100.0, 130.0], MED, 220.0, _no, _no, None)
    assert below == 0


def test_読み手が無ければ目安を越えない():
    """`judge` が無いときは `hi` までしか足さない (除外の手段が無い)"""
    out, below = rh._add_below([100.0, 130.0], MED, 160.0, _yes, _no, None)
    assert out[-1] <= 160.0 + 1


def test_読み手があれば目安を越えて試す():
    """越えたぶんは**読んで種の行と分かったものだけ**足す (13_p3 の型)"""
    out, _b = rh._add_below([100.0, 130.0], MED, 160.0, _yes, _no,
                            lambda a, b: 'species')
    assert out[-1] > 160.0 + 1
    out2, _b2 = rh._add_below([100.0, 130.0], MED, 160.0, _yes, _no,
                              lambda a, b: 'flow')
    assert out2[-1] <= 160.0 + 1


# --- 上へ足す --------------------------------------------------------------

def test_上端まで足す():
    prof = np.ones(300)
    out, above = rh._add_above([100.0, 130.0, 160.0], MED, 40.0, prof)
    assert above == 2 and out[0] == 40.0


def test_字が無ければ上へは足さない():
    prof = np.ones(300)
    prof[:100] = 0.0
    out, above = rh._add_above([100.0, 130.0, 160.0], MED, 40.0, prof)
    assert above == 0 and out[0] == 100.0


def test_画像の外へは出ない():
    prof = np.ones(300)
    out, _a = rh._add_above([20.0, 50.0, 80.0], MED, -100.0, prof)
    assert out[0] >= 0.0
