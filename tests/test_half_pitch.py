"""格子の刻みが印字の行の半分になっていないか (row_heights.is_half_pitch)

「・」や下線が行の中間に来ると黒画素に半分の周期が出る．切り出しの範囲が変わった
23_p2 は相関が 20 px で 0.205・40 px で 0.191 と曖昧で，自己相関だけの判定を
通らなかった (2026-09-10)．字の区間が 2 倍の刻みで 1 行 1 区間なら半分と分かる．
一方 17_p1 は階層の副行が 22 px の本物の刻み (相関 0.566) で，触ってはいけない．
"""
import numpy as np

from comptea import row_heights as rh


def _profile(n_rows, pitch, dot_w=6, text_w=14, dot_ink=20, text_ink=120, jitter=None):
    """1 行 (pitch px) に，字の帯と行の中間の「・」の帯を置いた黒画素の並び"""
    rng = np.random.default_rng(0)
    prof = np.zeros(n_rows * pitch, dtype=float)
    for i in range(n_rows):
        y = i * pitch
        j = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        prof[y + 4 + j:y + 4 + j + text_w] += text_ink
        prof[y + pitch // 2 + 4:y + pitch // 2 + 4 + dot_w] += dot_ink
    return prof


def test_字の区間が2倍の刻みで1行1区間なら半分():
    # 印字の行 40 px．字と「・」で 20 px の周期も出るが，字の区間は 40 px に 1 つ
    prof = _profile(60, 40, jitter=2)
    assert rh.is_half_pitch(prof, 0, len(prof), 20.0)
    assert not rh.is_half_pitch(prof, 0, len(prof), 40.0)


def test_相関が曖昧でも字の区間で決める():
    # 自己相関の判定を通らないようにして (ac_max=-1)，字の区間だけで半分と分かること
    prof = _profile(60, 40, jitter=2)
    assert rh.is_half_pitch(prof, 0, len(prof), 20.0, ac_max=-1.0)
    # 「・」が濃くて字の区間に数えられると 2 倍の刻みで 1 行 2 区間になり，半分とは言えない
    prof2 = _profile(60, 40, jitter=2, dot_ink=40)
    assert not rh.is_half_pitch(prof2, 0, len(prof2), 20.0, ac_max=-1.0)


def test_本物の細かい刻みは触らない():
    # 階層の副行 (17_p1): 22 px ごとに同じ字の帯が並ぶ．相関が強く，字の区間も 1 行 1 つ
    prof = np.zeros(60 * 22, dtype=float)
    for i in range(60):
        prof[i * 22 + 4:i * 22 + 16] = 100
    assert not rh.is_half_pitch(prof, 0, len(prof), 22.0)


def test_行が少なければ判定しない():
    # 20 px の刻みで 30 行に満たない並び (行の少ない表では相関が当てにならない)
    prof = _profile(12, 40)
    assert not rh.is_half_pitch(prof, 0, len(prof), 20.0)
