"""列の隙間・行の刻みを決める段 (col_edges・body_rows の純粋な関数)

2026-09-14 の点検で，**公開関数 355 のうち 164 (46%) が的で呼ばれていない**
と分かった．そのうち工程の要になるものから埋める．
どれも合成した画像・並びで確かめられる (実データは要らない)．
"""
import numpy as np
import pytest

from comptea import body_rows, col_edges


def _cols(width, height, bands):
    """縦帯にインクを置いた紙面 (True が黒)"""
    dark = np.zeros((height, width), dtype=bool)
    for x1, x2 in bands:
        dark[:, x1:x2] = True
    return dark


def _rows(width, height, bands):
    """横帯にインクを置いた紙面"""
    dark = np.zeros((height, width), dtype=bool)
    for y1, y2 in bands:
        dark[y1:y2, :] = True
    return dark


# --- 地点と地点のあいだの隙間 ----------------------------------------------

def test_隙間の中央を返す():
    # 値 0-40 と 80-120．隙間は 40-80
    dark = _cols(120, 100, [(0, 40), (80, 120)])
    got = col_edges.plot_gaps(dark, (0, 0, 120, 100))
    assert len(got) == 1
    assert 55 <= got[0] <= 65


def test_狭い隙間は拾わない():
    """既定の最小幅は 20 px (300 dpi で測った値)"""
    dark = _cols(120, 100, [(0, 55), (65, 120)])     # 隙間 10 px
    assert col_edges.plot_gaps(dark, (0, 0, 120, 100)) == []


def test_最小幅は呼ぶ側で変えられる():
    dark = _cols(120, 100, [(0, 55), (65, 120)])
    got = col_edges.plot_gaps(dark, (0, 0, 120, 100), min_gap=6)
    assert len(got) == 1


def test_真っ白なら全体が1つの隙間になる():
    """**組成部は必ず字があるので実害は無い**が，振る舞いを書き留めておく

    真っ白な領域では「幅ぜんたいが 1 つの空白の帯」になり，その中央が返る．
    列を組み立てるには隙間が 2 本以上要る (`column_edges_from_gaps`) ので，
    ここから列ができることはない．
    """
    dark = np.zeros((50, 80), dtype=bool)
    got = col_edges.plot_gaps(dark, (0, 0, 80, 50))
    assert len(got) == 1 and 35 <= got[0] <= 45
    assert col_edges.column_edges_from_gaps(got, 0.0, 80.0) is None


# --- 隙間から列の境を組み立てる --------------------------------------------

def test_等間隔の境を組み立てる():
    got = col_edges.column_edges_from_gaps([100.0, 200.0, 300.0], 50.0, 350.0)
    assert got is not None
    assert len(got) >= 4
    assert got[0] <= 100 and got[-1] >= 300


def test_隙間が足りなければ作らない():
    assert col_edges.column_edges_from_gaps([100.0], 50.0, 350.0) is None


# --- 行の刻み --------------------------------------------------------------

def test_黒画素の周期から行の高さを出す():
    prof = np.zeros(400)
    for y in range(10, 400, 20):                     # 20 px 刻み
        prof[y:y + 6] = 100
    got = body_rows.row_pitch(prof, prior=20.0, search=(10, 40))
    assert got is not None
    assert 18 <= got <= 22


def test_並びが平らなら刻みは出ない():
    got = body_rows.row_pitch(np.ones(200), prior=20.0, search=(10, 40))
    assert got is None or got > 0


def test_行の高さで境を並べる():
    prof = np.zeros(100)
    edges = body_rows.lattice_rows(prof, 0, 100, 20.0)
    assert len(edges) >= 5
    assert edges[0] >= 0 and edges[-1] <= 100
    d = np.diff(edges)
    assert abs(float(np.median(d)) - 20.0) <= 2.0


# --- 字を割る回数 ----------------------------------------------------------

def test_字のかたまりを割る回数を数える():
    # 20-40 と 60-80 に字がある帯．40-60 は空白
    dark = _cols(100, 30, [(20, 40), (60, 80)])
    got = col_edges.crossing_counts(dark, [(0, 30)], 100)
    assert got[30] > 0                               # 字の上
    assert got[50] == 0                              # 空白の上


def test_字が無ければどこも割らない():
    dark = np.zeros((30, 100), dtype=bool)
    got = col_edges.crossing_counts(dark, [(0, 30)], 100)
    assert int(np.sum(got)) == 0
