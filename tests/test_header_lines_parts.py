"""表頭の帯を作る小さな段 (header_lines.py)

2026-09-15 に足した．**傾きに沿って数える**投影と，境の置き直しを固める．
紙面が 0.5 度傾くだけで，幅 2,000 px では行が 19 px ずれ，真横に数えた投影では
行と行の谷が埋まる (2026-09-10 に分かったこと)．
"""
import numpy as np

from comptea import header_lines


def _slanted(h=400, w=600, pitch=40, slope=0.01, ink=8):
    """傾いた行を置いた紙面 (True が黒)．`slope` は x あたりの y のずれ"""
    d = np.zeros((h, w), dtype=bool)
    for y in range(20, h - pitch, pitch):
        for x in range(w):
            yy = int(y + slope * x)
            if 0 <= yy < h - ink:
                d[yy:yy + ink, x] = True
    return d


def test_傾きに沿って数えると山が鋭くなる():
    dark = _slanted(slope=0.02)
    flat = header_lines.slanted_profile(dark, 0, 600, 0.0)
    tilt = header_lines.slanted_profile(dark, 0, 600, 0.02)
    assert tilt.max() > flat.max()              # 同じ行がそろって数えられる


def test_勾配を選ぶ():
    dark = _slanted(slope=0.02)
    prof, slope = header_lines.slant_profile(dark, 40.0)
    assert len(prof) == dark.shape[0]
    assert abs(slope - 0.02) < 0.02             # おおよそ当たる


def test_傾きが無ければ勾配も0に近い():
    dark = _slanted(slope=0.0)
    _prof, slope = header_lines.slant_profile(dark, 40.0)
    assert abs(slope) < 0.01


# --- 値の行の内側に境を置かない --------------------------------------------

def test_行の内側の境を外へ出す():
    """kinki_021: 値の行 885-912 の真ん中 (897) に境が入っていた"""
    # `runs` は (中心, 上, 下) の 3 つ組
    got = header_lines.keep_out_of_runs([100.0, 897.0, 1000.0],
                                        [(898.5, 885.0, 912.0)])
    assert not any(885 < y < 912 for y in got)
    assert len(got) == 3                        # 数は変えない


def test_行の外の境は動かさない():
    got = header_lines.keep_out_of_runs([100.0, 940.0, 950.0],
                                        [(898.5, 885.0, 912.0)])
    assert list(got) == [100.0, 940.0, 950.0]


def test_行が無ければ何もしない():
    got = header_lines.keep_out_of_runs([100.0, 200.0], [])
    assert list(got) == [100.0, 200.0]
