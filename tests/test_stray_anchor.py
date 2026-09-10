"""段の目印のうち，本物の段と縦に重ならないものを捨てる (blocks.drop_stray_anchors)

表の下の注記が `sname` として検出されると段が 1 つ増え，右端の列を足す限界になる
(s01115_04_p2 の y 6447〜6596 の 149 px．指摘 13・35)．
"""
import pandas as pd

from comptea import blocks


def _det(rows):
    df = pd.DataFrame(rows, columns=['obj_name', 'x1', 'y1', 'x2', 'y2'])
    df['source_image'] = 'x.jpg'
    return df


def test_縦に重ならない目印は捨てる():
    df = _det([('sname', 100, 1500, 880, 3500),
               ('sname', 1900, 6400, 2270, 6550),        # 表の下の注記
               ('row', 100, 1500, 2200, 1540)])
    out, n = blocks.drop_stray_anchors(df)
    assert n == 1
    assert list(out.obj_name) == ['sname', 'row']
    assert len(blocks.split_blocks(out)) == 1


def test_折り返した右の段は残す():
    # 右の段は左の段と同じ高さから始まる (下端は短くてよい)
    df = _det([('sname', 100, 1500, 880, 3500),
               ('sname', 1900, 1500, 2270, 2200)])
    out, n = blocks.drop_stray_anchors(df)
    assert n == 0 and len(blocks.split_blocks(out)) == 2


def test_同じ列が縦に分かれた検出は1つの段にまとめて見る():
    # 1 列の種名が 2 つの箱になっている (01_p3・12_p1)．下の箱は上の箱と重ならないが，
    # x が重なるので同じ段．捨ててはいけない
    df = _det([('sname', 36, 1000, 811, 4300),
               ('sname', 40, 7700, 800, 9650),
               ('sname', 1900, 1000, 2270, 3000)])       # 本物の右の段
    out, n = blocks.drop_stray_anchors(df)
    assert n == 0


def test_目印が1つの段だけなら何もしない():
    df = _det([('sname', 36, 1000, 811, 4300), ('sname', 40, 7700, 800, 9650)])
    out, n = blocks.drop_stray_anchors(df)
    assert n == 0 and len(out) == 2
