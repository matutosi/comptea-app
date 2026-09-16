"""読み直しの高さを列ごとに測る (ocr.empty_limits)

2026-09-16 に足した．表全体の統計で 1 つの高さを決めていたので，
**ある列の箱を変えると，触っていない列の読み直しまで動いた**
(19_p2 で地点 1 以外が 7 件，04_p1 で本物の `+` が 1 件消えた)．
"""
import pandas as pd

from comptea import ocr


def _df(cols):
    """列ごとに (読めなかったセルの黒画素の割合, 読めたセルの割合) を並べた表"""
    rows, ratios, read = [], {}, {}
    i = 0
    for c, (empties, reads) in cols.items():
        for v in empties:
            rows.append({'col': c}); ratios[i] = v; i += 1
        for v in reads:
            rows.append({'col': c}); read[i] = v; i += 1
    return pd.DataFrame(rows), ratios, read


def test_ある列を変えても他の列の高さは動かない():
    """**これが直したかったこと**"""
    base = {1: ([0.00] * 10, [0.03] * 10), 2: ([0.01] * 10, [0.04] * 10)}
    df, r, rd = _df(base)
    before = ocr.empty_limits(df, r, rd)
    # 列 1 の読めなかったセルだけ，黒画素が大きく増えた (箱を狭めた・広げた)
    changed = {1: ([0.02] * 10, [0.03] * 10), 2: base[2]}
    df2, r2, rd2 = _df(changed)
    after = ocr.empty_limits(df2, r2, rd2)
    col2_before = {before[i] for i in r if df.at[i, 'col'] == 2}
    col2_after = {after[i] for i in r2 if df2.at[i, 'col'] == 2}
    assert col2_before == col2_after


def test_列ごとに違う高さになる():
    df, r, rd = _df({1: ([0.00] * 10, [0.03] * 10), 2: ([0.02] * 10, [0.08] * 10)})
    lim = ocr.empty_limits(df, r, rd)
    l1 = {lim[i] for i in r if df.at[i, 'col'] == 1}
    l2 = {lim[i] for i in r if df.at[i, 'col'] == 2}
    assert len(l1) == 1 and len(l2) == 1
    assert l1 != l2


def test_数が足りない列は表全体の高さに戻す():
    """出現の少ない地点では中央値が当てにならない"""
    df, r, rd = _df({1: ([0.00] * 10, [0.03] * 10), 2: ([0.05] * 2, [0.09] * 1)})
    lim = ocr.empty_limits(df, r, rd)
    table = ocr._thresholds(list(r.values()), list(rd.values()))
    assert {lim[i] for i in r if df.at[i, 'col'] == 2} == {table}


def test_列が無ければ表全体():
    df, r, rd = _df({1: ([0.00] * 10, [0.03] * 10)})
    df = df.drop(columns=['col'])
    lim = ocr.empty_limits(df, r, rd)
    table = ocr._thresholds(list(r.values()), list(rd.values()))
    assert set(lim.values()) == {table}


def test_読めたセルが無くても落ちない():
    df, r, rd = _df({1: ([0.00, 0.01, 0.02] * 4, [])})
    lim = ocr.empty_limits(df, r, rd)
    assert len(lim) == len(r)


def test_高さは何も無いと値があるのあいだ():
    blank, full = [0.0] * 10, [0.04] * 10
    limit, thin = ocr._thresholds(blank, full)
    assert 0.0 < limit < thin < 0.04
