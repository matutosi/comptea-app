"""組成表の載っていないページの判定 (filters.looks_like_no_table)

「本物の表なのに `col` が取れない段を守る」条件が，**一度も発動しない死んだ条件**
だった (2026-09-09)．`locate_items` は `obj_name` に `row` を作らないので
`(name == 'row').sum()` は常に 0 で，種名の列が 60 行あっても「表が無い」とされる．
1 調査区の 2 段組 (kinki_047・053) はこれで格子ごと失われていた．
"""
import pandas as pd

from comptea import filters


def _loc(**counts):
    rows = []
    for obj, n in counts.items():
        for i in range(n):
            rows.append(dict(obj_name=obj, x1=0.0, x2=10.0,
                             y1=float(i * 10), y2=float(i * 10 + 8)))
    return pd.DataFrame(rows, columns=['obj_name', 'x1', 'x2', 'y1', 'y2'])


def test_種名の列が並んでいれば表とみなす():
    # kinki_047 の型: 学名 60・和名 60・階層 29 があるのに comp も col も無い
    df = _loc(sname=60, species_col=60, layer=29, header=1)
    assert not filters.looks_like_no_table(df)


def test_組成セルがあれば表():
    assert not filters.looks_like_no_table(_loc(comp=20, sname=10))


def test_列があれば表():
    assert not filters.looks_like_no_table(_loc(col=5, sname=10))


def test_本文だけのページは表ではない():
    assert filters.looks_like_no_table(_loc(header=1))


def test_種名が数行しか無ければ表ではない():
    # 切れ端 (表題や凡例の帯) は種名の列が数行しか出ない
    assert filters.looks_like_no_table(_loc(sname=2, species_col=2))


def test_空なら表ではない():
    assert filters.looks_like_no_table(pd.DataFrame())
    assert filters.looks_like_no_table(None)
