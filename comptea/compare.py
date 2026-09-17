"""2 つの通しを比べる物差し (2026-09-16 に使い捨てのスクリプトから切り出した)

直しの良し悪しを**要確認の数だけで決めてはいけない**と，2026-09-16 だけで
4 回思い知らされた (`docs/lessons.md`「要確認の数では，誤りを測れない」)．要確認が減っても，本物の値が消えて
いたり (05_p2 で 17 件)，罫線を `1` と読んでいたり (kinki_045-1 は `5`・`4`・`3`
が全部 `1`)，隣の列へ値が移っていたりした (21_p4 は 1,524 セル)．

そこで使った 2 つの物差しを，ここに置く．

- `diff_long`: 縦持ちの表 (`comp_table_long.csv`) を 1 セルずつ突き合わせる．
  **行番号の無い行は，地点ごとの値の多重集合で比べる**．`(地点, 行)` を鍵に
  すると `NaN != NaN` で同じ行が毎回別物になり，変化を 10 倍に数えていた
  (押し出しの「他の地点での変化」が 253 件，正しくは 21 件)．
- `boundary_ink`: 箱の境 (左右の線) に黒画素が乗るセルの数．**格子だけで
  数秒で出る**．傾きが本当に合っていれば境は値を切らなくなり減る．
  **上下に分けて数える**と，一様でない歪みが見える (21_p4 は上 23% → 21% なのに
  下 7% → 44%)．
"""
import collections

import pandas as pd


def diff_long(a, b, value='comp_raw'):
    """2 つの縦持ちの表の違いを数える

    Args:
        a, b: `comp_table_long.csv` を読んだもの
    Returns:
        {'changed': 変わったセルの数, 'by_plot': 地点ごとの数 (Counter),
         'cells': [(地点, 行, 前, 後), ...] (行番号のある行だけ)}
        前・後の値が無いときは None
    """
    def keyed(df):
        rows = df[df['row_no'].notna()]
        return {(p, int(r)): (None if pd.isna(v) else str(v))
                for p, r, v in zip(rows['plot'], rows['row_no'], rows[value])}

    def loose(df):
        rows = df[df['row_no'].isna()]
        return collections.Counter(
            (p, None if pd.isna(v) else str(v))
            for p, v in zip(rows['plot'], rows[value]))

    ka, kb = keyed(a), keyed(b)
    cells = []
    by_plot = collections.Counter()
    for k in sorted(set(ka) | set(kb), key=lambda t: (t[0], t[1])):
        if ka.get(k) != kb.get(k):
            cells.append((k[0], k[1], ka.get(k), kb.get(k)))
            by_plot[k[0]] += 1
    la, lb = loose(a), loose(b)
    for k in set(la) | set(lb):
        d = abs(la.get(k, 0) - lb.get(k, 0))
        if d:
            by_plot[k[0]] += d
    return {'changed': sum(by_plot.values()), 'by_plot': by_plot, 'cells': cells}


def boundary_ink(dark, cells, sides=('x1', 'x2'), pad=1):
    """境の線 (±`pad` px) に黒画素が乗るセルの数

    Args:
        dark: `ink.binarize()` の結果 (True が黒)
        cells: x1・y1・x2・y2 を持つセルの表
        sides: 見る境．左だけなら ('x1',)
    """
    h, w = dark.shape
    n = 0
    for x1, y1, x2, y2 in zip(cells['x1'], cells['y1'], cells['x2'], cells['y2']):
        ya, yb = int(max(0, y1)), int(min(h, y2))
        if yb <= ya:
            continue
        xs = {'x1': x1, 'x2': x2}
        for s in sides:
            a, b = int(max(0, xs[s] - pad)), int(min(w, xs[s] + pad + 1))
            if b > a and dark[ya:yb, a:b].any():
                n += 1
                break
    return n


def boundary_ink_halves(dark, cells, sides=('x1', 'x2'), pad=1):
    """上半分と下半分に分けて `boundary_ink` を数える

    Returns:
        ((上の数, 上のセル数), (下の数, 下のセル数))
    """
    if len(cells) == 0:
        return (0, 0), (0, 0)
    mid = (float(cells['y1'].min()) + float(cells['y2'].max())) / 2.0
    top = cells[cells['y2'] <= mid]
    bot = cells[cells['y1'] >= mid]
    return ((boundary_ink(dark, top, sides, pad), len(top)),
            (boundary_ink(dark, bot, sides, pad), len(bot)))
