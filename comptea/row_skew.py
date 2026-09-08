"""紙面の傾きを，画像を回さずに**セルの座標だけ**列ごとにずらして直す(2026-09-09)

`pipeline.grid.deskew_page()` は画像を回して検出し直すが，(1) 表が 1 つの紙面だけ，
(2) ずれが行の高さの 0.3 倍未満なら直さない，(3) 回すと検出が減る紙面では元に
戻す，の 3 つで 146 表のうち 5 表にしか効かず，0.1〜0.8° の傾きが残っていた．
残った傾きでも組成部の**左端の列**ではセルが切れる(05_p2 は左 1/5 の列で 22%，
08_p1 は 35%．右端は 0〜2%)．格子の行は組成部の幅ぜんたいの黒画素で決めるので，
その中央では合っていても端では合わない．

ここでは検出も画像も触らず，格子ができたあとに傾きを測り，各セルの y を
その x に応じてずらす(行の境が斜めの直線になる)．行番号は変えない．

傾きの測り方: **各行のセルの中の黒画素の重心**を列ごとに取り，x に対して直線を
当てる(行ごとの傾き)．その中央値を採る．`deskew.estimate()` の「左右の帯の
相互相関」は「・」の周期で半行以上のずれが曖昧になる(15 px 先と 20 px 手前が
同じに見える)が，こちらは行の帯の中で測るので曖昧にならない．
"""

import numpy as np
import pandas as pd

from . import ink


MIN_COLS = 5            # 傾きを測るのに要る最小の列数


MIN_POINTS = 5          # 1 行の直線当てはめに要る最小の点数(字のあるセル)


MIN_SPAN = 300          # 点の x の広がりがこれ未満の行は使わない(px)


MAX_ROWS = 60           # 測る行の数の上限(等間隔に間引く)


MIN_INK = 4             # セルに字があるとみなす黒画素の数


MIN_SHIFT = 2.0         # 端のセルのずれがこの px 未満なら直さない


MAX_DEG = 1.5           # これを超える推定は信じない(deskew.py と同じ)


AGREE_MAX = 0.5         # 行ごとの傾きの散らばり(MAD / 中央値の絶対値)がこれを超えたら信じない


def cell_centroids(dark, cells, min_ink=MIN_INK):
    """各セルの黒画素の重心 (x 中心, y 重心) を返す(字の無いセルは除く)"""
    pts = []
    for c in cells.itertuples(index=False):
        x1, x2 = int(c.x1), int(c.x2)
        y1, y2 = int(c.y1), int(c.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        sub = dark[y1:y2, x1:x2]
        n = int(sub.sum())
        if n < min_ink:
            continue
        rows = sub.sum(axis=1).astype(float)
        yc = float(np.dot(rows, np.arange(len(rows))) / rows.sum())
        pts.append(((x1 + x2) / 2.0, y1 + yc))
    return pts


def estimate_slope(dark, comp, max_rows=MAX_ROWS):
    """組成部のセルから傾き(dy/dx)を測る

    Returns:
        (傾き, 使った行の数)．測れないときは (0.0, 0)
    """
    if comp.empty or comp['col'].nunique() < MIN_COLS:
        return 0.0, 0
    rows = sorted(comp['row'].unique())
    if len(rows) > max_rows:
        idx = np.linspace(0, len(rows) - 1, max_rows).astype(int)
        rows = [rows[i] for i in idx]
    slopes = []
    for r in rows:
        pts = cell_centroids(dark, comp[comp['row'] == r])
        if len(pts) < MIN_POINTS:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        if xs.max() - xs.min() < MIN_SPAN:
            continue
        slope = float(np.polyfit(xs, ys, 1)[0])
        slopes.append(slope)
    if len(slopes) < 3:
        return 0.0, 0
    s = np.array(slopes)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    if abs(med) > 0 and mad / abs(med) > AGREE_MAX and mad > 1e-4:
        return 0.0, len(slopes)          # 行ごとにばらばら(測り損ね)
    if abs(np.degrees(np.arctan(med))) > MAX_DEG:
        return 0.0, len(slopes)
    return med, len(slopes)


def shear_cells(df, slope, x0):
    """段のセルの y を，x に応じて `slope * (x中心 - x0)` だけずらす"""
    out = df.copy()
    xc = (out['x1'].astype(float) + out['x2'].astype(float)) / 2.0
    dy = np.round(slope * (xc - x0))
    out['y1'] = out['y1'].astype(float) + dy
    out['y2'] = out['y2'].astype(float) + dy
    return out


def fix_skew(img, df_loc):
    """段ごとに傾きを測り，セルの座標だけを傾ける

    Returns:
        (直した格子, 警告のリスト)．直す所が無ければ元の格子をそのまま返す
    """
    if df_loc is None or len(df_loc) == 0 or 'block' not in df_loc.columns:
        return df_loc, []
    if 'row' not in df_loc.columns or 'col' not in df_loc.columns:
        return df_loc, []
    dark = None
    pieces = []
    warnings = []
    changed = False
    for block, g in df_loc.groupby('block', sort=True):
        comp = g[g['obj_name'] == 'comp']
        if comp.empty or comp['col'].nunique() < MIN_COLS:
            pieces.append(g)
            continue
        if dark is None:
            dark = ink.binarize(img)
        slope, n_rows = estimate_slope(dark, comp)
        x_lo, x_hi = float(comp['x1'].min()), float(comp['x2'].max())
        x0 = (x_lo + x_hi) / 2.0
        edge = abs(slope) * (x_hi - x_lo) / 2.0
        if slope == 0.0 or edge < MIN_SHIFT:
            pieces.append(g)
            continue
        pieces.append(shear_cells(g, slope, x0))
        changed = True
        deg = float(np.degrees(np.arctan(slope)))
        warnings.append(
            f'段{block}: **紙面の傾き {deg:+.2f}° をセルの座標で直した**'
            f'(組成部の幅 {x_hi - x_lo:.0f} px で左右のずれ {2 * edge:.0f} px．'
            f'{n_rows} 行の黒画素の重心から)．画像は回していない．'
            '行の境が斜めになるので，overlay ではセルが列ごとに上下する')
    if not changed:
        return df_loc, warnings
    return pd.concat(pieces, ignore_index=True), warnings
