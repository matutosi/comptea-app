"""行を「線」でなく**単位の連なり**として追う (案 e の段階 1．2026-09-09)

人が行を読むときは，黒画素の谷 (投影) を探すのではなく，同じ見た目の**単位**
(「・」，階層の記号，種名の 1 行) が一定の**拍子**で並ぶのを目印の列で数え，
同じ高さにある字を結び付けて 1 行と読む．146 表で測ると，行の**数**は現行の格子で
ほぼ合っている (真値との一致 recall 1.000．拍と格子の差 ±2 以内 93%) が，
**行に字を付ける側**が崩れていた: 境を全列で共有して組成の谷に置くので，
タイプ打ちでは文字と「・」の高さが 10 px 違い，学名・和名の字を 25% の行で割る．
列ごとの y のずれは表ごとにほぼ一定 (学名 −10〜+12 px，四分位範囲 2〜12 px) で，
**列ごとの中央値 1 つで吸収できる**．中央値を引いてから最寄りの行に付け直すと，
学名と和名が同じ行に付く割合は 93% → 98% (14_p1 は 39% → 98%)．

段階 1 でするのは 2 つ．
- `fix_offsets`: 学名・和名・階層の列ごとに，単位の y 中心と行の中心との差の中央値
  `dy` を測り，受け入れ判定をすべて満たした列だけ y1/y2 を `dy` ずらす
  (`row_skew.shear_cells` の定数版．行番号 `row` は全列で共有したまま)．
  判定は |dy| ≥ 2 px，|dy| ≤ 0.45 p (実測の最大 0.42 p)，四分位範囲 ≤ 0.4 p，
  単位の数 ≥ max(5, 行数の 3 割) (10 個以上あり四分位範囲 ≤ 0.25 p ならこれは問わない．
  階層の列は行の一部にしか記号が無い)，**ずらして「字の上を通る境」が増えない**こと．
  最後の判定で，活字 (学名 4%・和名 3%) のように良い表を触って悪くしない．
- `check_beats`: 目印の列 (階層 + 組成の左端 3 列) の拍で行数を数え，格子と 3 行以上
  食い違えば警告する．目印は 1 列に決めない (階層だけ 54%，左端 1 列 81%，候補の中央値
  94% [格子と ±2 以内の段の割合])．端の行 (種群の見出し) は拍に出ないので学名の字で足す．

単位は列の x 帯の**縦の射影の連なり**で切る (連結成分に近い．OCR の検出器の箱は
組成では出ない: 「・」210 個に箱 1〜3 個)．切る前に下線 (横 ≥ max(80 px, 2p)) と
縦罫線 (縦 > 3p．相対値は長い表で効かない) を消す．高さ 1.5 p を超える単位 (下線で
つながった見出し，行間 3 px 未満の和名) は推定に入れない．

置き場所は `row_skew.fix_skew` の直後 (傾きを除いた残りだけを測る) で `cell_id` の前．
段階 2 (境を谷でなく拍の中点に置く) と段階 3 (見出し・凡例を note に付ける) は別途．
"""

import numpy as np
import pandas as pd

from . import ink
from .body_rows import _strip_long_runs


MIN_ROWS = 5            # 段に要る最小の行数


MIN_SHIFT = 2.0         # ずれがこの px 未満なら動かさない (row_skew と同じ)


MAX_SHIFT = 0.45        # 行の高さのこの倍を超えるずれは信用しない (実測の最大 0.42 p)


MAX_IQR = 0.4           # 単位のずれの四分位範囲がこの倍を超えたら信用しない (実測 ≤ 0.35 p)


MIN_UNITS = 5           # 列に要る最小の単位の数


MIN_MATCH = 0.3         # 単位の数が行数のこの割合に満たない列は触らない (複数階層で和名が空く表でも残る)


FEW_UNITS = 10          # ただし単位がこれ以上あり，ずれがそろって (下) いれば測ってよい


FEW_IQR = 0.25          # 少ない単位で測るときの四分位範囲の上限 (行の高さの倍数)


TALL_UNIT = 1.5         # 行の高さのこの倍より高い単位は推定に入れない (2 行以上がつながったもの)


MIN_AREA = 6            # これより黒画素の少ない連なりはごみ


UNIT_GAP = 2            # 縦にこの px 以下の隙間は同じ単位 (濁点・活字の足)


RULE_H = 80             # 横にこの px 以上続く黒は下線・罫線 (見出しの下線は 136 px 以上)


RULE_H_P = 2.0          # 同上 (行の高さの倍数．大きいほう)


RULE_V_P = 3.0          # 縦に行の高さのこの倍を超えて続く黒は縦罫線


CUT_INK = 0.5           # 境が字の上を通るとみなす黒画素 (列の黒画素の中央値に対する比)


ANCHOR_COLS = 3         # 拍の候補にする組成の左端の列の数


CV_MAX = 0.15           # 拍の間隔の散らばりがこれを超える段は検算しない (実測 0.01〜0.09)


BEAT_DIFF = 3           # 格子と拍の行数がこれ以上食い違えば警告


SHIFT_CLASSES = ('sname', 'species_col', 'layer')


def clean_rules(dark, x1, x2, y1, y2, pitch):
    """帯 `dark[y1:y2, x1:x2]` から下線・横罫線と縦罫線を消した写しを返す"""
    h, w = dark.shape
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    band = dark[y1:y2, x1:x2]
    if band.size == 0:
        return band.copy()
    out = _strip_long_runs(band, int(max(RULE_H, RULE_H_P * pitch)))
    out = _strip_long_runs(np.ascontiguousarray(out.T), int(RULE_V_P * pitch) + 1).T
    return np.ascontiguousarray(out)


def units_in_band(band, y0=0, min_area=MIN_AREA, gap=UNIT_GAP):
    """帯の縦の射影の連なりを単位にして [(y 中心, 高さ, 黒画素)] を返す (画像の座標)"""
    if band.size == 0:
        return []
    prof = band.sum(axis=1).astype(float)
    idx = np.flatnonzero(prof > 0)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    out = []
    for a, b in zip(starts, ends):
        seg = prof[a:b]
        area = float(seg.sum())
        if area < min_area:
            continue
        cy = float(np.dot(seg, np.arange(a, b)) / area)
        out.append((y0 + cy, float(b - a), area))
    return out


def beat_rows(cys, pitch):
    """単位の y 中心の並びから拍を数える

    間隔が 0.5 p 未満の単位は 1 つに束ね (2 段の値・濁点)，間隔の中央値を p とする．
    抜け (間隔 ≈ 2p, 3p) は拍子から補う．戻り値は dict(n, n_est, p, cv, first, last)．
    """
    cys = sorted(float(c) for c in cys)
    if not cys:
        return None
    merged = [[cys[0]]]
    for c in cys[1:]:
        if c - merged[-1][-1] < 0.5 * pitch:
            merged[-1].append(c)
        else:
            merged.append([c])
    centers = np.array([np.mean(m) for m in merged])
    if len(centers) < 2:
        return dict(n=len(centers), n_est=len(centers), p=float(pitch), cv=0.0,
                    first=float(centers[0]), last=float(centers[-1]))
    gaps = np.diff(centers)
    ok = gaps[(gaps >= 0.5 * pitch) & (gaps <= 1.6 * pitch)]
    p = float(np.median(ok)) if len(ok) >= 3 else float(pitch)
    k = np.maximum(1, np.round(gaps / p))
    resid = gaps - k * p
    # 行数は両端の差から (間隔ごとの丸めの誤りが積もらない．13_p2 で 65 → 64)
    n_est = int(1 + round((centers[-1] - centers[0]) / p))
    return dict(n=len(centers), n_est=n_est, p=p, cv=float(np.std(resid) / p),
                first=float(centers[0]), last=float(centers[-1]))


def cut_count(dark, x1, x2, ys, ratio=CUT_INK):
    """帯 `x1..x2` で，境 `ys` のうち字の上を通るものの数"""
    h = dark.shape[0]
    prof = dark[:, int(x1):int(x2)].sum(axis=1).astype(float)
    pos = prof[prof > 0]
    if len(pos) == 0:
        return 0
    thr = float(np.median(pos)) * ratio
    n = 0
    for y in np.asarray(ys, dtype=float):
        yi = int(round(y))
        if 0 <= yi < h and prof[yi] > thr:
            n += 1
    return n


def _band_x(cells):
    return int(cells['x1'].min()), int(cells['x2'].max())


def _row_centers(cells):
    """行ごとのセルの中心 y (列で違うときは中央値) を row 順に返す"""
    c = cells.groupby('row')[['y1', 'y2']].median()
    return ((c['y1'] + c['y2']) / 2.0).sort_index().values


def column_offset(cys, centers, pitch):
    """単位を最寄りの行の中心に付け，ずれの中央値 `dy` と，引いた後の四分位範囲を返す"""
    cys = np.asarray(cys, dtype=float)
    centers = np.sort(np.asarray(centers, dtype=float))
    if len(cys) == 0 or len(centers) == 0:
        return 0.0, 0.0, 0
    d1 = cys - centers[np.abs(cys[:, None] - centers[None, :]).argmin(axis=1)]
    dy = float(np.median(d1))
    moved = cys - dy
    d2 = moved - centers[np.abs(moved[:, None] - centers[None, :]).argmin(axis=1)]
    q1, q3 = np.percentile(d2, [25, 75])
    return dy, float(q3 - q1), int(len(cys))


def _units_of(dark, cells, pitch, drop_tall=True):
    """セルの帯 (x は列の範囲，y は格子の上下に半行の余裕) の単位の y 中心"""
    x1, x2 = _band_x(cells)
    y1 = float(cells['y1'].min()) - pitch / 2.0
    y2 = float(cells['y2'].max()) + pitch / 2.0
    band = clean_rules(dark, x1, x2, y1, y2, pitch)
    units = units_in_band(band, y0=max(0, int(y1)))
    if drop_tall:
        units = [u for u in units if u[1] <= TALL_UNIT * pitch]
    return [u[0] for u in units], (x1, x2)


def count_rows(dark, g, pitch):
    """目印の列 (階層 + 組成の左端 3 列) の拍で段の行数を数える

    戻り値は dict(n_est, p, cv, above, below, n_full, n_span, per_col)．目印が無ければ None．
    `n_est` は候補の列ごとの拍の数の中央値，`above`/`below` は拍の外に学名側の字がある
    行 (種群の見出し)，`n_full` = n_est + above + below，`n_span` は格子の高さを拍の
    間隔で割った行数 (格子と比べるのはこれ．拍の両端は格子の外の行を数えられない)．
    """
    comp = g[g['obj_name'] == 'comp']
    if comp.empty:
        return None
    cands = []
    layer = g[g['obj_name'] == 'layer']
    if not layer.empty:
        cands.append(('layer', layer))
    for c in sorted(comp['col'].unique())[:ANCHOR_COLS]:
        cands.append((f'comp{int(c)}', comp[comp['col'] == c]))
    per_col = {}
    for key, cells in cands:
        cys, _x = _units_of(dark, cells, pitch, drop_tall=False)
        r = beat_rows(cys, pitch)
        if r is None or r['n'] < MIN_UNITS:
            continue
        per_col[key] = r
    if not per_col:
        return None
    # 散らばりの大きい候補 (階層の列は単位が少なく字が縦につながる) は中央値から外す
    steady = [r for r in per_col.values() if r['cv'] <= CV_MAX]
    rs = steady or list(per_col.values())
    n_est = int(np.median([r['n_est'] for r in rs]))
    p = float(np.median([r['p'] for r in rs]))
    cv = float(np.median([r['cv'] for r in rs]))
    first = float(np.median([r['first'] for r in rs]))
    last = float(np.median([r['last'] for r in rs]))
    names = g[g['obj_name'].isin(['sname', 'species_col'])]
    above = below = 0
    if not names.empty:
        x1, x2 = _band_x(names)

        def has_text(ya, yb):
            band = clean_rules(dark, x1, x2, ya, yb, pitch)
            return any(u[2] >= MIN_AREA for u in units_in_band(band))

        above = int(has_text(first - 1.5 * p, first - 0.5 * p))
        below = int(has_text(last + 0.5 * p, last + 1.5 * p))
    extent = float(comp['y2'].max()) - float(comp['y1'].min())
    return dict(n_est=n_est, p=p, cv=cv, above=above, below=below,
                n_full=n_est + above + below, n_span=int(round(extent / p)),
                per_col=per_col)


def _pitch_of(comp):
    return float(np.median(comp['y2'].astype(float) - comp['y1'].astype(float)))


def check_beats(img, df_loc):
    """段ごとに拍の行数と格子の行数を比べ，3 行以上食い違えば知らせる (格子は変えない)"""
    if df_loc is None or len(df_loc) == 0 or 'block' not in df_loc.columns:
        return []
    dark = None
    warnings = []
    for block, g in df_loc.groupby('block', sort=True):
        comp = g[g['obj_name'] == 'comp']
        n_grid = int(comp['row'].nunique()) if not comp.empty else 0
        if n_grid < MIN_ROWS:
            continue
        if dark is None:
            dark = ink.binarize(img)
        r = count_rows(dark, g, _pitch_of(comp))
        if r is None or r['cv'] > CV_MAX:
            continue
        if abs(n_grid - r['n_span']) >= BEAT_DIFF:
            warnings.append(
                f'段{block}: 格子 {n_grid} 行に対し，目印の列の拍の間隔 {r["p"]:.1f} px では'
                f'格子の高さに {r["n_span"]} 行しか入らない'
                f'(単位の並びからは {r["n_full"]} 行．散らばり {r["cv"]:.2f}，候補 {len(r["per_col"])} 列)．'
                f'{BEAT_DIFF} 行以上食い違う．段階1で行の対応を目で確かめる')
    return warnings


def shift_column(df, mask, dy):
    """`mask` のセルの y1/y2 を `dy` ずらした写し (行番号は変えない)"""
    out = df.copy()
    out.loc[mask, 'y1'] = out.loc[mask, 'y1'].astype(float) + dy
    out.loc[mask, 'y2'] = out.loc[mask, 'y2'].astype(float) + dy
    out.loc[mask, 'note'] = 'y_shifted'
    return out


def fix_offsets(img, df_loc, classes=SHIFT_CLASSES):
    """列の種類ごとに y のずれを中央値で吸収する (行番号は共有のまま)

    Returns:
        (直した格子, 警告のリスト)．直す所が無ければ元の格子をそのまま返す
    """
    if df_loc is None or len(df_loc) == 0 or 'block' not in df_loc.columns:
        return df_loc, []
    if 'row' not in df_loc.columns or 'obj_name' not in df_loc.columns:
        return df_loc, []
    dark = None
    out = df_loc
    warnings = []
    changed = False
    for block, g in df_loc.groupby('block', sort=True):
        comp = g[g['obj_name'] == 'comp']
        n_rows = int(comp['row'].nunique()) if not comp.empty else 0
        if n_rows < MIN_ROWS:
            continue
        pitch = _pitch_of(comp)
        if not pitch > 0:
            continue
        if dark is None:
            dark = ink.binarize(img)
        for cls in classes:
            cells = g[g['obj_name'] == cls]
            if cells.empty:
                continue
            cys, (x1, x2) = _units_of(dark, cells, pitch)
            need = max(MIN_UNITS, int(np.ceil(MIN_MATCH * n_rows)))
            dy, iqr, n = column_offset(cys, _row_centers(cells), pitch)
            # 階層の列は行の一部にしか記号が無い (14_p1 は 211 行に 17 個)．
            # 数が少なくても**ずれがそろっていれば**測ってよい (四分位範囲 ≤ 0.25 p)
            if len(cys) < need and not (len(cys) >= FEW_UNITS and iqr <= FEW_IQR * pitch):
                warnings.append(f'段{block}: 列 {cls} はずらさなかった'
                                f'(単位が少ない: {len(cys)} < {need})')
                continue
            if abs(dy) < MIN_SHIFT:
                continue
            if abs(dy) > MAX_SHIFT * pitch or iqr > MAX_IQR * pitch:
                warnings.append(f'段{block}: 列 {cls} はずらさなかった'
                                f'(推定が信用できない: ずれ {dy:+.0f} px，四分位範囲 {iqr:.0f} px，'
                                f'行の高さ {pitch:.0f} px)')
                continue
            dy_px = int(round(dy))
            ys = np.concatenate([cells['y1'].astype(float).values,
                                 cells['y2'].astype(float).values])
            before = cut_count(dark, x1, x2, ys)
            after = cut_count(dark, x1, x2, ys + dy_px)
            if after > before:
                warnings.append(f'段{block}: 列 {cls} はずらさなかった'
                                f'(ずらすと字の上を通る境が {before} → {after} に増える)')
                continue
            mask = (out['block'] == block) & (out['obj_name'] == cls)
            out = shift_column(out, mask, dy_px)
            changed = True
            warnings.append(
                f'段{block}: **列 {cls} の y を {dy_px:+d} px ずらした**'
                f'(単位 {n} 個，四分位範囲 {iqr:.0f} px．字の上を通る境 {before} → {after})．'
                '行番号は共有のまま．タイプ打ちでは文字と「・」の高さが違う．'
                '段階1で目で確かめる')
    if not changed:
        return df_loc, warnings
    return out, warnings
