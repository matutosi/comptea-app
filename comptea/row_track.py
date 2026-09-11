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

import re

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


RULE_SLANT = 2          # 縦罫線を探すとき，左右にこの px までのぶれを許す (紙面に
                        # 0.1〜0.8 度の傾きが残るので，まっすぐな連なりを探すと
                        # 途中で切れて消せない．消し残ると帯の全部の行に黒画素が
                        # あることになり，単位が縦に数珠つなぎになる (2026-09-09)


MIN_ROW_INK = 2         # 1 行の黒画素がこれ以下なら，その行は字が無いものとして扱う
                        # (斑点 1〜3 px が単位をつなぐ．「・」は 5 px 角ほどあるので消えない)


CUT_INK = 0.5           # 境が字の上を通るとみなす黒画素 (列の黒画素の中央値に対する比)


ANCHOR_COLS = 3         # 拍の候補にする組成の左端の列の数


CV_MAX = 0.15           # 拍の間隔の散らばりがこれを超える段は検算しない (実測 0.01〜0.09)


BEAT_DIFF = 3           # 格子と拍の行数がこれ以上食い違えば警告


# **組成の列も追従の対象にする** (2026-09-10 ユーザ目視 10 回目: 17_p1 は紙面が 12,466 px と
# 高く，一定の刻みで進むと「ブナ」の周辺 (行 200 前後) で組成の行が半行ほど下へずれ，
# 値の上端を切っていた．種名の列は合っている)．採用の条件は他の列と同じ
# 「字の単位を境が割る数が減るときだけ」なので，合っている表では動かない
SHIFT_CLASSES = ('sname', 'species_col', 'layer', 'comp')


FOLLOW_K = 5            # 行ごとのずれを滑らかにする窓 (行数．3・5・10 で差は無く，3 は活字で雑音を拾った)


FOLLOW_MAX = 0.25       # 行ごとのずれの上限 (行の高さの倍数)


FOLLOW_STEP = 0.15      # 隣り合う行のずれの差の上限 (行の高さの倍数．段差を作らない)


FIT_REACH = 0.4         # 境を空白へ寄せて探す範囲 (行の高さの倍数)


MIN_BAND = 0.5          # 行の高さのこの倍を下回る帯は作らない


def _strip_slanted(band, min_len, slant=RULE_SLANT):
    """軸 1 に長く続く黒を，軸 0 のぶれを許して消した写しを返す

    傾いた罫線は，まっすぐな連なりを探す `_strip_long_runs` では途中で切れて
    残る．軸 0 の前後 `slant` px に広げた写しで「長い連なり」を見つけ，その範囲に
    ある**元の**黒画素だけを消す (広げた分まで消すと，罫線の脇の字が欠ける)．

    縦罫線には転置して渡す (軸 0 が画像の x になる)．横罫線にはそのまま渡す．
    """
    if band.size == 0 or slant <= 0:
        return _strip_long_runs(band, min_len)
    wide = band.copy()
    for k in range(1, slant + 1):
        wide[k:, :] |= band[:-k, :]
        wide[:-k, :] |= band[k:, :]
    keep = _strip_long_runs(wide, min_len)       # 長い連なりだけが False になる
    out = band.copy()
    out[wide & ~keep] = False
    return out


def clean_rules(dark, x1, x2, y1, y2, pitch, slant=RULE_SLANT):
    """帯 `dark[y1:y2, x1:x2]` から下線・横罫線と縦罫線を消した写しを返す

    **横罫線も上下のぶれを許して消す** (2026-09-10)．紙面は 0.5 度ほど傾いており，
    幅 3,800 px の罫線は 1 本の画素行に収まらない．まっすぐな連なりだけを消すと
    切れ端が残り，10_p1 の群落記号の枠は 1,335 → 444 px しか減らなかった．
    """
    h, w = dark.shape
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    band = dark[y1:y2, x1:x2]
    if band.size == 0:
        return band.copy()
    out = _strip_slanted(band, int(max(RULE_H, RULE_H_P * pitch)), slant)
    out = _strip_slanted(np.ascontiguousarray(out.T), int(RULE_V_P * pitch) + 1,
                         slant).T
    return np.ascontiguousarray(out)


def units_in_band(band, y0=0, min_area=MIN_AREA, gap=UNIT_GAP, floor=MIN_ROW_INK):
    """帯の縦の射影の連なりを単位にして [(y 中心, 高さ, 黒画素)] を返す (画像の座標)

    1 行の黒画素が `floor` 以下の行は「字が無い」とみなす．罫線の消し残りや紙の
    汚れが 1〜3 px の斑点として全部の行に出ることがあり，そのままだと単位が縦に
    数珠つなぎになる (03_p2 は 157 行が 39 単位になっていた．2026-09-09)．
    """
    if band.size == 0:
        return []
    prof = band.sum(axis=1).astype(float)
    idx = np.flatnonzero(prof > floor)
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


def unit_spans(band, y0=0, min_area=MIN_AREA, gap=UNIT_GAP, floor=MIN_ROW_INK):
    """帯の縦の射影の連なりを [(上, 下, 黒画素)] で返す (画像の座標)"""
    if band.size == 0:
        return []
    prof = band.sum(axis=1).astype(float)
    idx = np.flatnonzero(prof > floor)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    out = []
    for a, b in zip(starts, ends):
        area = float(prof[a:b].sum())
        if area >= min_area:
            out.append((y0 + float(a), y0 + float(b), area))
    return out


def split_count(spans, ys):
    """境 `ys` が単位を割る数 (単位の内側を境が通るもの)"""
    if not spans:
        return 0
    arr = np.asarray(sorted(float(y) for y in ys))
    n = 0
    for a, b, _area in spans:
        if b - a <= 1:
            continue
        if np.any((arr > a + 0.5) & (arr < b - 0.5)):
            n += 1
    return int(n)


def split_units(dark, x1, x2, ys, pitch):
    """境 `ys` が字の単位を割る数と，単位の数

    「セルの上辺の黒画素が閾値を超えるか」より厳しく，閾値に頼らない物差し．
    緩い閾値の物差しは，境を字の薄い所へ置くだけで 0 になってしまう (2026-09-09)．
    """
    arr = sorted(float(y) for y in ys)
    if len(arr) < 2:
        return 0, 0
    y0, y1e = arr[0] - pitch, arr[-1] + pitch
    band = clean_rules(dark, x1, x2, y0, y1e, pitch)
    spans = unit_spans(band, y0=max(0, int(y0)))
    return split_count(spans, arr), len(spans)


def row_offsets(cys, centers, pitch, k=FOLLOW_K):
    """行ごとのずれを，前後 `k` 行の中央値で滑らかにして返す

    列ごとの中央値 1 つでは，同じ列の中の散らばり (四分位範囲 2〜12 px) が残る．
    近くの行だけで測ると紙面の癖に従える．欠けた行 (字の無い行) は前後から埋め，
    上限 (`FOLLOW_MAX`) と隣との差 (`FOLLOW_STEP`) で縛って段差を作らない．
    """
    centers = np.asarray(centers, dtype=float)
    arr = np.asarray(sorted(float(c) for c in cys))
    d = np.full(len(centers), np.nan)
    if arr.size:
        for i, c in enumerate(centers):
            near = arr[(arr >= c - 0.5 * pitch) & (arr <= c + 0.5 * pitch)]
            if near.size:
                d[i] = float(np.median(near)) - c
    if np.all(np.isnan(d)):
        return np.zeros(len(centers))
    half = max(1, int(k) // 2)
    out = np.zeros(len(centers))
    for i in range(len(centers)):
        w = d[max(0, i - half):i + half + 1]
        w = w[~np.isnan(w)]
        out[i] = float(np.median(w)) if w.size else np.nan
    if np.any(np.isnan(out)):                       # 前後から埋める
        idx = np.flatnonzero(~np.isnan(out))
        out = np.interp(np.arange(len(out)), idx, out[idx])
    out = np.clip(out, -FOLLOW_MAX * pitch, FOLLOW_MAX * pitch)
    step = FOLLOW_STEP * pitch
    for i in range(1, len(out)):                    # 隣との差を縛る (前へ)
        out[i] = min(max(out[i], out[i - 1] - step), out[i - 1] + step)
    for i in range(len(out) - 2, -1, -1):           # 後ろへも
        out[i] = min(max(out[i], out[i + 1] - step), out[i + 1] + step)
    return out


def follow_edges(edges, offs):
    """行ごとのずれを境に移す (境は前後の行のずれの平均で動かす)"""
    e = np.asarray(edges, dtype=float).copy()
    o = np.asarray(offs, dtype=float)
    if len(o) + 1 != len(e) or len(o) == 0:
        return e
    e[0] += o[0]
    e[-1] += o[-1]
    e[1:-1] += (o[:-1] + o[1:]) / 2.0
    return e


def fit_edges(dark, x1, x2, edges, pitch, reach=FIT_REACH):
    """内側の境を，最寄りの**黒画素ゼロ**の位置へ寄せる (順序と最小の高さを保つ)

    字は行の高さ (22〜35 px) に対して 16〜20 px なので片側 3〜9 px の余裕がある．
    その余裕を使い切るには，境を 1 本ずつ空白へ置くのがいちばん効く (実データで
    字を割る割合 46% → 3%．2026-09-09)．空白は**黒画素ゼロ**で定める (緩い閾値だと
    薄い字の上を空白と見なす)．±`reach` 倍に空白が無い境は動かさない．
    """
    e = [float(v) for v in edges]
    if len(e) < 3:
        return e, 0
    h = dark.shape[0]
    prof = dark[:, int(x1):int(x2)].sum(axis=1)
    r = max(1, int(reach * pitch))
    out = [e[0]]
    stuck = 0
    for i in range(1, len(e) - 1):
        lo = max(int(out[-1] + MIN_BAND * pitch), int(e[i]) - r, 0)
        hi = min(int(e[i]) + r, int(e[i + 1] - MIN_BAND * pitch), h - 1)
        best = None
        if hi >= lo:
            zeros = np.flatnonzero(prof[lo:hi + 1] == 0)
            if zeros.size:
                cand = zeros + lo
                best = float(cand[np.abs(cand - e[i]).argmin()])
        if best is None:
            stuck += 1
            best = max(e[i], out[-1] + MIN_BAND * pitch)
        out.append(best)
    out.append(e[-1])
    return out, stuck


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


def _edges_of(cells):
    """1 つの列の帯の境 (行 n 本 + 1)．行の順に並べる"""
    c = cells.sort_values('row')
    y1 = c.drop_duplicates('row')['y1'].astype(float).values
    y2 = c.drop_duplicates('row')['y2'].astype(float).values
    return np.r_[y1, y2[-1]]


def _apply_edges(df, mask, edges, note):
    """境の並びを，その帯のセルの y1/y2 に書き戻す"""
    out = df.copy()
    idx = out.index[mask]
    rows = out.loc[idx, 'row'].astype(int)
    order = {r: i for i, r in enumerate(sorted(rows.unique()))}
    pos = rows.map(order).values
    out.loc[idx, 'y1'] = np.asarray(edges, dtype=float)[pos]
    out.loc[idx, 'y2'] = np.asarray(edges, dtype=float)[pos + 1]
    out.loc[idx, 'note'] = note
    return out


def fix_offsets(img, df_loc, classes=SHIFT_CLASSES):
    """列ごとの y のずれを吸収する (段階 1・1b．行番号は全列で共有のまま)

    3 段で作り，**字の単位を境が割る数**がいちばん小さい案を採る (元のままが最小なら
    触らない)．物差しに「セルの上辺の黒画素」を使ってはいけない: 閾値が緩く，境を
    薄い所へ置くだけで 0 になる (2026-09-09)．

    1. 列ごとの**中央値**でずらす (系統的なずれ．タイプ打ちの文字と「・」の 10 px)
    2. **行ごとの追従** (前後 5 行の中央値．紙面の癖で行ごとに散らばる分)
    3. 境を**字の間の空白**へ置く (残りを詰める．いちばん効く)

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
    fixed = {}
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
            if cells.empty or cells['row'].nunique() < MIN_ROWS:
                continue
            for x1 in sorted(cells['x1'].unique()):
                band = cells[cells['x1'] == x1]
                if band['row'].nunique() < MIN_ROWS:
                    continue
                w = _fit_band(dark, band, pitch)
                if w is None:
                    continue
                edges, note, msg = w
                mask = ((out['block'] == block) & (out['obj_name'] == cls)
                        & (out['x1'] == x1))
                if note is None:
                    warnings.append(f'段{block}: 列 {cls} は' + msg)
                    continue
                out = _apply_edges(out, mask, edges, note)
                changed = True
                fixed.setdefault((block, cls), []).append(msg)
    # **列ごとに 1 行にまとめる** (2026-09-10)．組成部を対象に足したら，地点が 128 列の
    # 17_p1 で警告が 120 本を超えた．直した本数と，字を割る単位の合計だけを出す
    for (block, cls), msgs in fixed.items():
        tot = [0, 0]
        for m in msgs:
            n = re.search(r'字を割る単位 (\d+) → (\d+)/(\d+)', m)
            if n:
                tot[0] += int(n.group(1))
                tot[1] += int(n.group(2))
        if len(msgs) == 1:
            warnings.append(f'段{block}: **列 {cls} の境を直した**' + msgs[0])
        else:
            warnings.append(
                f'段{block}: **列 {cls} の境を直した** ({len(msgs)} 列．'
                f'字を割る単位 {tot[0]} → {tot[1]})．行ごとに追従し，境を字の間の空白へ')
    if not changed:
        return df_loc, warnings
    return out, warnings


def _fit_band(dark, band, pitch):
    """1 つの列の帯で 3 段の案を作り，字を割る数がいちばん小さいものを返す

    Returns:
        (境, note, 説明) か，触らないときは (None, None, 理由)．測れなければ None
    """
    x1, x2 = _band_x(band)
    edges0 = _edges_of(band)
    n_rows = len(edges0) - 1
    cys, _x = _units_of(dark, band, pitch)
    need = max(MIN_UNITS, int(np.ceil(MIN_MATCH * n_rows)))
    y0, y1e = edges0[0] - pitch, edges0[-1] + pitch
    spans = unit_spans(clean_rules(dark, x1, x2, y0, y1e, pitch), y0=max(0, int(y0)))
    if not spans:
        return None
    base = split_count(spans, edges0)
    if base == 0:
        return None                       # 元から字を割っていない列は触らない (黙る)
    centers = (edges0[:-1] + edges0[1:]) / 2.0

    cands = [(edges0, None, 'ずらさなかった')]
    dy, iqr, n = column_offset(cys, centers, pitch) if cys else (0.0, 0.0, 0)
    few = len(cys) >= FEW_UNITS and iqr <= FEW_IQR * pitch
    ok = len(cys) >= need or few
    if ok and abs(dy) <= MAX_SHIFT * pitch and iqr <= MAX_IQR * pitch:
        e1 = edges0 + round(dy)
        cands.append((e1, f'y_shifted:{round(dy):+d}', f'({round(dy):+d} px ずらし'))
        offs = row_offsets(cys, centers + round(dy), pitch)
        e2 = follow_edges(e1, offs)
        cands.append((e2, 'y_followed',
                      f'({round(dy):+d} px ずらし，行ごとに追従'))
    else:
        e2 = edges0
    e3, stuck = fit_edges(dark, x1, x2, cands[-1][0], pitch)
    cands.append((e3, 'y_fitted', ('(' if len(cands) == 1 else cands[-1][2] + '，')
                  + '境を字の間の空白へ'))
    scores = [split_count(spans, e) for e, _n, _m in cands]
    best = int(np.argmin(scores))
    if best == 0 or scores[best] >= base:
        if not ok:
            return None, None, f'ずらさなかった(単位が少ない: {len(cys)} < {need})'
        return None, None, f'ずらさなかった(直しても字を割る数が {base} から減らない)'
    e, note, msg = cands[best]
    return (np.round(e, 1), note,
            f'{msg}．字を割る単位 {base} → {scores[best]}/{len(spans)}'
            + (f'．空白が無く動かせない境 {stuck} 本' if note == 'y_fitted' and stuck else '')
            + ')')
