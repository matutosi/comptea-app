"""1 調査区の 2 段組で，階層と被度の列を黒画素から作る (対策 G．2026-09-09)

資料には「1 つの調査区の出現種を左右 2 段に折り返した」紙面がある
(kinki_001・047・053・077 の 4 表)．1 段は

    学名 | 和名 | 階層 (B・S・K) | 被度 (3・3，1・2，+ など)

の 4 つからなる．**地点の列が 1 本しかない**ので `col` の検出が育たず (段あたり
0〜1 本，信頼度 0.20)，被度が丸ごと落ちるか，階層の上に乗る．047・053 は `col` が
0 本で格子そのものが作れていなかった (`filters.looks_like_no_table` の死んだ条件も
重なっていた．そちらは別に直した)．

**分かれ目は「行ごとのインクの塊の左端」の山**．素の縦の投影では決まらない
(和名のカタカナの字間 10〜14 px と，和名と階層のあいだの谷 11〜45 px が重なる)．
塊の左端を数えると，和名の中の山の間隔は行の高さの 0.6〜0.8 倍なのに対し，
**和名と階層のあいだだけ 2.1〜3.6 倍**あく (8 段で実測．重なりが無い)．
学名と和名のあいだはさらに大きく空くので，**いちばん大きい間隔ではなく
「最後の」大きな間隔**を採る．

階層と被度の境は，階層の左端から 0.3〜0.9 行の窓の中で票 (字のある行の数) が
行数の 15% 以下になる**右の谷**．左の谷を採ると `B₁` の添字が被度に入る．
窓を 1.3 行まで広げると 053 の段 1 で被度の中の谷を拾って外れた．

この処理は**検出の箱 (`col` と `layer`) を段ごとに差し替えるだけ**で，
`locate` 以降は何も変えない．
"""

import numpy as np
import pandas as pd

from . import ink


MIN_ROWS = 15           # 段の行がこれ未満なら当てない (統計が足りない．077 は 18・19 行，
                        # 053 の段 1 は 19 行なので 20 では弾いてしまう)


RUN_GAP = 6             # 行の中でインクをつなぐ幅 (px)．添字を字につなぐ


SMOOTH = 4              # 塊の左端の山をならす幅 (±px)


PEAK_K = 0.25           # 山とみなす高さ (段の字のある行数に対する比)


BIG_GAP = 2.0           # 和名と階層のあいだの山の間隔 (行の高さの倍数．実測 2.1〜3.6，
                        # 和名の中は 0.6〜0.8．重なりが無い)


LAYER_ROWS = 0.6        # 階層の山の高さ (行数に対する比．実測 0.81〜0.95)


LAYER_MIN = 0.3         # 階層と被度の境を探す窓の下限 (行の高さの倍数)


LAYER_MAX = 0.9         # 同上限 (1.3 にすると 053 の段 1 で外れる)


LOW_K = 0.15            # 谷とみなす票 (行数に対する比)


EDGE_K = 0.03           # 被度の右端とみなす票


RULE_FRAC = 0.5         # 横に**続けて**この割合を超える黒がある行は横罫線として落とす
                        # (黒画素の合計で見てはいけない．字の多い行が罫線と判定される)


LAYER_W = (0.4, 1.1)    # 検算: 階層の幅 (行の高さの倍数)


COVER_W = (0.7, 2.0)    # 検算: 被度の幅 (行の高さの倍数)


def _rows_of(det, block):
    r = det[(det['obj_name'] == 'row')]
    if r.empty:
        return None
    inside = r[(r['x1'] >= block[0] - 5) & (r['x2'] <= block[1] + 5)]
    use = inside if len(inside) >= 3 else r
    return sorted((float(a), float(b)) for a, b in zip(use['y1'], use['y2']))


def left_peaks(dark, x1, x2, rows, gap=RUN_GAP):
    """行ごとの「インクの塊の左端」を数えた並びを返す (x1 を 0 とする)"""
    x1, x2 = int(x1), int(x2)
    if x2 - x1 < 10:
        return np.zeros(0)
    hist = np.zeros(x2 - x1, dtype=int)
    for a, b in rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        band = dark[a:b, x1:x2]
        cols = band.any(axis=0)
        if _is_rule(cols):                    # 横罫線の行は使わない
            continue
        idx = np.flatnonzero(cols)
        if idx.size == 0:
            continue
        starts = np.r_[idx[0], idx[np.flatnonzero(np.diff(idx) > gap) + 1]]
        hist[starts] += 1
    return hist


def _is_rule(cols, frac=RULE_FRAC):
    """横に続けて `frac` 倍以上の黒があるか (罫線の行)"""
    if cols.size == 0:
        return False
    need = cols.size * frac
    best = cur = 0
    for v in cols:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
        if best > need:
            return True
    return False


def _smooth(hist, w=SMOOTH):
    if hist.size == 0:
        return hist
    k = np.ones(2 * w + 1)
    return np.convolve(hist.astype(float), k, mode='same')


def _peak_positions(hist, need):
    """山 (`need` 以上) の位置を，連なりごとに 1 つずつ返す"""
    on = np.flatnonzero(hist >= need)
    if on.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(on) > 1)
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]] + 1
    return [(int(a), int(b), float(hist[a:b].max())) for a, b in zip(starts, ends)]


def split_block(dark, x1, x2, rows, pitch):
    """1 段の中の (階層の左端, 被度の左端, 被度の右端) を返す．決まらなければ None"""
    n_row = len(rows)
    x1, x2 = int(x1), int(x2)
    if n_row < MIN_ROWS or x2 - x1 < 4 * pitch:
        return None
    hist = _smooth(left_peaks(dark, x1, x2, rows))
    peaks = _peak_positions(hist, PEAK_K * n_row)
    if len(peaks) < 3:
        return None
    # **最後の**大きな間隔 (和名と階層のあいだ)．学名と和名のあいだはもっと空く
    big = None
    for (a1, b1, _h1), (a2, _b2, h2) in zip(peaks[:-1], peaks[1:]):
        if a2 - b1 >= BIG_GAP * pitch and h2 >= LAYER_ROWS * n_row:
            big = (b1, a2)
    if big is None:
        return None
    lay_x = x1 + big[1]
    # 階層と被度の境: 階層の左端から 0.3〜0.9 行の窓で，票が低い所の**右の谷**
    prof = np.zeros(x2 - x1, dtype=int)
    for a, b in rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        cols = dark[a:b, int(x1):int(x2)].any(axis=0)
        if _is_rule(cols):
            continue
        prof += cols.astype(int)
    lo = int(big[1] + LAYER_MIN * pitch)
    hi = min(len(prof) - 1, int(big[1] + LAYER_MAX * pitch))
    if hi <= lo:
        return None
    low = np.flatnonzero(prof[lo:hi + 1] <= LOW_K * n_row)
    if low.size == 0:
        return None
    runs = np.split(low, np.flatnonzero(np.diff(low) > 1) + 1)
    a, b = runs[-1][0], runs[-1][-1]                 # 右の谷
    cov_x = x1 + lo + (a + b) / 2.0
    # 被度の右端: 票が消えて 0.6 行ぶん続く所
    right = x2
    tail = np.flatnonzero(prof[int(cov_x - x1):] > EDGE_K * n_row)
    if tail.size:
        end = int(cov_x - x1) + int(tail[-1]) + 1
        right = min(x2, x1 + end + max(2, int(0.1 * pitch)))
    if not (LAYER_W[0] * pitch <= cov_x - lay_x <= LAYER_W[1] * pitch):
        return None
    if not (COVER_W[0] * pitch <= right - cov_x <= COVER_W[1] * pitch):
        return None
    return float(lay_x), float(cov_x), float(right)


def _overlap(box, x1, x2):
    """箱と範囲 `x1`〜`x2` の重なりの割合 (箱の幅に対する比)"""
    a, b = float(box.x1), float(box.x2)
    if b <= a:
        return 0.0
    lo, hi = max(a, float(x1)), min(b, float(x2))
    return max(0.0, hi - lo) / (b - a)


def _block_spans(blocks, df_det):
    """段ごとの x の範囲を，種名の列の左端の並びから決める

    段の DataFrame をそのまま使ってはいけない: `row` の箱は紙面の全幅にわたるので，
    どの段も同じ範囲になり，右の段の値を左の段にも当ててしまう (2026-09-09)．
    """
    lefts = []
    for blk in blocks:
        names = blk[blk['obj_name'].isin(('sname', 'species_col'))]
        lefts.append(float(names['x1'].min()) if len(names) else float(blk['x1'].min()))
    right = float(df_det['x2'].max())
    out = []
    for i, a in enumerate(lefts):
        b = lefts[i + 1] if i + 1 < len(lefts) else right
        out.append((a, b))
    return out


def columns_from_ink(img, df_det):
    """1 調査区の 2 段組なら，段ごとに `layer` と `col` の検出の箱を作り直す

    Returns:
        (検出, 警告のリスト)．当てはまらなければ元の検出をそのまま返す
    """
    from .blocks import split_blocks
    if df_det is None or len(df_det) == 0 or 'obj_name' not in df_det:
        return df_det, []
    name = df_det['obj_name']
    if (name == 'plot_row').any():          # 地点の列がある = 普通の組成表
        return df_det, []
    names = df_det[name.isin(('sname', 'species_col'))]
    if names.empty:
        return df_det, []
    try:
        blocks = split_blocks(df_det)
    except Exception:                        # noqa: BLE001
        blocks = None
    if blocks is None or len(blocks) < 2:
        return df_det, []
    if int((name == 'col').sum()) > len(blocks):
        return df_det, []                    # 地点の列が育っている表
    dark = ink.binarize(img)
    src = df_det['source_image'].iloc[0] if 'source_image' in df_det else ''
    made = []
    done = []          # 差し替えた段の x の範囲
    warnings = []
    spans = _block_spans(blocks, df_det)
    for i, (blk, (x1, x2)) in enumerate(zip(blocks, spans), 1):
        # `split_blocks` は段ごとの**検出の DataFrame** を返す．x の範囲は
        # **種名の列の左端**から次の段の左端まで (行の箱は紙面の全幅にわたるので
        # そのまま使うと，どの段も同じ範囲になる)
        if blk is None or len(blk) == 0 or x2 - x1 < 10:
            continue
        rows = _rows_of(blk, (x1, x2)) or _rows_of(df_det, (x1, x2))
        if not rows:
            continue
        pitch = float(np.median([b - a for a, b in rows]))
        if not pitch > 0:
            continue
        if len(rows) < MIN_ROWS:
            warnings.append(f'段{i}: 1 調査区の 2 段組だが，行が少ない '
                            f'({len(rows)} 行 < {MIN_ROWS})．階層と被度の列は作らない')
            continue
        got = split_block(dark, x1, x2, rows, pitch)
        if got is None:
            continue
        lay_x, cov_x, right = got
        # **その段に列が無いか，列が階層を指しているとき**だけ差し替える．
        # 列が既に被度を指している段まで触ると，行の判定が動いて本物の行が落ちる
        # (049 は 9 行，054・076 も数行を失った．2026-09-09)
        cols = blk[blk['obj_name'] == 'col']
        if len(cols):
            on_cover = max(_overlap(c, cov_x, right) for c in cols.itertuples())
            on_layer = max(_overlap(c, lay_x, cov_x) for c in cols.itertuples())
            if on_cover >= 0.5 or on_cover >= on_layer:
                continue
        y1 = float(min(a for a, _b in rows))
        y2 = float(max(b for _a, b in rows))
        made.append(dict(obj_name='layer', x1=lay_x, x2=cov_x, y1=y1, y2=y2,
                         confidence=0.99, source_image=src))
        made.append(dict(obj_name='col', x1=cov_x, x2=right, y1=y1, y2=y2,
                         confidence=0.99, source_image=src))
        done.append((x1, x2))
        warnings.append(
            f'段{i}: **1 調査区の 2 段組とみなし，階層と被度の列を黒画素から作った** '
            f'(階層 {lay_x:.0f}-{cov_x:.0f}，被度 {cov_x:.0f}-{right:.0f} px，'
            f'{len(rows)} 行)．段階1で列の中身を目で確かめる')
    if not made:
        return df_det, warnings
    # **差し替えた段の中の箱だけ**を消す．表全体の `col`・`layer` を消すと，
    # 発動しなかった段が列を失う (049・076・043 などで 24〜74 セル減った．2026-09-09)
    mid = (df_det['x1'].astype(float) + df_det['x2'].astype(float)) / 2.0
    inside = pd.Series(False, index=df_det.index)
    for a, b in done:
        inside |= (mid >= a) & (mid < b)
    drop = df_det['obj_name'].isin(('col', 'layer')) & inside
    out = pd.concat([df_det[~drop], pd.DataFrame(made)], ignore_index=True)
    return out, warnings
