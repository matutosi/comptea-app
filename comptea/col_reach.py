"""組成部の右端の外にある地点の列を，黒画素から足す (col_reach.py)

`col` の検出が右端まで届かない表がある (2026-09-10 にユーザが目視: kinki_006 の 14 列目，
s01115_08_p2 の 24 列目，04_p2 の 8〜10 列目，15_p3 の 25〜28 列目．他に 09_p1・22_p1・
17_p1)．`axes.locate_edges` は検出と検出の**あいだ**しか内挿しないので，端の列は
丸ごと落ちる．

右端の境から列の間隔ぶんの帯を 1 つずつ見て，**本体にも表頭にも字がある**あいだ足す．
表頭が空の帯は足さない．右端の外に字があっても列でないものが 3 表あり，どれも
常在度の文字のはみ出し (10_p2 の「II(+-2)」，063-1・04_p1 のローマ数字) で表頭が空．
帯の右端は印字の隙間 (col_edges.plot_gaps) に置く (この資料に縦罫線はほぼ無い)．
"""
import numpy as np

from . import ink
from .col_edges import plot_gaps


REACH_MAX = 8            # 足す列の上限 (15_p3 は 4 列)


BODY_MIN = 0.3           # 帯の本体の黒画素 / 既存の列の中央値．これ未満は余白


BODY_MAX = 6.0           # 同上限．これを超えるのは文章や表題 (列ではない)


HEAD_MIN = 0.3           # 帯の表頭の黒画素 / 既存の列の中央値．これ未満は常在度など


ROWS_MIN = 0.3           # 表頭が無い表: 字のある行の割合がこれ未満なら字のはみ出し


FIT = 0.8                # 列の幅のこの割合が限界 (画像の端・次の段) の内側に無ければ足さない


GAP_LO, GAP_HI = 0.5, 1.5    # 帯の右端の隙間を探す範囲 (列の間隔の倍数)


def _row_frac(dark, x1, x2, y_edges):
    """帯の中で，字のある行の割合"""
    rows = [ink.ratio(dark, a, b, x1, x2) > 0 for a, b in zip(y_edges[:-1], y_edges[1:])]
    return float(np.mean(rows)) if rows else 0.0


def reach_right(dark, x_edges, y_edges, y_head=None, x_max=None):
    """右端の外に字のある列を足した境と，足した数を返す

    Args:
        dark: `ink.binarize()` の返り値 (True が黒)
        x_edges: 組成部の列の境 (長さ n+1)
        y_edges: 組成部の行の境 (長さ m+1)
        y_head: 表頭の項目行の y の範囲 (y1, y2)．無ければ None
        x_max: これより右へは足さない (次の段の左端など)．無ければ画像の右端
    Returns:
        (x_edges, n_added)
    """
    xs = [float(v) for v in x_edges]
    if len(xs) < 3 or len(y_edges) < 2:
        return np.asarray(xs, dtype=float), 0
    pitch = float(np.median(np.diff(xs)))
    if not np.isfinite(pitch) or pitch <= 0:
        return np.asarray(xs, dtype=float), 0
    yb1, yb2 = float(y_edges[0]), float(y_edges[-1])
    limit = float(dark.shape[1]) if x_max is None else min(float(x_max), float(dark.shape[1]))
    body = [ink.ratio(dark, yb1, yb2, a, b) for a, b in zip(xs[:-1], xs[1:])]
    base_body = float(np.median(body))
    if base_body <= 0:
        return np.asarray(xs, dtype=float), 0
    yh1 = yh2 = base_head = 0.0
    if y_head is not None:
        yh1, yh2 = float(y_head[0]), float(y_head[1])
        head = [ink.ratio(dark, yh1, yh2, a, b) for a, b in zip(xs[:-1], xs[1:])]
        base_head = float(np.median(head))
        if base_head <= 0:
            y_head = None            # 表頭に字が無い表は本体だけで見る
    added = 0
    while added < REACH_MAX:
        x2 = xs[-1]
        if x2 + pitch * FIT > limit:
            break
        nx = min(limit, x2 + pitch)
        gaps = plot_gaps(dark, (int(x2 + pitch * GAP_LO), int(yb1),
                                int(min(limit, x2 + pitch * GAP_HI)), int(yb2)))
        if gaps:
            nx = float(min(gaps, key=lambda g: abs(g - (x2 + pitch))))
        if nx - x2 < pitch * GAP_LO:
            break
        r = ink.ratio(dark, yb1, yb2, x2, nx)
        if r < base_body * BODY_MIN or r > base_body * BODY_MAX:
            break
        if y_head is not None:
            if ink.ratio(dark, yh1, yh2, x2, nx) < base_head * HEAD_MIN:
                break
        elif _row_frac(dark, x2, nx, y_edges) < ROWS_MIN:
            break
        xs.append(nx)
        added += 1
    return np.asarray(xs, dtype=float), added
