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


# **左側は隙間を探す範囲を狭くする** (2026-09-10)．右と同じ 0.5〜1.5 列ぶんで
# 探すと，帯が**項目名の領域**まで届くことがある (10_p2)．表頭のその x には
# 独文・和文の項目名がぎっしりあるので，「表頭が空なら止まる」の歯止めが効かず，
# 階層の列ごと飲み込んで階層が消えた (階層そのものの表頭比は 0.04)．
GAP_LO_L, GAP_HI_L = 0.75, 1.25


# **左側は本体のしきい値も上げる** (2026-09-10)．左端の外に本当に地点の列が
# ある表 (040・067・03_p1・09_p5・17_p1・20_p3) では，帯の黒画素は隣の列の
# 0.88〜2.03 倍ある．10_p2 が足そうとした帯は 0.51 倍で，これは列ではなく
# 余白と項目名の端．右側の 0.3 では通ってしまう
BODY_MIN_L = 0.7


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


def reach_left(dark, x_edges, y_edges, y_head=None, x_min=None):
    """**左端の外**に字のある列を足した境と，足した数を返す (2026-09-10)

    09_p5 は種名の列と組成部のあいだが 317 px あき，そこに**階層の列と組成の
    1 列目**が両方入っていた (ユーザ目視)．`reach_right` と同じ考えで，左へ
    1 列ずつ見て，本体にも表頭にも字のある帯だけ足します．

    **階層の列を巻き込まない**のは表頭の条件のおかげです．階層の列は本体には
    字があるが表頭は空なので，そこで止まります (右端の常在度と同じ形)．

    Args:
        x_min: これより左へは足さない (種名の列の右端など)．無ければ画像の左端
    """
    xs = [float(v) for v in x_edges]
    if len(xs) < 3 or len(y_edges) < 2 or y_head is None:
        # **表頭の範囲が分からないときは左へ伸ばさない** (2026-09-10)．右端では
        # 「字のある行の割合」で代われるが，左隣は階層の列で，記号がほぼ全行に
        # あるのでその物差しでは見分けられない．10_p2 は `plot_row` が段に
        # 残らず，階層の列を 1 列目として飲み込んで階層が消えた
        return np.asarray(xs, dtype=float), 0
    pitch = float(np.median(np.diff(xs)))
    if not np.isfinite(pitch) or pitch <= 0:
        return np.asarray(xs, dtype=float), 0
    yb1, yb2 = float(y_edges[0]), float(y_edges[-1])
    limit = 0.0 if x_min is None else max(0.0, float(x_min))
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
            return np.asarray(xs, dtype=float), 0
    added = 0
    while added < REACH_MAX:
        x1 = xs[0]
        if x1 - pitch * FIT < limit:
            break
        nx = max(limit, x1 - pitch)
        gaps = plot_gaps(dark, (int(max(limit, x1 - pitch * GAP_HI_L)), int(yb1),
                                int(x1 - pitch * GAP_LO_L), int(yb2)))
        if gaps:
            nx = float(min(gaps, key=lambda g: abs(g - (x1 - pitch))))
        if x1 - nx < pitch * GAP_LO_L:
            break
        r = ink.ratio(dark, yb1, yb2, nx, x1)
        if r < base_body * BODY_MIN_L or r > base_body * BODY_MAX:
            break
        if ink.ratio(dark, yh1, yh2, nx, x1) < base_head * HEAD_MIN:
            break
        xs.insert(0, nx)
        added += 1
    return np.asarray(xs, dtype=float), added
