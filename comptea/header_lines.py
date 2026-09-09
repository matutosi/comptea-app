"""表頭の項目行を，OCR の検出器 (EasyOCR の CRAFT) の箱から作る (2026-09-09)

表頭の項目名 (独文・和文) は 1 項目 1 行だが，黒画素の投影で行を切る
`locate._header_bands_from_names` はタイプ打ちで崩れる: 行間に谷が無くて 16 行が
1 区間になる (01_p2 は 3 行)，活字の足や句読点が「かけら」として別の行になる，
字数の少ない行が閾値の下に落ちる (先頭の「通し番号」)．
連結成分の外接矩形も不十分だった (点や濁点で行が数珠つなぎになる)．

OCR の**検出器**は文字を語や行にまとめた箱を返すので，箱の y 中心を束ねれば
そのまま行になる．文字認識はしない (`reader.detect`)．代表 6 表で真値と一致した
(01_p1: 17/17，01_p2: 16/16，05_p2: 19/18，017: 8/7+表題)．

行の帯の境は**次の項目名の上端の少し上**に置く (2026-09-09 ユーザ指示)．
1 つの項目の値は 2〜3 行にわたることがあり (「調査年月日」は '83 / 6 / 8 の 3 行)，
中点に置くと 2 行目・3 行目が次の項目の帯に落ちる．先頭・末尾の帯のうち，値側 (地点の列) に
字が無いものは表題・凡例なので落とす．箱が 3 行未満なら None を返し，呼び出し側は
投影の方式へ戻す (複数方式の候補から選ぶ形．2026-09-09 ユーザ提案)．
"""

import numpy as np

from . import ink


COARSE_GAP = 5.0        # 同じ行の箱とみなす y 中心の差 (px)．粗い束ね方


LINE_GAP = 0.5          # 行の高さのこの倍に満たない y の差は，同じ行として束ねる


TALL_BOX = 1.4          # 行の高さのこの倍より高い箱は，2 行以上がつながったものとみなして分ける


LEAD = 0.15             # 境は，次の項目名の上端からこの倍 (行の高さ) だけ上に置く


MIN_LINES = 3           # これより行が少なければ使わない (投影へ戻す)


VALUE_INK_MIN = 0.05    # 帯の値側の黒画素が，帯の中央値のこの倍未満なら「値の無い行」(表題・凡例)


DETECT_KW = dict(min_size=5, text_threshold=0.5, low_text=0.3, link_threshold=0.3,
                 width_ths=0.5, height_ths=0.5)


def _reader():
    """段階 2 と同じ reader を使い回す (作るのに 10 秒ほどかかる)．無ければ None"""
    try:
        from . import ocr
        return ocr.READER
    except Exception:            # noqa: BLE001  easyocr が無い・モデルが無い
        return None


def detect_boxes(img, box, reader=None):
    """`box` = (x1, y1, x2, y2) の中の字の箱を [(x1, y1, x2, y2), ...] で返す (画像の座標)"""
    reader = reader or _reader()
    if reader is None:
        return []
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return []
    crop = np.asarray(img.convert('RGB').crop((x1, y1, x2, y2)))
    horiz, _free = reader.detect(crop, **DETECT_KW)
    out = []
    for b in (horiz[0] if horiz else []):
        bx1, bx2, by1, by2 = (float(v) for v in b)
        out.append((x1 + bx1, y1 + by1, x1 + bx2, y1 + by2))
    return out


def _group(cys, thr):
    lines, cur = [], [cys[0]]
    for a, b in zip(cys[:-1], cys[1:]):
        if b - a > thr:
            lines.append(cur)
            cur = [b]
        else:
            cur.append(b)
    lines.append(cur)
    return [float(np.mean(l)) for l in lines]


def line_centers(boxes, pitch=None, coarse=COARSE_GAP, merge=LINE_GAP):
    """箱を y 中心で行に束ね，行の中心の y を昇順で返す

    閾値は**組成部の行の高さ `pitch` の半分**にする (代表 13 表で真値と一致した条件)．
    箱の高さを閾値にしてはいけない (2026-09-09): 検出器の箱は 2 行にまたがることがあり，
    隣の行まで併合する (01_p1 が 17 行 → 10 行)．箱の中心の間隔から行の高さを推定する手も
    駄目だった: 同じ行の独文と和文が別の塊になり，推定が半分になって行が割れる
    (01_p2 が 35 行)．`pitch` が無いときだけ，粗く束ねてから間隔の中央値で併合する．
    """
    if not boxes:
        return []
    if pitch:
        # **行の高さの 1.4 倍より高い箱は，行の数に分ける**．検出器は行間の狭い 2 行を
        # 1 つの箱にまとめることがあり (01_p1 の「Ort d. Aufn.」と「Größe d. Probefläche」)，
        # その中心は両者のあいだに来て，行が 1 つ減る
        cys = []
        for b in boxes:
            h = b[3] - b[1]
            k = int(round(h / float(pitch))) if h > float(pitch) * TALL_BOX else 1
            if k <= 1:
                cys.append((b[1] + b[3]) / 2.0)
            else:
                step = h / k
                cys += [b[1] + step * (i + 0.5) for i in range(k)]
        return _group(sorted(cys), float(pitch) * merge)
    cys = sorted((b[1] + b[3]) / 2.0 for b in boxes)
    first = _group(cys, coarse)
    if len(first) < 3:
        return first
    est = float(np.median(np.diff(first)))
    return _group(first, est * merge)


def line_spans(boxes, pitch=None, coarse=COARSE_GAP, merge=LINE_GAP):
    """箱を行に束ね，行ごとの (中心, 上端, 下端) を昇順で返す

    `line_centers` と同じ束ね方だが，帯の境を「次の項目名の直前」に置くために
    上端と下端も持つ．
    """
    if not boxes:
        return []
    items = []
    for b in boxes:
        y1, y2 = float(b[1]), float(b[3])
        h = y2 - y1
        k = int(round(h / float(pitch))) if (pitch and h > float(pitch) * TALL_BOX) else 1
        if k <= 1:
            items.append(((y1 + y2) / 2.0, y1, y2))
        else:
            step = h / k
            for i in range(k):
                a, c = y1 + step * i, y1 + step * (i + 1)
                items.append(((a + c) / 2.0, a, c))
    items.sort()
    if pitch:
        thr = float(pitch) * merge
    else:
        first = _group(sorted(i[0] for i in items), coarse)
        thr = (float(np.median(np.diff(first))) * merge if len(first) >= 3 else coarse)
    lines, cur = [], [items[0]]
    for a, b in zip(items[:-1], items[1:]):
        if b[0] - a[0] > thr:
            lines.append(cur)
            cur = [b]
        else:
            cur.append(b)
    lines.append(cur)
    return [(float(np.mean([i[0] for i in g])), float(min(i[1] for i in g)),
             float(max(i[2] for i in g))) for g in lines]


def bands_from_spans(spans, top, bottom, pitch=None, lead=LEAD):
    """行の (中心, 上端, 下端) から帯の境を作る

    境は**次の項目名の上端の少し上**に置く．中点に置いてはいけない (2026-09-09
    ユーザ指示): 1 つの項目の値は 2〜3 行にわたることがあり (「調査年月日」は
    '83 / 6 / 8 の 3 行)，中点だと 2 行目・3 行目が次の項目の帯に落ちる．

        -------------------
        通し番号   AA  AA
                   01  02
        -------------------
        年月日     10  10
                   10  15
        -------------------
    """
    if len(spans) < 2:
        return None
    cys = [s[0] for s in spans]
    p = float(pitch) if pitch else float(np.median(np.diff(cys)))
    gap = lead * p
    edges = [max(float(top), spans[0][1] - gap)]
    for i in range(1, len(spans)):
        e = spans[i][1] - gap                       # 次の項目名の直前
        edges.append(max(e, spans[i - 1][2] + 1))   # 前の行の字は切らない
    edges.append(float(bottom))
    return np.maximum.accumulate(np.array(edges, dtype=float))


def bands_from_centers(centers, top, bottom):
    """行の中心の並びから帯の境を作る (隣の中点．両端は行の間隔の半分)"""
    if len(centers) < 2:
        return None
    c = np.asarray(centers, dtype=float)
    pitch = float(np.median(np.diff(c)))
    edges = [max(float(top), c[0] - pitch / 2)]
    edges += [float(v) for v in (c[:-1] + c[1:]) / 2]
    edges.append(min(float(bottom), c[-1] + pitch / 2))
    return np.maximum.accumulate(np.array(edges, dtype=float))


SNAP_REACH = 0.4        # 境を寄せて探す範囲 (行の高さの倍数)


def snap_to_gap(edges, dark, value_x, pitch, reach=SNAP_REACH):
    """境を，値の側の黒画素が**いちばん少ない** y へ寄せる (順序は保つ)

    項目名の直前に置くだけでは，次の項目の値の 1 行目に乗る (値と項目名はほぼ同じ
    高さに組まれる)．本体と違って「黒画素ゼロ」は使えない: 地点が 25 列もあると，
    どの y にもどれかの列の字がある．いちばん少ない所へ寄せる．
    """
    e = [float(v) for v in edges]
    if len(e) < 3:
        return np.array(e)
    x1, x2 = int(value_x[0]), int(value_x[1])
    prof = dark[:, x1:x2].sum(axis=1).astype(float)
    h = len(prof)
    r = max(1, int(reach * pitch))
    out = [e[0]]
    for i in range(1, len(e) - 1):
        lo = max(int(out[-1]) + 1, int(e[i]) - r, 0)
        hi = min(int(e[i]) + r, int(e[i + 1]) - 1, h - 1)
        if hi < lo:
            out.append(max(e[i], out[-1] + 1))
            continue
        seg = prof[lo:hi + 1]
        best = int(np.flatnonzero(seg == seg.min())[
            np.abs(np.flatnonzero(seg == seg.min()) + lo - e[i]).argmin()]) + lo
        out.append(float(best))
    out.append(e[-1])
    return np.maximum.accumulate(np.array(out, dtype=float))


def drop_unvalued(edges, dark, value_x, min_ratio=VALUE_INK_MIN):
    """先頭・末尾の帯のうち，値側に字の無いもの (表題・凡例) を落とす"""
    if edges is None or len(edges) < 3:
        return edges
    x1, x2 = int(value_x[0]), int(value_x[1])
    prof = dark[:, x1:x2].sum(axis=1).astype(float)
    inks = [float(prof[int(a):int(b)].sum()) for a, b in zip(edges[:-1], edges[1:])]
    pos = [v for v in inks if v > 0]
    if not pos:
        return edges
    thr = float(np.median(pos)) * min_ratio
    lo, hi = 0, len(inks)
    while lo < hi - 1 and inks[lo] < thr:
        lo += 1
    while hi > lo + 1 and inks[hi - 1] < thr:
        hi -= 1
    return edges[lo:hi + 1]


def header_bands(img, box, value_x, pitch=None, reader=None, dark=None):
    """表頭の項目名の領域 `box` から帯の境を作る．作れなければ None

    Args:
        box: 項目名の領域 (x1, y1, x2, y2)．y1 は表頭の枠の 1 行上まで広げて渡す
        value_x: 値側 (地点の列) の x の範囲 (x1, x2)．表題・凡例の帯を落とすのに使う
        pitch: 組成部の行の高さ (px)．行を束ねる閾値 (この半分) に使う
    """
    boxes = detect_boxes(img, box, reader=reader)
    spans = line_spans(boxes, pitch=pitch)
    if len(spans) < MIN_LINES:
        return None
    edges = bands_from_spans(spans, box[1], box[3], pitch=pitch)
    if dark is None:
        dark = ink.binarize(img)
    if edges is not None and pitch:
        edges = snap_to_gap(edges, dark, value_x, float(pitch))
    edges = drop_unvalued(edges, dark, value_x)
    if edges is None or len(edges) < MIN_LINES + 1:
        return None
    return edges
