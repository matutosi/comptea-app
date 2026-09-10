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


SNAP_MIN_BAND = 0.6     # 寄せた結果でも，帯の高さは行の高さのこの倍を下回らせない
                        # (下限が無いと隣り合う境が寄り合って 9〜13 px の帯ができた:
                        # 01_p2 で 2 本，03_p2 で 4 本．2026-09-09)


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
    band = max(2, int(SNAP_MIN_BAND * pitch))
    out = [e[0]]
    for i in range(1, len(e) - 1):
        lo = max(int(out[-1]) + band, int(e[i]) - r, 0)
        hi = min(int(e[i]) + r, int(e[i + 1]) - band, h - 1)
        if hi < lo:
            out.append(max(e[i], out[-1] + band))
            continue
        seg = prof[lo:hi + 1]
        best = int(np.flatnonzero(seg == seg.min())[
            np.abs(np.flatnonzero(seg == seg.min()) + lo - e[i]).argmin()]) + lo
        out.append(float(best))
    out.append(max(e[-1], out[-1] + band))
    return np.maximum.accumulate(np.array(out, dtype=float))


VALUE_SNAP = 0.5        # 値の区切りへ寄せる範囲 (行の高さの倍数)


VALUE_INK_THR = 0.1     # 値の側で「字がある」とみなす黒画素 (行の中央値に対する比)


VALUE_MIN_H = 0.2       # 値の行の高さの下限 (行の高さの倍数)．句読点のかけらを落とす


def value_lines(dark, box, pitch, thr=VALUE_INK_THR, min_h=VALUE_MIN_H):
    """値の側の行 (中心, 上端, 下端) を**黒画素の連なり**から作る

    **OCR の箱では駄目でした** (2026-09-10 に測った)．値は列ごとに数字が並ぶので，
    箱を y で束ねると崩れます (22_p3 では 20 行が 13 行になり，4 行ぶんが 1 つの
    182 px の塊になった)．項目名と違って値は整然と並ぶので，黒画素の投影が
    そのまま行になります (同じ表で高さ 24〜27 px の 20 行が出た)．
    """
    x1, x2 = int(box[0]), int(box[2])
    y1, y2 = int(box[1]), int(box[3])
    prof = dark[y1:y2, x1:x2].sum(axis=1).astype(float)
    if not (prof > 0).any():
        return []
    lo = max(1.0, float(np.median(prof[prof > 0])) * thr)
    floor = max(2.0, float(pitch) * min_h)
    out, start = [], None
    for i, v in enumerate(prof):
        if v > lo:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= floor:
                out.append((y1 + (start + i) / 2.0, float(y1 + start), float(y1 + i)))
            start = None
    if start is not None and len(prof) - start >= floor:
        out.append((y1 + (start + len(prof)) / 2.0, float(y1 + start), float(y2)))
    return out


TALL_ITEM = 2.0         # 項目名の行がこの倍 (行の高さ) より高ければ，2 行がつながったもので
                        # 中心が当てにならない．隣り合う境は動かさない (05_p2 の 92 px の行)


def bands_from_pairs(item_spans, value_spans, edges, pitch=None):
    """項目名の行と値の行を**対応付け**て，値の行を割らない境に置き直す

    項目名の区切りをそのまま使うと，値が 2〜3 行にわたる項目で値の行を割ります．
    寄せるだけでは足りませんでした (05_p2 は 21 本中 14 本が値の行の内側で，
    ±0.5 行の範囲に逃げ場が無い)．

    **値の行を，いちばん近い項目名の行に割り当てます** (2026-09-10 ユーザ指示の
    「項目名の区切りに対応しない値の区切りは落とす」に当たります)．
    境は「前の項目の最後の値の行」と「次の項目の最初の値の行」のあいだに置きます．
    値の行が無い項目 (表題・凡例) では，元の境をそのまま使います．

    Args:
        item_spans: 項目名の行 (中心, 上端, 下端)
        value_spans: 値の側の行 (同じ形)
        edges: いまの境 (長さ len(item_spans)+1)
    """
    n = len(item_spans)
    if n < 2 or len(value_spans) < 2 or len(edges) != n + 1:
        return np.asarray(edges, dtype=float)
    ac = np.array([s[0] for s in item_spans], dtype=float)
    owner = [int(np.abs(ac - v[0]).argmin()) for v in value_spans]
    tall = [pitch is not None and (s[2] - s[1]) > TALL_ITEM * float(pitch)
            for s in item_spans]
    out = [float(edges[0])]
    for i in range(1, n):
        e = float(edges[i])
        if tall[i - 1] or tall[i]:
            out.append(max(e, out[-1]))
            continue
        # **動かすのは，境が値の行の内側にあるときだけ**．外にある境まで置き直すと，
        # 値が 1 行ずつの項目が 1 つの帯にまとまる (05_p2 の調査面積と海抜高．2026-09-10)
        if not any(v[1] < e < v[2] for v in value_spans):
            out.append(max(e, out[-1]))
            continue
        prev = [v for v, o in zip(value_spans, owner) if o == i - 1]
        cur = [v for v, o in zip(value_spans, owner) if o == i]
        # **値が複数行にわたる項目の境だけ**動かす．1 行ずつの項目まで置き直すと，
        # 隣り合う 2 項目が 1 つの帯にまとまる (05_p2 の調査面積と海抜高)
        if (len(prev) > 1 or len(cur) > 1) and prev and cur and prev[-1][2] <= cur[0][1]:
            e = (prev[-1][2] + cur[0][1]) / 2.0
        out.append(max(e, out[-1]))
    out.append(max(float(edges[-1]), out[-1]))
    return np.maximum.accumulate(np.array(out, dtype=float))


def bands_from_value_lines(value_spans, top, bottom):
    """値の側の行 (黒画素の連なり) の**あいだ**を境にして帯を作る (1 行 1 帯)

    2026-09-10 の方針転換: 表頭も組成部と同じく，正確に区切れるなら区切りすぎる側に
    倒す．値の行と行のあいだは字を割らないので正確．項目が複数行にわたるぶんは
    帯が増えるが，項目名の無い帯として段階 3 が合成する．
    """
    if len(value_spans) < 2:
        return None
    cuts = [(a[2] + b[1]) / 2.0 for a, b in zip(value_spans[:-1], value_spans[1:])]
    edges = [float(top)] + [float(c) for c in cuts] + [float(bottom)]
    return np.maximum.accumulate(np.array(edges, dtype=float))


def snap_to_value_lines(edges, spans, pitch, reach=VALUE_SNAP):
    """境を，**値の側の行と行のあいだ**へ寄せる (対応する区切りがあるときだけ)

    項目名の区切りをそのまま使うと，値が 2〜3 行にわたる項目で値の行を割ります
    (22_p3 の「調査年月日」は '83 / 6 / 8 の 3 行で，2 行目と 3 行目のあいだに
    項目名の区切りが落ちていました)．

    **値の側にも同じやり方で区切りを仮に作り，項目名の区切りに対応するものだけを
    使います** (2026-09-10 ユーザ指示)．値の行は項目より多いので，対応しない値の
    区切りは落とします．寄せるのは行の高さの半分まで，順序は保ちます．
    """
    e = [float(v) for v in edges]
    if len(e) < 3 or len(spans) < 2:
        return np.asarray(e, dtype=float)
    cuts = np.array([(a[2] + b[1]) / 2.0 for a, b in zip(spans[:-1], spans[1:])],
                    dtype=float)
    if not len(cuts):
        return np.asarray(e, dtype=float)
    r = float(reach) * float(pitch)
    out = [e[0]]
    for i in range(1, len(e) - 1):
        d = np.abs(cuts - e[i])
        j = int(d.argmin())
        v = float(cuts[j]) if d[j] <= r else e[i]
        out.append(max(v, out[-1]))
    out.append(max(e[-1], out[-1]))
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
        # **値の側の行ごとに区切る** (2026-09-10 ユーザ指示の方針転換: 表頭も組成部と
        # 同じく，正確に区切れるなら区切りすぎる側に倒し，段階 3 で合成する)．
        # 項目名の行から作った帯は，値が 2〜3 行にわたる項目や項目名の印字が値と
        # ずれた表で境が値の行を割る．値の行 (黒画素の連なり) の間なら字を割らない．
        # 項目名の無い帯は段階 3 (`plot_table`) が前の項目の続きとして合成する
        vs = value_lines(dark, (value_x[0], box[1], value_x[1], box[3]), float(pitch))
        if len(vs) >= MIN_LINES:
            edges = bands_from_value_lines(vs, float(edges[0]), float(edges[-1]))
        else:
            edges = bands_from_pairs(spans, vs, edges, pitch=float(pitch))
            edges = snap_to_gap(edges, dark, value_x, float(pitch))
    edges = drop_unvalued(edges, dark, value_x)
    if edges is None or len(edges) < MIN_LINES + 1:
        return None
    return edges
