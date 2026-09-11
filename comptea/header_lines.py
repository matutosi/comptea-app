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


def snap_to_gap(edges, dark, value_x, pitch, reach=SNAP_REACH, slope=0.0):
    """境を，値の側の黒画素が**いちばん少ない** y へ寄せる (順序は保つ)

    項目名の直前に置くだけでは，次の項目の値の 1 行目に乗る (値と項目名はほぼ同じ
    高さに組まれる)．本体と違って「黒画素ゼロ」は使えない: 地点が 25 列もあると，
    どの y にもどれかの列の字がある．いちばん少ない所へ寄せる．

    `slope` を渡すと，**その傾きに沿って**数えた投影で寄せる (2026-09-10)．
    帯は傾きに沿って作ってあるのに，寄せる先を水平な投影で測ると，左右の
    谷の中間へ引き戻され，端の列で字を割った (03_p2 の右端)．
    """
    e = [float(v) for v in edges]
    if len(e) < 3:
        return np.array(e)
    x1, x2 = int(value_x[0]), int(value_x[1])
    if slope:
        prof = slanted_profile(dark, x1, x2, slope)
    else:
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


VALUE_GAP_MIN = 0.15    # 連なりの隙間が行の高さのこの倍より近ければ，同じ行としてつなぐ


SLANT_BIN = 16          # 傾きを測るときに列をまとめる幅 (px)
SLANT_MAX = 0.03        # 試す勾配の上限 (約 1.7°)
SLANT_MIN_W = 8.0       # 領域の幅が行の高さのこの倍未満なら傾きを測らない


def slant_profile(clean, pitch, bin_w=SLANT_BIN, max_slope=SLANT_MAX):
    """**傾きに沿って**黒画素を数えた投影と，採った勾配を返す (2026-09-10)

    紙面が 0.5° 傾いているだけで，幅 2000 px では行が 19 px ずれ，行と行の谷が
    埋まります (03_p2 の値の側は 17 行が 2 つの連なりになり，5% 分位でも 39 と
    しきい値 15 を下回らなかった)．「行間の狭いタイプ打ちで行がつながる」と
    見ていたものの正体はこれで，紙面の傾きでした．

    列をまとめた小さな投影を作り，勾配を変えながらずらして足し合わせ，山が
    いちばん鋭くなる (二乗和が最大の) 勾配を採ります．返す投影は領域の**中央**の
    座標系のもので，列ごとの上下は `row_skew` があとで付けます．
    """
    h, w = clean.shape
    if w < max(bin_w * 2.0, float(pitch) * SLANT_MIN_W):
        return clean.sum(axis=1).astype(float), 0.0
    n = max(2, int(np.ceil(w / float(bin_w))))
    xs = np.linspace(0, w, n + 1).astype(int)
    bins = [clean[:, a:b].sum(axis=1).astype(float)
            for a, b in zip(xs[:-1], xs[1:])]
    xc = (xs[:-1] + xs[1:]) / 2.0 - w / 2.0
    drift = int(min(max_slope * w, float(pitch) * 4.0))
    if drift < 1:
        return clean.sum(axis=1).astype(float), 0.0
    best, best_s, best_v = None, 0.0, -1.0
    for d in range(-drift, drift + 1):
        s = d / float(w)
        prof = _shift_sum(bins, xc, h, s)
        v = float((prof ** 2).sum())
        if v > best_v:
            best, best_s, best_v = prof, s, v
    return best, best_s


def _shift_sum(bins, xc, h, s):
    """短冊の投影 `bins` を，勾配 `s` に沿ってずらして足す (領域の中央の座標系)"""
    prof = np.zeros(h, dtype=float)
    for k in range(len(bins)):
        sh = int(round(-s * xc[k]))
        if sh == 0:
            prof += bins[k]
        elif sh > 0:
            prof[sh:] += bins[k][:h - sh]
        else:
            prof[:h + sh] += bins[k][-sh:]
    return prof


def slanted_profile(dark, x1, x2, slope, bin_w=SLANT_BIN):
    """`dark[:, x1:x2]` を，決まった勾配 `slope` に沿って数えた投影 (全高)"""
    sub = dark[:, int(x1):int(x2)]
    h, w = sub.shape
    if w < 1:
        return np.zeros(h, dtype=float)
    n = max(1, int(np.ceil(w / float(bin_w))))
    xs = np.linspace(0, w, n + 1).astype(int)
    bins = [sub[:, a:b].sum(axis=1).astype(float) for a, b in zip(xs[:-1], xs[1:])]
    xc = (xs[:-1] + xs[1:]) / 2.0 - w / 2.0
    return _shift_sum(bins, xc, h, float(slope))


def value_lines(dark, box, pitch, thr=VALUE_INK_THR, min_h=VALUE_MIN_H, info=None,
                gap_min=None):
    """値の側の行 (中心, 上端, 下端) を**黒画素の連なり**から作る

    **OCR の箱では駄目でした** (2026-09-10 に測った)．値は列ごとに数字が並ぶので，
    箱を y で束ねると崩れます (22_p3 では 20 行が 13 行になり，4 行ぶんが 1 つの
    182 px の塊になった)．項目名と違って値は整然と並ぶので，黒画素の投影が
    そのまま行になります (同じ表で高さ 24〜27 px の 20 行が出た)．
    """
    x1, x2 = int(box[0]), int(box[2])
    y1, y2 = int(box[1]), int(box[3])
    if x2 - x1 < 3 or y2 - y1 < 3:
        return []
    # **罫線を消してから投影する** (2026-09-10 ユーザ指摘: kinki_088・063-1 は値の列の
    # 左端に縦罫線があり，投影が途切れずに 9 行が 1 つの帯になった)．組成部と同じ道具
    from . import row_track
    clean = row_track.clean_rules(dark, x1, x2, y1, y2, float(pitch))
    if clean.size == 0:
        return []
    # **傾きに沿って数える**．全幅で真横に数えると，0.5° の傾きでも行が重なる
    prof, slope = slant_profile(clean, float(pitch))
    if info is not None:
        info['slope'] = float(slope)
    if not (prof > 0).any():
        return []
    lo = max(1.0, float(np.median(prof[prof > 0])) * thr)
    floor = max(2.0, float(pitch) * min_h)
    runs = _runs_above(prof, lo, floor)
    # **近すぎる連なりはつなぐ** (2026-09-11 ユーザ目視: 07_p3 の「高木層の高さ」は
    # 値が「−」ばかりで，横棒 (6 px) と数字 (11 px) が 2 px 空いただけで別の行に
    # なっていた)．行と行の隙間は行の高さの 2 割ほどあるので，それより近いものは
    # 同じ行．この表の本物の隙間は 7〜13 px，割れていた所は 2 px
    if gap_min is None:
        gap_min = max(3.0, float(pitch) * VALUE_GAP_MIN)
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < gap_min:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    runs = merged
    # **行の高さの 1.4 倍より高い連なりは，その中で閾値を上げて切り直す** (2026-09-10)．
    # 行間の狭いタイプ打ちでは 25 列の字の上下が重なり，投影が行のあいだで 1 割まで
    # 落ちない (14_p4 は 8 行が 1 帯，03_p2 は 8 行)．連なりの中の山の 2〜5 割まで
    # 閾値を上げ，行の高さに収まる片に分かれたところで採る．分かれなければそのまま
    # **切り直しは，傾きに沿った投影になってから使う** (2026-09-10 ユーザ目視 10 回目:
    # 16_p2 は値の先頭 5 行が 1 帯)．以前 (水平な投影) は切り直した境が字に乗って
    # 字を割る境が 5.4% → 9.3% と増えたので取り下げたが，谷が埋まっていたのは傾きの
    # せいだった．傾きに沿った投影では谷が山の 1〜3 割まで落ちる (16_p2 の 57 列でも
    # 92〜270 / 889)．行の高さの 1.4 倍より高い連なりだけ，谷で切り直す
    out = []
    for a, b in runs:
        parts = [(a, b)]
        if b - a > float(pitch) * TALL_BOX:
            parts = _resplit_tall(prof, a, b, float(pitch), floor)
        for s_, e_ in parts:
            out.append((y1 + (s_ + e_) / 2.0, float(y1 + s_), float(y1 + e_)))
    return out


def _runs_above(prof, lo, floor):
    """`prof` が `lo` を超える区間 [a, b) のうち，長さ `floor` 以上のもの"""
    runs, start = [], None
    for i, v in enumerate(prof):
        if v > lo:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= floor:
                runs.append((start, i))
            start = None
    if start is not None and len(prof) - start >= floor:
        runs.append((start, len(prof)))
    return runs


RESPLIT_FRACS = (0.2, 0.35, 0.5)   # つながった連なりを切り直すときの閾値 (連なりの山に対する比)


def _resplit_tall(prof, a, b, pitch, floor, fracs=RESPLIT_FRACS):
    """連なり [a, b) を，中の閾値を上げて行の高さに収まる片に分ける．分かれなければそのまま"""
    seg = np.asarray(prof[a:b], dtype=float)
    peak = float(seg.max())
    if peak <= 0:
        return [(a, b)]
    for f in fracs:
        sub = _runs_above(seg, peak * f, floor)
        if len(sub) >= 2 and all(e - s <= pitch * TALL_BOX for s, e in sub):
            # 片と片のあいだ (谷) の中央を境にし，両端は元の連なりの端まで
            cuts = [(sub[i][1] + sub[i + 1][0]) // 2 for i in range(len(sub) - 1)]
            bounds = [0] + cuts + [len(seg)]
            return [(a + s, a + e) for s, e in zip(bounds[:-1], bounds[1:])]
    return [(a, b)]


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


OWN_GAP_MIN = 1.0       # 項目名の行をつなぐ幅 (px)．値と違い，隣の項目が 3 px で
                        # 接することがあるので，事実上つながない


OWN_PITCH_LO = 0.4      # 決め直す刻みの下限 (渡された刻みに対する比)．
                        # 下げすぎると 1 行が上下に割れる


def bands_from_own_lines(dark, box, pitch, fallback):
    """項目名の領域 `box` の**自分の行**から帯を作る (2026-09-10 ユーザ指摘 2)

    値の帯をずらして寄せるだけでは，項目名の字を横切る境が残ります．項目名の行の
    数は値と違い (09_p1 は独文 9 行に対し値 12 行)，どう動かしても余る境が字に乗る
    ためです．項目名の側も黒画素の連なりで行を切り，その間を境にします．
    段階 3 (`plot_table`) は帯どうしの縦の重なりで組にするので，数が違ってよい．

    行が 3 つ未満なら `fallback` (値の帯をずらしたもの) をそのまま返します．

    **刻みは項目名自身の行の高さで決め直します** (2026-09-11 ユーザ目視 13 回目:
    「表頭の項目の行区切りが足りない」13 表)．渡される `pitch` は組成部や表頭の
    行の高さで，項目名の行はそれより低いことがあります (16_p2 は 55 px に対し
    項目名は 30 px)．刻みが大きいと短い行が捨てられ (`VALUE_MIN_H`)，隙間も
    つながって (`VALUE_GAP_MIN`) 行が足りません (9 行 → 真値 12〜13 行)．
    1 度切って高さの中央値を採り，それで切り直します．
    """
    # **項目名は値とは独立に切る** (2026-09-11 ユーザ指示: 対応は後処理で)．
    # (1) 「近すぎる連なりはつなぐ」は値のための規則 (「−」と数字が 2 px 空く)．
    #     項目名では隣の項目をつないでしまう (16_p2 の「調査面積 (林縁方位)」と
    #     「方位」は隙間 3 px)．つながない (1 px だけ許す)
    # (2) 傾きに沿った投影の基準は列の中央だが，項目名は左寄せで印字される．
    #     箱を**字のある範囲**に縮めてから投影する (21_p1 は幅 512 px × 傾き
    #     0.023 の半分 ≈ 6 px 境がずれ，「調査番号」の下端を通っていた)
    box = _shrink_to_ink(dark, box)
    lines = value_lines(dark, box, pitch, gap_min=OWN_GAP_MIN)
    if len(lines) < MIN_LINES:
        return np.asarray(fallback, dtype=float)
    own = float(np.median([b - a for _m, a, b in lines]))
    if OWN_PITCH_LO * pitch < own < pitch:
        again = value_lines(dark, box, own, gap_min=OWN_GAP_MIN)
        if len(again) > len(lines):
            lines = again
    return bands_from_value_lines(lines, float(box[1]), float(box[3]))


def _shrink_to_ink(dark, box, pad=4):
    """箱の x を，字のある範囲 (黒画素の 2〜98% 点) に縮める．y は変えない"""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = max(0, x1), min(dark.shape[1], x2)
    y1, y2 = max(0, y1), min(dark.shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 2:
        return box
    col = dark[y1:y2, x1:x2].sum(axis=0).astype(float)
    if col.sum() <= 0:
        return box
    cum = np.cumsum(col) / col.sum()
    lo = int(np.searchsorted(cum, 0.02))
    hi = int(np.searchsorted(cum, 0.98)) + 1
    nx1, nx2 = x1 + max(0, lo - pad), x1 + min(x2 - x1, hi + pad)
    if nx2 - nx1 < 8:
        return box
    return (float(nx1), float(box[1]), float(nx2), float(box[3]))


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


TOP_REACH_ROWS = 8      # 表頭の上端を上へ伸ばす行数の上限
TOP_COVER_MIN = 0.5     # 値の行とみなすのに要る，字のある地点の列の割合
TOP_GAP_MAX = 1.5       # 行と行の間隔がこの倍 (行の高さ) を超えたら，そこで止める


def extend_top(dark, value_x, x_edges, top, pitch, reach=TOP_REACH_ROWS,
               cover=TOP_COVER_MIN, gap=TOP_GAP_MAX):
    """表頭の上端 `top` を，値の行が続く限り上へ伸ばした y を返す (2026-09-10)

    `plot_row`・`header` の検出枠は先頭の項目 (群落記号・通し番号・調査番号・
    調査年月日) まで届かないことがある (10_p1 は 4 項目 6 行が枠の外)．検出枠の上を
    傾きに沿った投影で行に切り，**地点の列の半分以上に字がある行**が 1.5 行以内の
    間隔で続いていれば，そこまで含める．表題や群落名は数列にしかかからない．
    """
    y0 = max(0.0, float(top) - float(pitch) * reach)
    if float(top) - y0 < pitch:
        return float(top)
    lines = value_lines(dark, (value_x[0], y0, value_x[1], float(top) + pitch * 0.5),
                        float(pitch))
    if not lines:
        return float(top)
    # **またぎを見るときは罫線を消す** (2026-09-10)．群落記号の行は枠で組まれており，
    # 枠の横線がすべての列の境をまたぐので，そのままでは凡例と区別できない
    # (10_p1 は「a」「b」の枠が表頭に入らなかった)．線を消せば残るのは短い記号だけ
    from . import row_track
    clean = row_track.clean_rules(dark, int(value_x[0]), int(value_x[1]),
                                  int(y0), int(float(top) + pitch * 0.5), float(pitch))
    cols = [(float(a), float(b)) for a, b in zip(x_edges[:-1], x_edges[1:])]
    if not cols:
        return float(top)
    new_top = float(top)
    last = float(top)
    stop_bottom = None          # 止めた行 (凡例など) の下端．上端はこれより上へ出さない
    for _c, a, b in sorted(lines, key=lambda t: -t[1]):     # 上端の高い順 = 下から上へ
        if a >= float(top):
            continue
        if last - b > pitch * gap:
            break
        have = np.mean([ink.ratio(dark, a, b, xa, xb) > 0 for xa, xb in cols])
        if have < cover:
            stop_bottom = float(b)
            break
        # **字が列の境をまたぐ行は凡例・表題** (15_p5 は地点 4 列で，凡例の文が列の
        # 半分にかかり，値の行に見えた)．値は列の中に収まるので境をまたがない
        if _cross_frac(clean, a - y0, b - y0, x_edges, x0=value_x[0]) >= TOP_CROSS_MAX:
            stop_bottom = float(b)
            break
        new_top = float(a)
        last = float(a)
    if new_top >= float(top):
        return float(top)
    y = new_top - pitch * 0.3
    if stop_bottom is not None:
        # 凡例の字の上に上端を置かない: 凡例の下端と値の行の上端の中間まで
        y = max(y, (stop_bottom + new_top) / 2.0)
    return max(0.0, y)


TOP_CROSS_MAX = 0.4     # 字がまたぐ列の境がこの割合以上なら，値の行ではない


def _cross_frac(dark, y1, y2, x_edges, half=3, x0=0.0):
    """行 `y1`-`y2` で，字が列の境をまたいでいる境の割合 (境の両側 `half` px に字)

    `x0` は `dark` の左端に当たる画像の x (切り出した配列を渡すときに使う)．
    """
    inner = [float(x) - float(x0) for x in x_edges[1:-1]]
    if not inner:
        return 0.0
    band = dark[max(0, int(y1)):max(0, int(y2))]
    if band.size == 0:
        return 0.0
    n = 0
    for x in inner:
        xi = int(x)
        if xi <= 0 or xi >= band.shape[1]:
            continue
        left = band[:, max(0, xi - half):xi]
        right = band[:, xi:xi + half]
        # **同じ画素行で両側に字があること**．帯ぜんたいで見ると，隣り合う列の
        # 値が別々の高さにあるだけで「またいだ」ことになる (10_p1 は 1.00 になった)
        if left.size and right.size and (left.any(axis=1) & right.any(axis=1)).any():
            n += 1
    return n / len(inner)


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


NAME_OFFSET_REACH = 1.0     # 項目名の行と値の行を組にする距離の上限 (行の高さの倍数)


def name_offset(item_spans, value_spans, pitch):
    """項目名の行が値の行からどれだけ上下にずれて印字されているか (中央値，px)

    組成部の「行番号は共有し，y は列ごとに持つ」と同じ考え (2026-09-10)．値の行の
    あいだで切った帯をそのまま項目名に当てると，項目名を割る (項目名は値より
    1 行の半分ほど下に組まれる紙面が多い: 22_p3 は 1 行半)．項目名の行ごとに
    いちばん近い値の行との差を取り，その中央値だけ項目名の側の帯をずらす．
    組が 3 つ未満なら 0．
    """
    if not item_spans or not value_spans or not pitch:
        return 0.0
    vc = np.array([v[0] for v in value_spans], dtype=float)
    ds = []
    for s in item_spans:
        j = int(np.abs(vc - s[0]).argmin())
        d = s[0] - vc[j]
        if abs(d) <= NAME_OFFSET_REACH * float(pitch):
            ds.append(d)
    return float(np.median(ds)) if len(ds) >= 3 else 0.0


def header_bands(img, box, value_x, pitch=None, reader=None, dark=None, info=None):
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
        _vinfo = {} if info is None else info
        vs = value_lines(dark, (value_x[0], box[1], value_x[1], box[3]),
                         float(pitch), info=_vinfo)
        slope = float(_vinfo.get('slope', 0.0) or 0.0)
        if info is not None:
            info['spans'] = spans
            info['values'] = vs
            info['boxes'] = boxes        # 独文と和文を x で分けてずれを取るため
        if len(vs) >= MIN_LINES:
            edges = bands_from_value_lines(vs, float(edges[0]), float(edges[-1]))
            # 境を値の側の黒画素が最少の y へ寄せる (つながった行を切り直した境は
            # 字に乗ることがある．2026-09-10)．傾きに沿って数えた投影で
            edges = snap_to_gap(edges, dark, value_x, float(pitch), slope=slope)
        else:
            edges = bands_from_pairs(spans, vs, edges, pitch=float(pitch))
            edges = snap_to_gap(edges, dark, value_x, float(pitch), slope=slope)
        # **値の行の内側を通る境は，行の外へ出す** (2026-09-11)
        edges = keep_out_of_runs(edges, vs)
    edges = drop_unvalued(edges, dark, value_x)
    if edges is None or len(edges) < MIN_LINES + 1:
        return None
    return edges


SHEAR_MIN = 2.0         # 端の列のずれがこの px 未満なら傾けない


RUN_KEEP = 2            # 値の行の端からこの px 以内なら，行の内側とはみなさない


def keep_out_of_runs(edges, runs, keep=RUN_KEEP):
    """**値の行の内側を通る境を，行の外へ出す** (2026-09-11 ユーザ指摘: kinki_021)

    表頭の帯の境は，値の行 (黒画素の連なり) のあいだに置くのが正しい形です
    (2026-09-10 ユーザ定義: 1 つの文字の途中に境が無い)．上端を伸ばす処理や
    項目名の側の行から来た境が，値の行の真ん中を通ることがあります
    (kinki_021 は通し番号の数字を上下に割っていた)．

    行の内側の境は，近い方の端の外へ出します．出すと隣の境を越えるなら落とします．
    """
    if edges is None or len(edges) < 3 or not runs:
        return edges
    out = [float(edges[0])]
    for e in [float(v) for v in edges[1:-1]]:
        for _c, a, b in runs:
            if a + keep < e < b - keep:
                e = (a - keep) if (e - a) < (b - e) else (b + keep)
                break
        if e > out[-1] + keep:
            out.append(e)
    last = float(edges[-1])
    while len(out) > 1 and out[-1] >= last - keep:
        out.pop()
    out.append(last)
    return np.asarray(out, dtype=float)


def shear_header_values(img, df_loc, dark=None):
    """表頭の値のセルを，**表頭の領域で測った傾き**で列ごとに上下させる (2026-09-10)

    値の帯は `value_lines` が領域の中央の座標で作るので，端の列では傾きのぶん
    ずれる．組成部の傾き (`row_skew`) を掛けると 2〜12 px 合わず，02_p2・05_p1
    では組成部の傾きが「小さい」と判定されて補正が働かない．ここでは表頭の値の
    領域そのもので勾配を測る (`slant_profile`)．

    工程の**最後**に掛けます．途中で掛けると `col_edges.align_header_columns` が
    列ごとに違う y を別の帯と数え，セルが 6 倍に膨れた．

    Returns:
        (直した格子, 警告のリスト)．直す所が無ければ元の格子をそのまま返す
    """
    if df_loc is None or len(df_loc) == 0 or 'obj_name' not in df_loc.columns:
        return df_loc, []
    hv = df_loc['obj_name'] == 'header_value'
    if not hv.any() or 'block' not in df_loc.columns:
        return df_loc, []
    from . import row_track
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if dark is None:
        dark = ink.binarize(img)
    out = df_loc
    warnings = []
    for block, g in df_loc[hv].groupby('block', sort=True):
        x1, x2 = int(g['x1'].min()), int(g['x2'].max())
        y1, y2 = int(g['y1'].min()), int(g['y2'].max())
        cb = comp[comp['block'] == block] if 'block' in comp.columns else comp
        pitch = float(np.median(cb['y2'] - cb['y1'])) if len(cb) else float(
            np.median(g['y2'] - g['y1']))
        if x2 - x1 < 4 or y2 - y1 < 4 or not pitch > 0:
            continue
        clean = row_track.clean_rules(dark, x1, x2, y1, y2, pitch)
        if clean.size == 0:
            continue
        _prof, slope = slant_profile(clean, pitch)
        edge = abs(slope) * (x2 - x1) / 2.0
        if slope == 0.0 or edge < SHEAR_MIN:
            continue
        x0 = (x1 + x2) / 2.0
        mask = hv & (df_loc['block'] == block)
        xc = (df_loc.loc[mask, 'x1'].astype(float) + df_loc.loc[mask, 'x2'].astype(float)) / 2.0
        dy = np.round(slope * (xc - x0))
        out = out.copy() if out is df_loc else out
        out.loc[mask, 'y1'] = out.loc[mask, 'y1'].astype(float) + dy
        out.loc[mask, 'y2'] = out.loc[mask, 'y2'].astype(float) + dy
        deg = float(np.degrees(np.arctan(slope)))
        warnings.append(
            f'段{block}: **表頭の値の帯を，表頭で測った傾き {deg:+.2f}° で列ごとに'
            f'上下させた**(幅 {x2 - x1} px で左右のずれ {2 * edge:.0f} px)．'
            '組成部の傾きとは別に測っている')
    return out, warnings
