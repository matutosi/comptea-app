"""検出器を使わずに組成表全体を見つける (table_find.py)

2026-09-11 ユーザ指示「検出器を使わない組成表全体の検出．各部分は不明でよい」．
物体検出 (YOLO) が落ちる紙面 (学習データが活字 121 件だけで折込は 0 件) のための
別経路．**表のどこに何があるか (部分) は決めない**．紙面のどこに組成表があるか
(箱) だけを返す．

手順は 2026-09-11 の試行で当たったもの (本のページ 10 枚中 9 枚，折込は表の数が
23 枚中 14 枚で一致)．

1. 紙面を**タイルに分けて** OCR する．EasyOCR の `readtext` は既定で長辺を
   2560 px に縮める (`canvas_size`) ので，A0 級の紙面をそのまま渡すと項目名が
   1 個しか読めない (2400 px のタイルに分けると 20 個)．
2. **表頭の項目名** (「通し番号」「Lfd. Nr.」「調査年月日」…) を目印にする．
   `plot_table.match_item` の辞書をそのまま使う．見出し (Trennarten・群集標徴種)
   は 1 つの表に何度も出るので数えない (足すと 14/23 → 10/23 に悪化した)．
3. 近い目印をまとめる (長辺の 1 割以内)．**1 まとまり = 1 表**．
4. まとまりの中心を含む**黒画素の塊** (`split_sheet.blob_boxes`) の外接矩形を
   表の箱にする．項目名は表の左半分にしかないので，目印の外接矩形では値の側が
   入らない．塊が無ければ目印の外接矩形を広げて返す．
5. 目印がほとんど無ければ**横倒し**なので，回して読み直す (s01115_04 の 3 表は
   横倒しで，そのままでは 0 個，時計回りに回すと 48 個)．読んだ位置は元の紙面の
   座標へ戻す．

弱いのは「1 枚に何枚あるか」で，1 表が 2 つに割れる紙面と 2 表が 1 つになる
紙面がある．いまの幾何の切り分け (`split_sheet.find_tables`．23 枚中 21 枚) には
及ばない．**幾何で箱を作り，こちらで検算する**使い方が実がある (箱 70 個のうち
57 個にまとまりがちょうど 1 つ入る)．
"""
import re

import numpy as np
from PIL import Image

from . import ink
from .plot_table import match_item
from .row_kinds import kind_of_text
from .split_sheet import blob_boxes


TILE = 2400             # タイルの一辺 (px)．EasyOCR が縮めない大きさ
OVERLAP = 8             # タイルの重なり (一辺のこの分の 1)
DIST = 0.10             # 目印をまとめる距離 (紙面の長辺に対する比)．
                        # 0.06 → 9/23，0.10 → 14/23，0.13 → 12/23
LEAST = 2               # まとまりに要る項目名の数 (1 個だけの拾いは数えない)
PAD = 40                # 塊が無いときに目印の外接矩形へ足す余白 (px)
MAX_FRAC = 0.85         # 塊が紙面のこの割合を超えたら箱にしない (膨張しすぎ)
ORIENTATIONS = ('up', 'cw', 'ccw')

HEAD = re.compile(r'(群集標徴|群落区分|区分種|随伴種|Begleiter|Kennart|Trennart'
                  r'|Differential|標徴種)', re.I)

# **独文の語彙は目印にしない** (2026-09-11 に測って取り下げた)．「Lfd. Nr.」
# 「Datum」などを足すと，(a) 表の下の注記 (「Datum」「Ort d. Aufn.」を含む) が
# 偽のまとまりになり (s01115_12 は 2 表 → 3 箱)，(b) 独文の項目名が隣の表と
# 縦に連なって 2 つのまとまりを橋渡しする (s01115_09 は 5 表 → 4 箱)．
# 辞書 (`match_item`) だけなら両方とも表の数と一致する


# --- 2. 文字の型で目印を選ぶ --------------------------------------------

def classify(items):
    """読み [(文字列, x, y)] から目印だけを (種類, x, y, 文字列) で返す

    種類は 'once' (出現 1 回の種の見出し) / 'item' (表頭の項目名) / 'head' (見出し)．
    表の位置を決めるのに使うのは 'item' だけ．
    """
    out = []
    for text, x, y in items:
        t = (text or '').strip()
        if not t:
            continue
        if kind_of_text(t) == 'once':
            k = 'once'
        elif match_item(t) is not None:
            k = 'item'
        elif HEAD.search(t):
            k = 'head'
        else:
            continue
        out.append((k, float(x), float(y), t))
    return out


# --- 3. 近い目印をまとめる ----------------------------------------------

def cluster(points, dist, least=1):
    """点をつなげてまとめる．x と y のどちらも `dist` 未満なら同じまとまり

    Returns:
        まとまりのリスト (それぞれ点のリスト)．`least` 個に満たないものは落とす
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if (abs(pts[i][0] - pts[j][0]) < dist
                    and abs(pts[i][1] - pts[j][1]) < dist):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(pts[i])
    return [g for g in groups.values() if len(g) >= least]


# --- 5. 回す・戻す -------------------------------------------------------

_HOW = {'cw': Image.ROTATE_270, 'ccw': Image.ROTATE_90}


def rotate(im, how):
    """'cw' は時計回り，'ccw' は反時計回りに 90 度回す．'up' はそのまま"""
    if how == 'up':
        return im
    return im.transpose(_HOW[how])


def rotate_back(x, y, size, how):
    """回した画像の (x, y) を，元の画像 (大きさ `size` = (幅, 高さ)) の座標へ戻す"""
    w, h = size
    if how == 'up':
        return float(x), float(y)
    if how == 'cw':                 # 元 (x, y) → 回した (h-1-y, x)
        return float(y), float(h - 1 - x)
    if how == 'ccw':                # 元 (x, y) → 回した (y, w-1-x)
        return float(w - 1 - y), float(x)
    raise ValueError(how)


# --- 1. タイルに分けて読む ----------------------------------------------

def _read_tiles(im, reader, tile):
    """画像をタイルに分けて読み，目印をその画像の座標で返す"""
    w, h = im.size
    step = max(1, tile - tile // OVERLAP)
    marks = []
    for x in range(0, w, step):
        for y in range(0, h, step):
            c = im.crop((x, y, min(x + tile, w), min(y + tile, h)))
            if c.width < 20 or c.height < 20:
                continue
            try:
                found = reader.readtext(np.asarray(c.convert('RGB')),
                                        detail=1, paragraph=False)
            except Exception:                   # noqa: BLE001  読めなくても進む
                continue
            items = []
            for pts, text, _conf in found:
                cx = x + float(np.mean([p[0] for p in pts]))
                cy = y + float(np.mean([p[1] for p in pts]))
                items.append((text, cx, cy))
            marks += classify(items)
    return marks


def read_marks(im, reader=None, tile=TILE, least=LEAST, orientations=ORIENTATIONS):
    """目印を読む．そのままで足りなければ回して読み直す

    Returns:
        (目印 [(種類, x, y, 文字列)] を**元の画像の座標**で, 採った向き)
    """
    if reader is None:
        from . import ocr
        reader = ocr.READER
    best, best_how = [], orientations[0]
    for how in orientations:
        img = rotate(im, how)
        raw = _read_tiles(img, reader, tile)
        marks = [(k, *rotate_back(x, y, im.size, how), t) for k, x, y, t in raw]
        n_item = sum(1 for k, *_ in marks if k == 'item')
        if n_item >= least:
            return marks, how
        if n_item > sum(1 for k, *_ in best if k == 'item'):
            best, best_how = marks, how
    return best, best_how


# --- 4. まとまりを含む塊を箱にする ---------------------------------------

def box_for(group, blobs, pad=PAD, max_frac=None, size=None):
    """まとまりの中心を含む塊の外接矩形．無ければ目印の外接矩形を広げる

    Args:
        max_frac: 塊が `size` のこの割合を超えたら使わない (紙面ぜんぶが 1 つの
            塊になることがある)
    """
    xs = [p[0] for p in group]
    ys = [p[1] for p in group]
    mx1, my1, mx2, my2 = min(xs), min(ys), max(xs), max(ys)
    # **目印の外接矩形に触れる塊をすべて合わせる**．中心を含む塊 1 つだけでは，
    # 字の段落ごとに割れた小さな塊を掴む (s01115_12 は幅 104 px の塊になった)
    hit = []
    for b in blobs:
        x1, y1, x2, y2 = (float(v) for v in b)
        if x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2:
            continue
        area = (x2 - x1) * (y2 - y1)
        if max_frac and size and area > max_frac * float(size[0]) * float(size[1]):
            continue
        hit.append((x1, y1, x2, y2))
    if hit:
        return (int(min(b[0] for b in hit)), int(min(b[1] for b in hit)),
                int(max(b[2] for b in hit)), int(max(b[3] for b in hit)))
    x1, y1 = mx1 - pad, my1 - pad
    x2, y2 = mx2 + pad, my2 + pad
    if size:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(size[0]), x2), min(float(size[1]), y2)
    return int(x1), int(y1), int(x2), int(y2)


# --- 通し ----------------------------------------------------------------

def find_tables(im, reader=None, tile=TILE, dist=DIST, least=LEAST, pad=PAD):
    """紙面から組成表の箱 (x1, y1, x2, y2) を見つける (検出器を使わない)

    Args:
        im: 紙面 (PIL の画像)
        reader: OCR の読み手 (`readtext` を持つもの)．無ければ EasyOCR
        dist: 目印をまとめる距離 (紙面の長辺に対する比)
        least: まとまりに要る項目名の数
    Returns:
        箱のリスト (上から順)．見つからなければ空
    """
    Image.MAX_IMAGE_PIXELS = None
    marks, _how = read_marks(im, reader=reader, tile=tile, least=least)
    points = [(x, y) for k, x, y, _t in marks if k == 'item']
    groups = cluster(points, max(im.size) * dist, least=least)
    if not groups:
        return []
    blobs = blob_boxes(ink.binarize(im))
    boxes = [box_for(g, blobs, pad=pad, max_frac=MAX_FRAC, size=im.size)
             for g in groups]
    return sorted(set(boxes), key=lambda b: (b[1], b[0]))
