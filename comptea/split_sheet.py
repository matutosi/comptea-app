"""大きな折り込みのシートを，表ごとの画像に切り分ける

`s01115`(日本植生誌 近畿 付表)のような **A0 級の折り込み1枚**には，
表が2つ3つ載っている．1枚のまま検出にかけると学習時と縮尺が合わず，
行がほとんど取れない(2026-09-02 に測った．9344x12873 の1枚で `row` は 0 件)．

切り分けは**空白の帯**で行う．s01115_01 で実測すると，
1つの表の**中**にできる空白の帯は

    縦  最大 2.5% of W (地点の列のあいだ)
    横  最大 1.5% of H (表頭と本体のあいだ)

なのに対し，シートの中で表と表を**隔てる**帯は 14% / 7.6% あり，はっきり分かれる．
既定のしきい値(4% / 3%)はこの間に置いてある．

**縦に切ってから，その中を横に切る**(この順でしか切らない)．
逆にすると，種名の列と組成部のあいだの空きで1つの表を割ってしまう．
縦の切れ目はシート全体で見るので，表ごとの内部の空きでは切れない．
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

from . import ink                                      # noqa: E402

Image.MAX_IMAGE_PIXELS = None                   # 折り込みは1億画素を超える

# **しきい値は正解と突き合わせて決めた**(2026-09-02)．
# 隅の札に，そのシートに載っている表の番号が並んでいる(「Tab.4, 6, 7.」)．
# 23 枚を読んで正解(68 表)を作り，`yolo/truth/s01115_tables.tsv` に置いた．
#
#   幅/高さのしきい値   表の合計   23 枚中の一致   不足   過剰
#   4% / 3%(前)          42/68        10          -26      0
#   2% / 2%              55/68        15          -14      1
#   1% / 1%              63/68        17           -7      2   ← これ
#
# 空白の割合で判定する版も試したが**同じ結果**だった．
# ここで効くのは統計量ではなく，しきい値の側．
# 1% はシートの幅・高さに対する比なので，表の中にできる空白
# (s01115_01 で実測すると幅の 2.5%・高さの 1.5% だが，それは
#  切り出したあとの表に対する比で，シート全体に対しては 1% 未満)
# では切れない．
MIN_GUTTER = 0.01       # 縦に切る空白帯の幅．画像の幅に対する比
MIN_GAP = 0.01          # 横に切る空白帯の高さ．画像の高さに対する比
MIN_BAND_PX = 120       # 比で足りないほど小さい画像のための下限(px)
MARGIN = 40             # 切り出しに付ける余白(px)
ROT_MIN_SIDE = 1500     # これより小さい切れ端は，向きを判定しない
ROT_MAX_RATIO = 0.2     # 横の帯 / 縦の帯 がこれ未満なら横倒し
ROT_BLANK_RATIO = 0.995 # 「空白」とみなす，黒でない画素の割合
ROT_BAND_MIN = 3        # 帯とみなす最小の幅(px)
MIN_AREA = 0.01         # これ未満の切れ端は表とみなさない(シート全体に対する面積比)
MAX_SPLIT_ROUNDS = 6    # 切り分けを繰り返す上限
BLANK_TOL_MIN = 3       # 「空白」とみなす黒画素の数の下限
BLANK_TOL_DIV = 1000    # 帯を横切る長さの何分の1まで汚れを許すか

TRAIN_LONG = 3300       # 学習画像の長辺の中央値(88 枚)
TRAIN_LONG_MAX = 3546   # 学習画像の長辺の最大．ここまでは学習時の値をそのまま使う
TRAIN_IMGSZ = 1280      # 学習時の imgsz(正方のレターボックス)
IMGSZ_MIN = 1280        # これより下げても良いことがないので下限にする
# 上限は**GPU が通る大きさ**で決めてある(2026-09-02 に 4288・4992・5632 を試し，
# どれも動いた)．長辺 12866 px までは頭打ちにならないので，手元の折り込み
# (最も高い表で 12703 px)はすべて学習時と同じ縮尺で回せる．
# なお **`imgsz` を上げても，地点の多い横長の表は行が取れない**．
# そちらは縮尺ではなく行の横幅の問題で，`strips.py` が受け持つ
IMGSZ_MAX = 4992


def auto_imgsz(width, height, step=32):
    """学習時と同じ縮尺になる `imgsz` を返す

    学習は長辺 3300 px の画像を `imgsz=1280` で行った(実効の縮尺 0.388)．
    大きな画像を既定の 1280 のまま渡すと字が潰れて行が取れないので，
    **長辺に比例させて** `imgsz` を上げ，縮尺を学習時に合わせる．

    返り値は 32 の倍数．上限で頭打ちにしたかどうかは `imgsz_capped()` で分かる．
    """
    long = max(width, height)
    # **学習画像と同じ大きさなら，学習時の値をそのまま使う**．
    # 比で計算すると 3365 px の画像が 1312 になり，32 画素の違いで
    # `col` が1本も取れなくなることがある(kinki_047．2026-09-02)．
    # 学習画像の長辺は 3123-3546 に収まっているので，その中は動かさない
    if long <= TRAIN_LONG_MAX:
        return TRAIN_IMGSZ
    want = long * TRAIN_IMGSZ / TRAIN_LONG
    size = int(round(want / step)) * step
    return max(IMGSZ_MIN, min(IMGSZ_MAX, size))


def imgsz_capped(width, height):
    """`auto_imgsz()` が上限で頭打ちになったか(縮尺が学習時より小さい)"""
    return max(width, height) * TRAIN_IMGSZ / TRAIN_LONG > IMGSZ_MAX


def load_page(path, page=0, dpi=300):
    """PDF か画像を開いて濃淡の PIL 画像を返す

    スキャンした PDF は，1ページに1枚の画像が貼ってあるだけのことが多い．
    描画し直すと再標本化で字が甘くなるので，**貼ってある画像をそのまま取り出す**．
    取り出せないとき(複数枚が貼ってある，図形で描いてある)だけ `dpi` で描画する．
    """
    if not path.lower().endswith('.pdf'):
        return Image.open(path).convert('L')
    import io
    import fitz                                 # PyMuPDF
    doc = fitz.open(path)
    pg = doc[page]
    imgs = pg.get_images(full=True)
    if len(imgs) == 1:
        info = doc.extract_image(imgs[0][0])
        return Image.open(io.BytesIO(info['image'])).convert('L')
    pix = pg.get_pixmap(dpi=dpi)
    return Image.frombytes('L' if pix.n == 1 else 'RGB',
                           (pix.width, pix.height), pix.samples).convert('L')


def _tol(cross):
    """「空白」とみなす黒画素の数．帯を横切る長さ `cross` に比例させる

    原寸のスキャンには点状の汚れが散っているので，**0 を求めると帯が繋がらない**．
    s01115_01 の仕切り(幅 1312 px)は，0 を求めると最長 50 px に切れてしまい，
    6 まで許すと 1294 px の1本になる(6〜25 のあいだは結果が変わらない)．
    """
    return max(BLANK_TOL_MIN, cross // BLANK_TOL_DIV)


def _blank_bands(profile, min_len, tol):
    """黒画素が `tol` 以下の区間のうち，長さが `min_len` 以上のものを返す"""
    out, start = [], None
    for i, v in enumerate(profile):
        if v <= tol:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(profile) - start >= min_len:
        out.append((start, len(profile)))
    return out


def _cut_points(profile, min_len, tol):
    """空白の帯の**中央**を切れ目にする(両端の帯は切れ目にしない)"""
    bands = _blank_bands(profile, min_len, tol)
    n = len(profile)
    return [(a + b) // 2 for a, b in bands if a > 0 and b < n]


def _bbox(dark):
    """黒画素の外形 (x1, y1, x2, y2)．黒が無ければ None

    点状の汚れで外形が広がらないよう，ここでも `_tol()` まで見逃す．
    """
    h, w = dark.shape
    rows = np.nonzero(dark.sum(axis=1) > _tol(w))[0]
    cols = np.nonzero(dark.sum(axis=0) > _tol(h))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _line_bands(profile, ratio=ROT_BLANK_RATIO, min_w=ROT_BAND_MIN):
    """「空白の割合」が `ratio` 以上つづく帯の数(= 字の行のあいだの数)"""
    n, start = 0, None
    for i, v in enumerate(profile):
        if v >= ratio:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_w:
                n += 1
            start = None
    return n


def looks_rotated(dark, min_side=ROT_MIN_SIDE, max_ratio=ROT_MAX_RATIO):
    """表が 90 度回して組まれているか

    `s01115_04` の 3 表(Tab.12・13・14)は，**紙面の都合で横倒しに組んである**．
    種名も表頭の項目名も下から上へ読む．comptea は「種 = 行，地点 = 列」を
    前提にしているので，そのままでは扱えない(Tab.12 は 46 地点の表から
    102 列の格子ができていた)．

    見分け方は**字の行が走る向き**．正しい向きなら，行と行のあいだの
    空白の帯は横向きに数多く並ぶ．横倒しなら縦向きに並ぶ．
    実測(2026-09-02，65 表)．

        横倒しの 3 表   横の帯 / 縦の帯 = 0.016 - 0.075
        正しい向き      0.49 以上

    **小さすぎる切れ端は見ない**．見出しだけの帯(高さ 666 px など)は
    字の行が数本しかなく，比が当てにならない．
    """
    h, w = dark.shape
    if min(h, w) < min_side:
        return False
    hb = _line_bands((~dark).mean(axis=1))
    vb = _line_bands((~dark).mean(axis=0))
    return vb > 0 and hb / vb < max_ratio


def _split_box(dark, box, gutter, gap):
    """箱を1回だけ切る(縦に切ってから，その中を横に)．返すのは箱の並び"""
    x1, y1, x2, y2 = box
    sub = dark[y1:y2, x1:x2]
    if not sub.any():
        return [box]
    xs = [0] + _cut_points(sub.sum(axis=0), gutter, _tol(y2 - y1)) + [x2 - x1]
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        part = sub[:, a:b]
        if not part.any():
            continue
        ys = [0] + _cut_points(part.sum(axis=1), gap, _tol(b - a)) + [y2 - y1]
        for c, d in zip(ys[:-1], ys[1:]):
            if part[c:d].any():
                out.append((x1 + a, y1 + c, x1 + b, y1 + d))
    return out or [box]


# ---- 縮小した塊で切り分ける (2026-09-10 ユーザ提案) ----------------------------
#
# 空白の帯は**シート全体を貫く**必要があるので，表が階段状に置かれた紙面
# (s01115_09 の 5 表) では通らず，注記が右へ伸びているだけで切れなくなる．
# **縮めてしまえば，表 1 つが 1 つの塊になる**．
#
# 要は縮め方．**ブロックの中に黒が 1 画素でもあれば黒**として縮める
# (平均で縮めると「・」だけの組成部が薄まって消え，塊が表題や種名だけになる)．
# **膨張はしない**．「1 画素でもあれば黒」の縮小そのものが膨張として働くので，
# そのうえ膨らませると紙面がぜんぶ 1 つの塊になる (23 枚で測ると，膨張 2 で
# 塊は 74 → 144 と暴れる)．23 枚の掃引で最も真値に近いのは 1/8・黒 4・膨張なし．
#
#   縮小  黒  膨張   塊の合計 (真値 68)   枚の一致 (23 枚)
#   1/8    4   なし          74               18   ← これ
#   1/8    1   なし          79               14
#   1/16   4    3           107                8
#   1/32   4    3            71               14
#
# ただし**塊だけでは帯に勝てません** (帯は 64/68・17 枚)．塊は「1 つの表が 2 つに
# 割れる」紙面 (05・01) を作ってしまうので，**帯で切った箱の中だけ**に当てる．
BLOB_SCALE = 8          # 縮小の率 (300 dpi のスキャンで，字 1 つが 1 画素ほどになる)
BLOB_INK = 4            # ブロックの中に黒がこれだけあれば黒とみなす (点状の汚れを落とす)
BLOB_DILATE = 1         # 縮小後の膨張 (1 = 膨張しない)
BLOB_MIN_AREA = 0.01    # 塊の面積 / 見ている範囲．これ未満は札や折り目の汚れ
BLOB_MAX = 4            # 塊がこれより多ければ使わない (字の段落ごとに割れている)
BLOB_MIN_PART = 0.05    # 塊 1 つが範囲のこの割合を占めること (小さい断片では切らない)．
                        # 0.10 だと s01115_09 の Tab.51 (紙面の 5.5%) が切り離せない
BLOB_MAX_FILL = 0.85    # 塊の合計が範囲のこの割合を超えたら切らない (すでに 1 つの表)
BLOB_STOP_AREA = 0.001  # 下へ伸ばすとき，止まる目印にする塊の面積の下限 (注記の段落も拾う)
BLOB_REACH_GAP = 0.03   # 下へ伸ばすとき，塊の高さのこの割合より離れた塊で止める
                        # (注記は表のすぐ下に続くが，次の表は離れて置かれる)


def shrink_ink(dark, scale=None, min_ink=None):
    """`scale` 角のブロックに黒が `min_ink` 個以上あれば黒，として縮める"""
    scale = BLOB_SCALE if scale is None else scale
    min_ink = BLOB_INK if min_ink is None else min_ink
    h, w = dark.shape
    hh, ww = (h // scale) * scale, (w // scale) * scale
    if hh < scale or ww < scale:
        return np.zeros((0, 0), dtype=np.uint8)
    blk = dark[:hh, :ww].reshape(hh // scale, scale, ww // scale, scale)
    return (blk.sum(axis=(1, 3)) >= min_ink).astype(np.uint8)


def blob_boxes(dark, scale=None, min_ink=None, dilate=None, min_area=None):
    """縮小した紙面の黒画素の塊 (連結成分) の外接矩形を，元の座標で返す"""
    scale = BLOB_SCALE if scale is None else scale
    dilate = BLOB_DILATE if dilate is None else dilate
    min_area = BLOB_MIN_AREA if min_area is None else min_area
    small = shrink_ink(dark, scale, min_ink)
    if small.size == 0:
        return []
    if dilate >= 2:
        small = cv2.dilate(small, np.ones((dilate, dilate), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(small, connectivity=8)
    area = small.shape[0] * small.shape[1]
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w * h < area * min_area:
            continue
        out.append((int(x * scale), int(y * scale),
                    int((x + w) * scale), int((y + h) * scale)))
    return sorted(out, key=lambda b: (b[0], b[1]))


def split_by_blobs(dark, box, max_blobs=None, min_part=None, max_fill=None, **kw):
    """1 つの箱を，中の塊で切り直す (切れなければ元の箱 1 つを返す)

    空白の帯は**紙面を貫く**必要があるので，表が階段状に置かれた紙面では通りません
    (s01115_09 は Tab.50 の注記が右へ伸びているだけで縦の帯が消え，5 表が 3 つに
    しか切れませんでした)．塊は伸びた注記ごと 1 つの表にまとめるので，そこで切れます．

    **切るのは，はっきり分かれているときだけ**にします．
      - 塊が 2〜4 個 (5 個以上は字の段落ごとに割れている)
      - どの塊も範囲の 1 割以上 (小さい断片は切り離さない)
      - 塊の合計が範囲の 85% 以下 (覆っていれば，もともと 1 つの表)
    """
    max_blobs = BLOB_MAX if max_blobs is None else max_blobs
    min_part = BLOB_MIN_PART if min_part is None else min_part
    max_fill = BLOB_MAX_FILL if max_fill is None else max_fill
    x1, y1, x2, y2 = box
    sub = dark[y1:y2, x1:x2]
    if sub.size == 0:
        return [box]
    blobs = blob_boxes(sub, **kw)
    if not 2 <= len(blobs) <= max_blobs:
        return [box]
    area = (x2 - x1) * (y2 - y1)
    sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in blobs]
    if min(sizes) < area * min_part or sum(sizes) > area * max_fill:
        return [box]
    stops = blob_boxes(sub, min_area=BLOB_STOP_AREA, **kw)
    blobs = _reach_down(blobs, y2 - y1, stops)
    return [(x1 + a, y1 + c, x1 + b, y1 + d) for a, c, b, d in blobs]


def _reach_down(blobs, height, stops=None, gap=BLOB_REACH_GAP):
    """塊の下端を，**すぐ下に続く小さい塊のあいだ**だけ伸ばす

    **表の下の注記を切り落とさないため**です．注記は表と離れると別の塊になりますが
    (段落ごとに分かれるので，表の塊としては小さすぎて数に入りません)，そこには
    調査地・調査年月日・出典が書いてあり，表頭に無い地点の情報をここから採ります．

    **範囲の下端まで伸ばしてはいけません**．次の表の表題を巻き込みます
    (s01115_23_p2 が Tab.149 の表題を取り込み，字を割る割合が 74% になった)．
    真下の塊を 1 つずつ辿り，**間隔が塊の高さの 3% を超えたら止めます**．
    注記は表のすぐ下に続きますが，次の表は離れて置かれます．
    横に広げないのは，隣の表を巻き込むためです．
    """
    cand = list(stops if stops is not None else blobs)
    out = []
    for x1, y1, x2, y2 in blobs:
        bottom = y2
        limit = max(10, (y2 - y1) * gap)
        for _ in range(MAX_SPLIT_ROUNDS):
            below = [b for b in cand
                     if b[3] > bottom and min(x2, b[2]) - max(x1, b[0]) > 0]
            if not below:
                break
            nb = min(below, key=lambda b: b[1])
            if nb[1] - bottom > limit:
                break
            bottom = nb[3]
        out.append((x1, y1, x2, int(min(bottom, height))))
    return out


def find_tables(dark, min_gutter=MIN_GUTTER, min_gap=MIN_GAP,
                min_area=MIN_AREA, margin=MARGIN, use_blob=True):
    """表ごとの箱を返す

    並びは**左の段から，段の中は上から**(縦組みのシートに合わせた列優先)．
    `dark` は `ink.binarize()` の返り値(True が黒)．

    **切れなくなるまで繰り返す**(2026-09-03 ユーザ提案)．
    紙面は必ずしも「縦に切ってから横」の1回では分かれない．
    s01115_09 は 5 表が階段状に置かれており，上下で縦の仕切りの位置が違う
    (上は x4600，下は x3750)．シート全体を貫く空白の縦帯が無いので，
    1回では上下 2 つにしか切れなかった．横に切ってからもう一度縦に切れば
    分かれる．

    **しきい値はシート全体に対する比のまま**にする．切り出した箱の大きさで
    決め直すと，箱が小さくなるほどしきい値が下がり，1つの表を
    種名の列と組成部の隙間で割ってしまう．
    """
    h, w = dark.shape
    gutter = max(int(w * min_gutter), MIN_BAND_PX)
    gap = max(int(h * min_gap), MIN_BAND_PX)

    boxes = [(0, 0, w, h)]
    for _ in range(MAX_SPLIT_ROUNDS):
        out = []
        for box in boxes:
            out.extend(_split_box(dark, box, gutter, gap))
        if len(out) == len(boxes):
            break
        boxes = out

    # **帯で切れなかった箱を，縮小した塊で切る** (2026-09-10 ユーザ提案)．
    # 帯のあとに置くのが要で，先に当てると 1 つの表が 2 つに割れます (05・01)．
    if use_blob:
        for _ in range(MAX_SPLIT_ROUNDS):
            out = []
            for box in boxes:
                out.extend(split_by_blobs(dark, box))
            if len(out) == len(boxes):
                break
            boxes = out

    result = []
    for x1, y1, x2, y2 in boxes:
        bb = _bbox(dark[y1:y2, x1:x2])
        if bb is None:
            continue
        bx1, by1, bx2, by2 = bb
        box = (max(0, x1 + bx1 - margin), max(0, y1 + by1 - margin),
               min(w, x1 + bx2 + margin), min(h, y1 + by2 + margin))
        if (box[2] - box[0]) * (box[3] - box[1]) < w * h * min_area:
            continue                            # 見出しの札や折り目の汚れ
        result.append(box)
    return sorted(result, key=lambda b: (b[0], b[1]))


NOTE_BLANK = 0.002      # 行の黒画素が幅のこの割合未満なら空白の行


def note_boxes(dark, boxes, gap=BLOB_REACH_GAP, blank=NOTE_BLANK):
    """表の箱ごとに，**すぐ下に続く注記**の箱を返す (無ければ None)

    表の下には「出現 1 回の種」と調査地・調査年月日・出典が書かれており，
    **表頭に無い地点の情報の出どころ**です (2026-09-11 ユーザ指摘: 02_p1・02_p2 で
    欠落)．帯で切った箱では，これが丸ごと落ちていました．

    **表の画像には含めません**．含めると画像が高くなり，検出器の入力の縮尺が
    変わって列や階層の枠が動きます (09_p5 は組成の 1 列目が階層の枠に食われた)．
    注記は別の画像として書き出し，読む側 (`once_page.parse_site_notes`) が使います．

    箱の x の幅で黒画素の並びを見て，空白の行が箱の高さの `gap` ぶん続くまで
    下へ辿ります．**他の箱の上端は越えません**．

    Returns:
        [(x1, y1, x2, y2) または None] を `boxes` と同じ並びで
    """
    h, w = dark.shape
    out = []
    for x1, y1, x2, y2 in boxes:
        limit_y = min([b[1] for b in boxes
                       if b[1] >= y2 and min(x2, b[2]) - max(x1, b[0]) > 0] + [h])
        room = max(10, int((y2 - y1) * gap))
        prof = dark[:, int(x1):int(x2)].sum(axis=1)
        thr = max(1.0, (x2 - x1) * blank)
        bottom, run, y = y2, 0, int(y2)
        while y < int(limit_y):
            if prof[y] < thr:
                run += 1
                if run > room:
                    break
            else:
                run = 0
                bottom = y + 1
            y += 1
        top = None
        for y in range(int(y2), int(bottom)):    # 注記の最初の字まで詰める
            if prof[y] >= thr:
                top = y
                break
        out.append((x1, top, x2, int(bottom)) if top is not None and bottom - top > 10
                   else None)
    return out


def check_rotation(image):
    """紙面が 90 度回して組まれていれば警告を返す (対策 H の入口．2026-09-10)

    折込の切り出しでは `cut_table` が向きを見て回すが，本のページ (s01114) は
    そのまま検出にかけていた．kinki_014 は横倒しで，種名が下から上へ読む向き
    (ユーザ指摘 22・42)．157 枚で回転と判定されるのはこの 1 枚だけ (横の帯 3 本・
    縦の帯 39 本．他は比 0.32 以上)．**ここでは回さず，知らせるだけ**
    (回して検出し直す段は，この検査で対象が 1 枚と分かってから入れる)．

    Args:
        image: PIL の画像か，そのパス
    Returns:
        警告のリスト (回っていなければ空)
    """
    im = image if isinstance(image, Image.Image) else Image.open(image)
    dark = ink.binarize(im)
    if not looks_rotated(dark):
        return []
    hb = _line_bands((~dark).mean(axis=1))
    vb = _line_bands((~dark).mean(axis=0))
    return [f'**この紙面は 90 度回して組まれている**(字の行の帯が横 {hb} 本・縦 {vb} 本)．'
            '種名が下から上へ読む向きで，このままでは行と列が入れ替わった格子になる．'
            '画像を時計回りに 90 度回してからかけ直す(対策 H)．']


def cut_table(im, box):
    """紙面から表を 1 つ切り出す(**横倒しなら 90 度回して正す**)

    切り出しはここに 1 つだけ置く(2026-09-07)．前は切り出しと回転が
    3 か所に写してあり，**アプリだけが回していなかった**．横倒しの表を
    回さずに渡すと，行と列が入れ替わったまま格子ができる
    (Tab.12 は 46 地点の表から 102 列になった)．

    Returns:
        (切り出した画像, 回したかどうか)
    """
    cut = im.crop(tuple(int(v) for v in box))
    if looks_rotated(ink.binarize(cut)):
        # 横倒しに組まれた表．**時計回りに 90 度回して**正す
        return cut.transpose(Image.ROTATE_270), True
    return cut, False


def split_sheet(path, outdir, page=0, dpi=300, **kw):
    """シートを表ごとの画像に切り分けて書き出す．書いた場所と箱の一覧を返す"""
    im = load_page(path, page=page, dpi=dpi)
    dark = ink.binarize(im)
    boxes = find_tables(dark, **kw)
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(outdir, exist_ok=True)
    notes = note_boxes(dark, boxes)
    written = []
    for i, (box, note) in enumerate(zip(boxes, notes), 1):
        dst = os.path.join(outdir, f'{stem}_p{i}.png')
        cut, _ = cut_table(im, box)
        cut.save(dst)
        written.append((dst, box))
        if note is not None:
            # **注記は別の画像**．表の画像に含めると検出器の入力の縮尺が変わる
            im.crop(tuple(int(v) for v in note)).save(
                os.path.join(outdir, f'{stem}_p{i}_note.png'))
    return written


def main():
    p = argparse.ArgumentParser(
        description='大きな折り込みのシートを，表ごとの画像に切り分ける')
    p.add_argument('sheet', help='PDF か画像')
    p.add_argument('--outdir', default='.', help='書き出し先')
    p.add_argument('--page', type=int, default=0, help='PDF のページ(0 から)')
    p.add_argument('--dpi', type=int, default=300, help='描画するときの解像度')
    p.add_argument('--min-gutter', type=float, default=MIN_GUTTER,
                   help='縦に切る空白帯の幅(画像の幅に対する比)')
    p.add_argument('--min-gap', type=float, default=MIN_GAP,
                   help='横に切る空白帯の高さ(画像の高さに対する比)')
    p.add_argument('--dry-run', action='store_true', help='書き出さずに数えるだけ')
    a = p.parse_args()

    im = load_page(a.sheet, page=a.page, dpi=a.dpi)
    print(f'シート  : {a.sheet}  {im.size[0]} x {im.size[1]}')
    dark = ink.binarize(im)
    boxes = find_tables(dark, min_gutter=a.min_gutter, min_gap=a.min_gap)
    print(f'--- 表 {len(boxes)} 個 ---')
    stem = os.path.splitext(os.path.basename(a.sheet))[0]
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        w, h = x2 - x1, y2 - y1
        sz = auto_imgsz(w, h)
        note = '  ← 上限で頭打ち(縮尺が学習時より小さい)' if imgsz_capped(w, h) else ''
        print(f'  p{i}  x {x1}-{x2}  y {y1}-{y2}  ({w} x {h})  imgsz={sz}{note}')
        if not a.dry_run:
            os.makedirs(a.outdir, exist_ok=True)
            dst = os.path.join(a.outdir, f'{stem}_p{i}.png')
            cut, turned = cut_table(im, (x1, y1, x2, y2))
            if turned:
                print(f'      **横倒しに組まれていたので 90 度回した** '
                      f'({cut.size[0]} x {cut.size[1]})')
            cut.save(dst)
            print(f'      書いた: {dst}')


if __name__ == '__main__':
    main()
