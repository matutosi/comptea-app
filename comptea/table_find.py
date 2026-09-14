"""検出器を使わずに組成表全体を見つける (table_find.py)

2026-09-11 ユーザ指示「検出器を使わない組成表全体の検出．各部分は不明でよい」．
物体検出 (YOLO) が落ちる紙面 (学習データが活字 121 件だけで折込は 0 件) のための
別経路．**表のどこに何があるか (部分) は決めない**．紙面のどこに組成表があるか
(箱) だけを返す．

**2026-09-12 の水準**: 本のページ 78 枚で格子との IoU >= 0.7 が **64 枚**，
折込 23 枚で表の数が合うのが **19 枚** (その日の朝は 53 枚・14 枚)．

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

2026-09-12 に足した 3 つ (ユーザ指示「次の一手を 1 から順に実装」)．

3b. **縦に並んだ項目名のまとまりだけ**を表とみなす (`keep_vertical`)．表頭の
    項目名は左端をそろえて縦に並ぶ．本文に紛れた語と，**表頭が文章で書かれた
    紙面** (kinki_026 の「Feld-Nr. 調査番号: SS-30. Datum d. Aufn. …」) は
    1 行に流れる．これだけで余分な箱が 75 → 44，折込は 14 → 19 枚になった．
3c. まとめる距離を**項目名の縦の間隔**から決める (`group_dist`)．紙面の長辺の
    1 割は紙面ごとに合わない．間隔の 16 倍が本のページ・折込の両方で最良
    (倍率を 4〜16 で振って決めた)．
4c. 塊が掴めなければ**白い縦の隙間**で伸ばす (`grow_box`)．**紙面を見て組成表と
    分かるのは，値の列のあいだに白い縦の隙間が何本も，行をまたいで同じ x に
    通っているから**で，本文にはこれが無い (2026-09-12 ユーザ助言)．黒画素の塊は
    A0 の折込では効くが，本のページでは字が段落ごとに割れて**塊が 0 個**になる
    (78 枚中 14 枚)．膨張と面積の下限を振っても IoU は 0.4 止まりだった．

**測って取り下げたもの**
- 箱の**大きさ**で偽の箱を落とす: 本のページの余分な箱は減る (39 → 32) が，
  折込の**本物の小さい表**まで落ちる (19 → 13 枚)．既定では使わない (`BOX_REL`)．
- 「表らしさ」で落とす (`looks_table`): 余分な箱 27 → 25 だが折込が 19 → 18．
  既定では使わない (`drop_flat`)．
- 塊から作った箱にも隙間で**伸ばす**: 本のページの当たりが 56 → 43 と大きく
  悪化 (41 枚で悪くなった)．塊が掴めているときは，その端の方が表の端に近い．

4d. **上下端を目印で押さえる** (`clamp_box`)．**表頭の項目名より上に表は無い**
    (上にあるのは表題や本文)．**「出現 1 回の種」は表のすぐ下**にあるので，その上で
    切る．どちらも読んだ目印そのもので，測り直す必要がない (当たり 58 → 60)．
4e. **箱の中で，列の隙間が続いている範囲だけを採る** (`table_span`)．表頭が文章で
    書かれ，しかも本文が項目名と同じ語を使う紙面があり (kinki_043 の本文
    「高木層の高さ 7 m，植被率 80%」)，目印が本文に散って箱が本文ごと覆う．
    **肝心なのは「3 行まとめて見る」こと** (`GUT_WIN`)．1 行ずつでは日本語の
    本文も語間が隙間に見える．3 行まとめると隙間は同じ x に通ったものだけが
    残り，表 3〜5 本に対し本文 0 本とはっきり分かれる (043・026・020 で実測)．
    **行の高さは黒画素から推す** (`line_pitch`)．目印の間隔は，目印が本文に
    散る紙面では当てにならない (043 は 1157 px になり，窓が高すぎて効かなかった)．
    当たり 60 → **64**．

**残る弱点**
- 折込の「1 枚に何枚あるか」(19/23)．外す 4 枚のうち 16・21 は**2 つ目・4 つ目の
  表の項目名が読めていない** (OCR の取りこぼし)，17 は**ごみを項目名と誤認**して
  横一列になり落ちる．いまの幾何の切り分け (`split_sheet.find_tables`．21/23)
  にはまだ届かない．
- 本のページで外れる 14 枚．箱が表の一部にしか掛からない紙面と，本文の語が
  項目名と重なる紙面．
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


# --- 3b. 縦に並んだ項目名だけを表とみなす ---------------------------------

X_TOL = 0.02            # 縦の連なりとみなす x のずれ (紙面の長辺に対する比)
Y_TOL = 0.004           # 同じ行とみなす y の差 (同上)．同じ行の語は 1 つと数える
VERT_LEAST = 2          # 縦の連なりに要る項目名の数


def vertical_run(points, x_tol, y_tol=0.0):
    """左端をそろえて**縦に並ぶ**点の最大の数

    表頭の項目名は「通し番号・調査番号・調査年月日…」と左端をそろえて縦に並びます．
    一方，(a) 本文に紛れた語 (b) **表頭が文章で書かれた紙面** (kinki_026 の
    「Feld-Nr. 調査番号: SS-30. Datum d. Aufn. 調査年月日: …」) は 1 行に流れます．
    縦の連なりの数で，その 2 つを分けます．
    同じ行の語は 1 つと数えます (`y_tol`)．2 段組の見出しが横に並ぶため．
    """
    pts = [(float(x), float(y)) for x, y in points]
    best = 0
    for x0, _y0 in pts:
        ys = sorted(y for x, y in pts if abs(x - x0) <= x_tol)
        n = 0
        last = None
        for y in ys:
            if last is None or y - last > y_tol:
                n += 1
                last = y
        best = max(best, n)
    return best


def keep_vertical(groups, x_tol, least=VERT_LEAST, y_tol=0.0):
    """縦の連なりが `least` に満たないまとまりを落とす"""
    return [g for g in groups if vertical_run(g, x_tol, y_tol) >= least]


# --- 3c. まとめる距離を項目名の並びから決める -----------------------------

PITCH_MUL = 16.0        # 項目名の縦の間隔のこの倍までを同じ表とみなす．
                        # 折込 23 枚で 4→16，5→16，6→17，8→16，10→16，12→17，
                        # 16→19 枚が一致．本のページは 6 以上どれも当たり 56
PITCH_LEAST = 3         # 間隔を出すのに要る項目名の数


def mark_pitch(points, x_tol, least=PITCH_LEAST):
    """縦に並ぶ項目名の**間隔の中央値** (出せなければ None)

    表頭の項目名は一定の間隔で縦に並びます．紙面の大きさで距離を決めると，
    A0 の折込では大きすぎて隣の表を橋渡しし，小さな表では足りません．
    **紙面が自分で示している物差し**を使います．
    """
    pts = [(float(x), float(y)) for x, y in points]
    best = []
    for x0, _y0 in pts:
        ys = sorted(y for x, y in pts if abs(x - x0) <= x_tol)
        if len(ys) > len(best):
            best = ys
    if len(best) < least:
        return None
    gaps = [b - a for a, b in zip(best[:-1], best[1:]) if b - a > 0]
    if not gaps:
        return None
    gaps.sort()
    return float(gaps[len(gaps) // 2])


def group_dist(points, size, x_tol, dist=DIST, mul=PITCH_MUL):
    """まとめる距離．項目名の間隔が出ればそれを使い，出なければ紙面の長辺で決める"""
    pitch = mark_pitch(points, x_tol)
    if pitch:
        return pitch * mul
    return float(max(size)) * dist


# --- 4b. 小さすぎる箱を落とす --------------------------------------------

BOX_REL = 0.0           # いちばん大きい箱の面積に対する下限．**既定では使わない**．
                        # 本のページの偽の箱は減る (余分な箱 39 → 32) が，折込の
                        # **本物の小さい表**まで落ちる (23 枚中 19 → 13．2026-09-12)


def _touches(box, blobs):
    """箱が塊から作られたか (塊のどれかと重なるか)"""
    x1, y1, x2, y2 = box
    for b in blobs:
        if not (b[2] < x1 or b[0] > x2 or b[3] < y1 or b[1] > y2):
            return True
    return False


def drop_small(boxes, rel=BOX_REL):
    """いちばん大きい箱より極端に小さい箱を落とす

    本文の語を項目名と取った偽の箱 (kinki_015 の「5 種類以下で，」) は，
    表の箱よりずっと小さくなります．1 枚に大小の表が載る紙面もあるので，
    落とすのは**極端なもの**だけにします．箱が 1 つなら落としません．
    """
    if len(boxes) < 2:
        return list(boxes)
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    big = max(area(b) for b in boxes)
    if big <= 0:
        return list(boxes)
    return [b for b in boxes if area(b) >= big * rel]


# --- 4c. 白い縦の隙間で表の範囲を伸ばす -----------------------------------
#
# **紙面を見て組成表と分かるのは，値の列のあいだに白い縦の隙間が何本も，行を
# またいで同じ x に通っているから**です (2026-09-12 ユーザ助言「人や AI が
# 組成表だと認識する方法をよく考慮する」)．本文にはこれがありません．
# 黒画素の塊 (`blob_boxes`) は A0 の折込では効きますが，本のページでは字が
# 段落ごとに割れて塊が 0 個になります (47 枚中 5 枚)．膨張と面積の下限を
# 振っても IoU は 0.4 止まりで，**塊は本のページには合わない道具**でした．

GUT_WIN = 3.0           # 隙間を見る窓の高さ (行の高さの倍数)．
                        # **1 行ずつ見てはいけない**: 日本語の本文も 1 行なら
                        # 隙間が空く．3 行まとめると，表は隙間 3〜5 本，本文は
                        # 0 本とはっきり分かれる (kinki_043・026・020 で実測)
GUT_MIN_W = 0.5         # 隙間とみなす幅 (行の高さの倍数)
GUT_LEAST = 3           # 表らしい帯に要る隙間の数
GROW_MISS = 2           # 隙間が足りない帯がこれだけ続いたら止める

# **伸ばすのは塊が掴めなかったときだけ** (2026-09-12 に測って決めた)．
# 塊から作った箱にも伸ばすと，本のページ 78 枚で当たりが 56 → 43 に落ち，
# 41 枚で悪くなった．塊が掴めているときは，その端の方が表の端に近い


def line_pitch(dark):
    """紙面の**行の高さ**を黒画素から推す

    項目名の間隔 (`mark_pitch`) は，目印が本文に散る紙面では当てにならない
    (kinki_043 は 1157 px になった)．隙間を見る窓の高さはこちらで決めます．
    """
    from .noyolo import guess_pitch
    try:
        p = float(guess_pitch(dark))
    except Exception:                           # noqa: BLE001
        p = 0.0
    return p if p >= 4 else max(4.0, dark.shape[0] * 0.01)


def gutters(dark, x1, x2, y1, y2, min_w):
    """帯 `[y1:y2, x1:x2]` を縦に貫く白い列の並び [(左, 右)]

    幅が `min_w` に満たないものと，帯の左右の余白は数えません．
    """
    h, w = dark.shape
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 - x1 < 3 or y2 - y1 < 1:
        return []
    col = dark[y1:y2, x1:x2].any(axis=0)
    if not col.any():
        return []
    lo, hi = int(np.argmax(col)), int(len(col) - np.argmax(col[::-1]))
    out = []
    run = None
    for i in range(lo, hi):
        if not col[i]:
            run = i if run is None else run
        elif run is not None:
            if i - run >= min_w:
                out.append((x1 + run, x1 + i))
            run = None
    return out


def looks_table(dark, box, pitch, least=GUT_LEAST):
    """箱の中が**表らしい**か (白い縦の隙間が `least` 本以上あるか)

    大きさで落とすと，折込の**本物の小さい表**まで落ちます (23 枚中 19 → 14)．
    表かどうかは大きさでなく**中の様子**で決めます．
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    min_w = max(4.0, float(pitch)) * GUT_MIN_W
    return len(gutters(dark, x1, x2, y1, y2, min_w)) >= least


def grow_box(dark, seed, pitch, min_w=None, least=GUT_LEAST, miss=GROW_MISS):
    """目印の箱 `seed` から，表らしい帯が続く限り上下へ伸ばす

    表らしい = 白い縦の隙間が `least` 本以上ある．文章の行は隙間が無いので止まります．
    左右は，伸ばした範囲のインクの端に合わせます (項目名は表の左半分にしかない)．
    """
    h, w = dark.shape
    pitch = max(4.0, float(pitch))
    min_w = pitch * GUT_MIN_W if min_w is None else min_w
    x1, y1, x2, y2 = (float(v) for v in seed)
    top, bot = max(0.0, y1), min(float(h), y2)

    def ok(a, b):
        return len(gutters(dark, 0, w, a, b, min_w)) >= least

    n = 0
    while bot + pitch <= h:
        if ok(bot, bot + pitch):
            bot += pitch
            n = 0
        else:
            n += 1
            bot += pitch
            if n > miss:
                bot -= pitch * (miss + 1)
                break
    n = 0
    while top - pitch >= 0:
        if ok(top - pitch, top):
            top -= pitch
            n = 0
        else:
            n += 1
            top -= pitch
            if n > miss:
                top += pitch * (miss + 1)
                break
    band = dark[max(0, int(top)):min(h, int(bot)), :]
    if band.size and band.any():
        cols = np.flatnonzero(band.any(axis=0))
        x1, x2 = float(cols[0]), float(cols[-1] + 1)
    return int(x1), int(max(0.0, top)), int(x2), int(min(float(h), bot))


# --- 4e. 箱の中で，表らしい範囲だけを取る ---------------------------------
#
# 本のページで外す紙面には，**表頭が文章**で書かれ，しかも**本文が項目名と同じ語**
# を使うものがある (kinki_043 の本文「高木層の高さ 7 m，植被率 80%」)．目印が
# 本文に散るので，箱が本文ごと覆ってしまう．
# **表は，列の隙間が縦に続いている範囲**なので，箱の中でその続きがいちばん長い
# ところだけを採る．

SPAN_LEAST = 3          # 表らしい帯に要る隙間の数 (2→62，3→64，4→61 枚)
SPAN_MISS = 3           # 隙間が足りない帯をこれだけまでは間に挟んでよい
                        # (1→62，2〜5→64 枚．3 を採る)
SPAN_MIN = 4            # 表とみなすのに要る帯の数


def table_span(dark, box, pitch, least=SPAN_LEAST, miss=SPAN_MISS,
               least_rows=SPAN_MIN, min_w=None):
    """箱の中で，表らしい帯がいちばん長く続く範囲 (y1, y2)．無ければ None"""
    x1, y1, x2, y2 = (float(v) for v in box)
    pitch = max(4.0, float(pitch))
    min_w = pitch * GUT_MIN_W if min_w is None else min_w
    ys = []
    y = y1
    while y + pitch <= y2 + pitch:
        ys.append((y, min(y + pitch, y2)))
        y += pitch
    flags = [len(gutters(dark, x1, x2, a, b, min_w)) >= least for a, b in ys]
    best = cur = None
    gap = 0
    for i, f in enumerate(flags):
        if f:
            cur = i if cur is None else cur
            gap = 0
            if best is None or (i - cur) > (best[1] - best[0]):
                best = (cur, i)
        elif cur is not None:
            gap += 1
            if gap > miss:
                cur = None
                gap = 0
    if best is None or (best[1] - best[0] + 1) < least_rows:
        return None
    return ys[best[0]][0], ys[best[1]][1]


# --- 4d. 上下端を目印で押さえる -------------------------------------------
#
# 本のページで外す紙面は，**x はほぼ正しいのに y の範囲が広すぎる**ものが大半
# だった (kinki_024 は真値 1591-1764 に対し箱 605-1910，043 は 2018-3151 に対し
# 602-3291．2026-09-12)．塊も隙間も，表の上下の本文まで拾ってしまう．
#
# **人は 2 つの手がかりで表の上下を決めている**．
#   - **表頭の項目名より上に表は無い** (上にあるのは表題や本文)
#   - **「出現 1 回の種」は表のすぐ下**にある (その上で切れる)
# どちらも読んだ目印そのものなので，あらためて測る必要がない．

TOP_PAD = 2.0           # 上端は，いちばん上の項目名からこの行数ぶんまで上を許す
ONCE_PAD = 0.5          # 「出現 1 回の種」の目印から，この行数ぶん上で切る
CLAMP_MIN = 3.0         # 押さえた結果がこの行数ぶんより低くなるなら，押さえない


def clamp_box(box, marks, once, pitch, top_pad=TOP_PAD, once_pad=ONCE_PAD,
              min_h=CLAMP_MIN):
    """箱の上下端を，項目名と「出現 1 回の種」の目印で押さえる

    Args:
        marks: そのまとまりの項目名 [(x, y)]
        once: 紙面の「出現 1 回の種」の目印 [(x, y)]
        pitch: 項目名の縦の間隔 (無ければ紙面から決めた目安)
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    pitch = max(4.0, float(pitch))
    ys = [float(y) for _x, y in marks]
    if ys:
        y1 = max(y1, min(ys) - pitch * top_pad)
    below = [float(y) for _x, y in once if not ys or float(y) > min(ys)]
    if below:
        y2 = min(y2, min(below) - pitch * once_pad)
    if y2 - y1 < pitch * min_h:                 # 潰れるなら押さえない
        return tuple(int(v) for v in box)
    return int(x1), int(y1), int(x2), int(y2)


# --- 4f. 「出現 1 回の種」「随伴種」からも表を拾う -------------------------
#
# 折込で表の数が合わない紙面は，**2 つ目・4 つ目の表の項目名が読めていない**
# ことが多い (s01115_16 の 2 つ目の表は，項目名が 1 個しか読めなかった)．
# しかし，その表の**「出現 1 回の種」と「随伴種」は読めている**．
# **「出現 1 回の種」は表のすぐ下**，**「随伴種」は表の中**にあるので，
# そこを手がかりに，上へ隙間をたどって表を拾う，という案．
#
# **測って取り下げた** (2026-09-12)．既定では使わない (`notes=False`)．
#   - すでにある箱と少しでも重なったら捨てる形 … 折込 19 枚のまま変わらない
#     (拾った箱が上の表の箱に接して，ことごとく捨てられる)
#   - 重なりを割合で見る形 … 偽の箱が増えて**折込が 19 → 4 枚**に崩れる．
#     1 つの表に「出現 1 回の種」と「Außerdem」が両方あり，種が 2 つできる．
#     `head` (随伴種) を種に足すとさらに悪い．
# 折込の弱点は**読みの取りこぼし**なので，箱の作り方ではなく読み手の側で直す．

NOTE_MIN_ROWS = 6       # 拾った箱に要る高さ (窓の数)
NOTE_REACH = 5          # 目印から上へ，この窓の数まで表を探す
NOTE_OVERLAP = 0.5      # すでにある箱とこの割合より重なるなら拾わない


def boxes_from_notes(dark, once, head, size, boxes, pitch,
                     least=GUT_LEAST, min_rows=NOTE_MIN_ROWS):
    """目印 `once`・`head` のうち，どの箱にも入らないものから表を拾う

    Args:
        once: 「出現 1 回の種」の目印 [(x, y)]．表の**すぐ下**
        head: 「随伴種」などの目印 [(x, y)]．表の**中**
        boxes: すでに作った箱
        pitch: 隙間を見る窓の高さ (行の高さの `GUT_WIN` 倍)
    """
    h, w = dark.shape
    pitch = max(4.0, float(pitch))
    out = []
    seeds = [(x, y, True) for x, y in once] + [(x, y, False) for x, y in head]
    for x, y, below in seeds:
        if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes + out):
            continue
        # 「出現 1 回の種」と表のあいだには余白や注記があるので，**上へ探しながら
        # 最初の「表らしい帯」を種にする** (`NOTE_REACH` 窓まで)
        min_w = pitch * GUT_MIN_W / GUT_WIN
        seed = None
        start = y + pitch * 0.5 if not below else y - pitch * 0.5
        for k in range(NOTE_REACH):
            b2 = start - pitch * k
            a2 = b2 - pitch
            if a2 < 0:
                break
            if len(gutters(dark, 0, w, a2, b2, min_w)) >= least:
                seed = (0, a2, w, b2)
                break
        if seed is None:
            continue
        b = grow_box(dark, seed, pitch, min_w=min_w, least=least)
        if (b[3] - b[1]) < pitch * min_rows:
            continue
        # **重なりは割合で見る**．少しでも触れたら捨てると，上の表の箱に接した
        # だけで落ちる (s01115_16 の 2 つ目の表)
        area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        over = 0.0
        for bb in boxes + out:
            ix = max(0, min(b[2], bb[2]) - max(b[0], bb[0]))
            iy = max(0, min(b[3], bb[3]) - max(b[1], bb[1]))
            over = max(over, ix * iy / area)
        if over > NOTE_OVERLAP:
            continue
        out.append(b)
    return out


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

# --- 折込は表が横に並ぶ ----------------------------------------------------
#
# 2026-09-13 に実データで確かめた: **折込 23 枚のうち 19 枚で表が横に並ぶ**
# (表頭のまとまりの x が 2 群以上に分かれる)．同じ列に縦に積まれた表の
# x の差は 60〜100 px しかないので，**紙面の幅の 2 割以上離れた所**だけを
# 列の切れ目とみなす．**2 段組は折込には無い** (ユーザ確認) ので，
# 学名の列は 1 つの表に 1 本 = 横に並ぶ表の数だけある．
COL_GAP = 0.20          # 列の切れ目とみなす x の隔たり (紙面の幅に対する比)
ROW_GAP = 0.05          # 同じ表の表頭とみなす y の隔たり (紙面の高さに対する比)
COL_X_TOL = 0.10        # 同じ列とみなす x の隔たり (同上，幅に対する比)
TABLE_LEAST = 3         # 表頭とみなすのに要る項目名の数
ONCE_GAP = 0.02         # 「1 回出現の種」を 1 か所とみなす y の隔たり


def columns(points, width, gap=COL_GAP):
    """目印を**横に並ぶ列**に分ける

    Args:
        points: (x, y) の並び
        width: 紙面の幅
    Returns:
        列ごとの点の並び (x の順)
    """
    if not points:
        return []
    out = []
    for p in sorted(points, key=lambda q: q[0]):
        if out and p[0] - max(q[0] for q in out[-1]) <= width * gap:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def _by_y(points, gap):
    """y の隔たりで切る"""
    out = []
    for p in sorted(points, key=lambda q: q[1]):
        if out and p[1] - max(q[1] for q in out[-1]) <= gap:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def count_tables(points, size, once=(), least=TABLE_LEAST):
    """紙面に表が何枚あるかを，目印だけから見積もる

    **表頭のまとまりが表の上端，「1 回出現の種」が下端**という並びを使う
    (ユーザの基本構造．2026-09-13)．**表頭は非必須**なので，
    列ごとに**表頭と「1 回出現の種」の多い方**を採る．

    実測 (折込 23 枚): 表頭だけ 18 枚・「1 回出現の種」だけ 17 枚・
    **多い方 19 枚**が，いまの切り分けの枚数と一致した．
    """
    w, h = size
    n = 0
    for col in columns(list(points) + list(once), w):
        heads = [p for p in col if p in set(points)]
        ones = [p for p in col if p in set(once)]
        a = len([g for g in _by_y(heads, h * ROW_GAP) if len(g) >= least])
        b = len(_by_y(ones, h * ONCE_GAP))
        n += max(a, b)
    return max(n, 1)


def _heads_in(col, points):
    got = []
    for p in col:
        if any(p is q or p == q for q in points):
            got.append(p)
    return got


def refine_columns(dark, bounds, spans):
    """列の境を，**隣り合う表頭のあいだのいちばん広い白い隙間**へ寄せる

    表頭は表の左端にあるので，**表頭の中間は左の表の本体の中**に落ちる
    (2026-09-13 の実データ: 幅 2823 px の表に対し箱が 5627 px になった)．

    Args:
        dark: 紙面のインク (True が黒)
        bounds: 列の境 (両端を含む)．`len(spans) + 1` 本
        spans: 列ごとの表頭の x の範囲 [(x1, x2)]
    Returns:
        寄せた境 (両端は動かさない)
    """
    out = list(bounds)
    if dark is None or len(out) < 3:
        return out
    col = dark.any(axis=0) if dark.ndim == 2 else dark
    for i in range(1, len(out) - 1):
        lo = int(spans[i - 1][1])           # 左の列の表頭の右端
        hi = int(spans[i][0])               # 右の列の表頭の左端
        lo, hi = max(0, lo), min(len(col), hi)
        if hi - lo < 2:
            continue
        # いちばん長い「インクの無い」並びの中央へ
        best = run = 0
        end = -1
        for x in range(lo, hi):
            if not col[x]:
                run += 1
                if run > best:
                    best, end = run, x
            else:
                run = 0
        if best >= 2:
            out[i] = end - best // 2
    return out


GUT_MIN_W = 0.02        # 表の境とみなす隙間の幅 (箱の幅に対する比)
GUT_MIN_H = 0.80        # その隙間が箱の高さのこの割合以上つづくこと


def split_at_gutters(dark, box, min_w=GUT_MIN_W, min_h=GUT_MIN_H):
    """箱を，**縦に長い白い隙間**で左右に分ける

    **表頭の見つからない表は列を作らない**ので，1 つの帯に 2 つの表が
    横に並んだまま入ることがある (2026-09-13 の実データ)．
    **折込に 2 段組は無い**ので，帯の中の縦に長い隙間は表と表の境とみなす．
    """
    keep = tuple(int(v) for v in box)
    # **先にインクへ縮める**．そうしないと箱の外側の余白で切ってしまう
    x1, y1, x2, y2 = shrink_to_ink(dark, box, pad=0)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return [keep]
    sub = dark[y1:y2, x1:x2]
    # **その x にインクが無い画素行の割合**が高いところが隙間
    empty = ((~sub).sum(axis=0) / float(y2 - y1)) >= min_h
    least = max(2, int((x2 - x1) * min_w))
    cuts = []
    run = 0
    for i, e in enumerate(list(empty) + [False]):
        if e:
            run += 1
            continue
        # **端に接する空白は余白**なので境にしない
        if run >= least and i - run > 0 and i < len(empty):
            cuts.append(x1 + i - run // 2)
        run = 0
    if not cuts:
        return [keep]
    out = []
    edges = [keep[0]] + cuts + [keep[2]]
    for a, b in zip(edges, edges[1:]):
        if b - a > least:
            out.append((a, keep[1], b, keep[3]))
    return out or [keep]


def structure_boxes(points, size, once=(), least=TABLE_LEAST, dark=None,
                    split=False):
    """目印から**表の箱**を作る (基本構造にしたがう)

    基本構造 (2026-09-13 ユーザ):
        表題 → 表頭項目+値 → 学名・和名・組成 → 1 回出現の種 → 地点情報

    **表頭のまとまりが表の上端**，**同じ列の次の表頭の直前が下端**．
    横は**列と列の中間**で分ける (折込 23 枚中 19 枚で表が横に並ぶ)．
    **表頭は非必須**なので，表頭の無い列は紙面の上から下までを 1 つとする．

    Returns:
        [(x1, y1, x2, y2)]．x の順，同じ列では y の順
    """
    w, h = size
    pts = list(points)
    ones = list(once)
    cols = columns(pts + ones, w)
    if not cols:
        return [(0, 0, int(w), int(h))]

    span = [(min(p[0] for p in c), max(p[0] for p in c)) for c in cols]
    bounds = [0.0]
    for i in range(1, len(cols)):
        bounds.append((span[i - 1][1] + span[i][0]) / 2)
    bounds.append(float(w))
    # **境は白い隙間へ寄せる** (表頭の中間は左の表の本体の中に落ちる)
    bounds = refine_columns(dark, bounds, span)
    out = []
    for i, col in enumerate(cols):
        x1, x2 = bounds[i], bounds[i + 1]
        heads = _heads_in(col, pts)
        tops = [min(g, key=lambda p: p[1])[1]
                for g in _by_y(heads, h * ROW_GAP) if len(g) >= least]
        tops.sort()
        if not tops:
            tops = [0.0]                      # 表頭が無い列は紙面の上から
        for j, top in enumerate(tops):
            y1 = 0 if j == 0 else top
            y2 = tops[j + 1] if j + 1 < len(tops) else h
            box = (int(x1), int(y1), int(x2), int(y2))
            # **隙間での分割は既定では使わない**．2026-09-13 の実測で，
            # **紙面の 23 枚中 11 枚に端から端まで通る隙間が無い**
            # (表が互い違いに並ぶため)．当てると分けすぎる (68 → 89 箱)
            out += (split_at_gutters(dark, box) if split else [box])
    return out


SHRINK_PAD = 40         # 縮めたあとに付ける余白 (px)


def shrink_to_ink(dark, box, pad=SHRINK_PAD):
    """箱を**その中のインクの外接矩形**まで縮める

    `structure_boxes` は紙面を隙間なく分けるので，そのままでは余白を含む．
    切り出しに使うときはここで縮める．インクが無ければ元の箱のまま．
    """
    h, w = dark.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x2, y2)
    sub = dark[y1:y2, x1:x2]
    rows = np.flatnonzero(sub.any(axis=1))
    cols = np.flatnonzero(sub.any(axis=0))
    if not len(rows) or not len(cols):
        return (x1, y1, x2, y2)
    a = max(0, x1 + int(cols[0]) - pad)
    b = max(0, y1 + int(rows[0]) - pad)
    c = min(w, x1 + int(cols[-1]) + 1 + pad)
    d = min(h, y1 + int(rows[-1]) + 1 + pad)
    return (a, b, c, d)


def _gap_near(dark, lo, hi, x1, x2, axis):
    """`lo`〜`hi` のあいだで，いちばん長いインクの無い並びの中央"""
    if dark is None or hi - lo < 3:
        return None
    lo, hi = int(max(0, lo)), int(hi)
    if axis == 'y':
        sub = dark[lo:hi, int(x1):int(x2)]
        empty = ~sub.any(axis=1)
    else:
        sub = dark[int(x1):int(x2), lo:hi]
        empty = ~sub.any(axis=0)
    best = run = 0
    end = -1
    for i, e in enumerate(list(empty) + [False]):
        if e:
            run += 1
            if run > best:
                best, end = run, i
        else:
            run = 0
    return lo + end - best // 2 if best >= 2 else None


def split_between_heads(box, groups, dark=None):
    """**2 つ以上の表頭が入る箱**を，表頭のあいだで切る

    2026-09-13 の検算で見つかった 3 箱を調べると，2 つの表頭は
    **y だけ離れている** (縦に積まれている) ことが多かった．
    紙面ぜんぶの隙間で切ると分けすぎる (箱 1 つが 15 分割) ので，
    **2 つの表頭のあいだ**だけを見る．画像を渡すと白い隙間へ寄せる．
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    if len(groups) < 2:
        return [(x1, y1, x2, y2)]
    tops = sorted((min(p[1] for p in g), min(p[0] for p in g),
                   max(p[0] for p in g)) for g in groups)
    lefts = sorted((min(p[0] for p in g), min(p[1] for p in g)) for g in groups)
    # **縦に積まれているか，横に並んでいるか**を，隔たりの大きい方で決める
    dy = tops[-1][0] - tops[0][0]
    dx = lefts[-1][0] - lefts[0][0]
    out = []
    if dy >= dx:
        edges = [y1]
        for a, b in zip(tops, tops[1:]):
            mid = (a[0] + b[0]) / 2
            got = _gap_near(dark, a[0], b[0], x1, x2, 'y')
            edges.append(int(got if got is not None else mid))
        edges.append(y2)
        for a, b in zip(edges, edges[1:]):
            out.append((x1, a, x2, b))
    else:
        edges = [x1]
        for a, b in zip(lefts, lefts[1:]):
            mid = (a[0] + b[0]) / 2
            got = _gap_near(dark, a[0], b[0], y1, y2, 'x')
            edges.append(int(got if got is not None else mid))
        edges.append(x2)
        for a, b in zip(edges, edges[1:]):
            out.append((a, y1, b, y2))
    return out


def check_boxes(boxes, points, size, once=(), least=TABLE_LEAST):
    """切り分けた箱を**目印で検算**する

    2026-09-13 の実測で分かったこと: **目印だけで切り出しを作り直すのは無理**
    (紙面 23 枚のうち 11 枚に端から端まで通る白い隙間が無く，表は互い違いに
    並ぶ)．幾何の切り分け (21/23) を置き換えられない．
    **そこで検算に使う**: 1 つの箱に**表頭のまとまりがちょうど 1 つ**入るか．

    Returns:
        箱ごとに {'box', 'heads', 'once', 'ok', 'why'}
    """
    w, h = size
    groups = [g for c in columns(list(points), w)
              for g in _by_y(c, h * ROW_GAP) if len(g) >= least]
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b
        inside = []
        for g in groups:
            cx = sum(p[0] for p in g) / len(g)
            cy = min(p[1] for p in g)
            if x1 <= cx < x2 and y1 <= cy < y2:
                inside.append(g)
        n = len(inside)
        m = sum(1 for p in once if x1 <= p[0] < x2 and y1 <= p[1] < y2)
        why = ''
        if n > 1:
            why = f'表頭が {n} つ入る (切り足りない)'
        elif n == 0:
            why = '表頭が無い (表でない切れ端か，表頭の読めない表)'
        out.append({'box': tuple(b), 'heads': n, 'once': m,
                    'ok': n == 1, 'why': why, 'groups': inside})
    return out


def recheck_boxes(im, boxes, points, size, once=(), reader=None,
                  least=TABLE_LEAST, tile=TILE):
    """検算で**表頭が無いと出た箱**だけ，切り出して読み直す

    2026-09-13 の実測: 「表頭が無い」6 箱のうち **4 箱は本物の表**で，
    格子には表頭項目が 11〜20 個あった．**紙面ぜんぶを読んだときに目印が
    拾えなかっただけ**．小さく切り出すと読める (タイルや注記と同じ)．
    """
    got = check_boxes(boxes, points, size, once=once, least=least)
    for g in got:
        if g['heads']:
            continue
        x1, y1, x2, y2 = (int(v) for v in g['box'])
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = im.crop((x1, y1, x2, y2))
        # read_marks は (目印, 採った向き) を返す
        marks, _how = read_marks(crop, reader=reader, tile=tile)
        pts = [(float(m[1]) + x1, float(m[2]) + y1) for m in marks
               if m[0] == 'item']
        if not pts:
            continue
        w, h = size
        groups = [c for c in _by_y(pts, h * ROW_GAP) if len(c) >= least]
        if groups:
            g['groups'] = groups
            g['heads'] = len(groups)
            g['ok'] = len(groups) == 1
            g['why'] = '' if len(groups) == 1 else f'表頭が {len(groups)} つ入る'
    return got


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

def find_by_marks(im, reader=None, tile=TILE, dist=DIST, least=LEAST,
                  pad=PAD):
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
    dark = ink.binarize(im)
    blobs = blob_boxes(dark)
    once = [(x, y) for k, x, y, _t in marks if k == 'once']
    head = [(x, y) for k, x, y, _t in marks if k == 'head']
    return boxes_from(points, blobs, im.size, dist=dist, least=least, pad=pad,
                      dark=dark, once=once, head=head)


def boxes_from(points, blobs, size, dist=DIST, least=LEAST, pad=PAD,
               x_tol=X_TOL, y_tol=Y_TOL, vert_least=VERT_LEAST,
               box_rel=BOX_REL, use_pitch=True, dark=None, drop_flat=False,
               mul=PITCH_MUL, once=None, clamp=True, span=True,
               span_least=SPAN_LEAST, span_miss=SPAN_MISS,
               head=None, notes=False, use_head=False):
    """目印と塊から表の箱を作る (OCR を切り離した部分．案を測るのに使う)"""
    long = float(max(size))
    d = (group_dist(points, size, long * x_tol, dist=dist, mul=mul) if use_pitch
         else long * dist)
    groups = cluster(points, d, least=least)
    groups = keep_vertical(groups, long * x_tol, least=vert_least,
                           y_tol=long * y_tol)
    if not groups:
        return []
    lp = line_pitch(dark) if dark is not None else 0.0
    boxes = []
    for g in groups:
        b = box_for(g, blobs, pad=pad, max_frac=MAX_FRAC, size=size)
        # **塊が掴めなかったら，白い縦の隙間で伸ばす** (本のページは字が段落ごとに
        # 割れて塊が 0 個になる)．目印の外接矩形のままでは表頭の左半分しか入らない
        pitch = mark_pitch(g, float(max(size)) * x_tol) or float(max(size)) * 0.01
        if dark is not None and not _touches(b, blobs):
            b = grow_box(dark, b, lp * GUT_WIN, min_w=lp * GUT_MIN_W)
        # **上下端を目印で押さえる** (表頭の項目名より上・「出現 1 回の種」より下は表でない)
        if clamp:
            b = clamp_box(b, g, once or [], pitch)
        # **箱の中で，列の隙間が続いている範囲だけを採る** (本文を覆ってしまった分を削る)
        if dark is not None and span:
            sp = table_span(dark, b, lp * GUT_WIN, least=span_least,
                            miss=span_miss, min_w=lp * GUT_MIN_W)
            if sp is not None:
                b = (b[0], int(sp[0]), b[2], int(sp[1]))
        boxes.append(b)
    # **項目名が読めなかった表を，「出現 1 回の種」「随伴種」から拾う**
    if dark is not None and notes and (once or head):
        boxes += boxes_from_notes(dark, once or [],
                                  (head or []) if use_head else [],
                                  size, boxes, lp * GUT_WIN)
    boxes = drop_small(set(boxes), rel=box_rel)
    # **表らしくない箱を落とす** (本文の語を拾った偽の箱)．**既定では使わない**:
    # 本のページの余分な箱は 27 → 25 に減るが，折込が 19 → 18 枚に落ちる
    if dark is not None and drop_flat and len(boxes) > 1:
        pitch = float(max(size)) * 0.005
        keep = [b for b in boxes if looks_table(dark, b, pitch)]
        if keep:
            boxes = keep
    return sorted(boxes, key=lambda b: (b[1], b[0]))
