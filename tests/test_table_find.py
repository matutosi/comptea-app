"""検出器を使わずに組成表全体を見つける (comptea/tf.py)

2026-09-11 ユーザ指示「検出器を使わない組成表全体の検出．各部分は不明で
よい．TDD で実装」．

手順は実測で当たったもの (2026-09-11 の試行．本のページ 9/10，折込は表の数が
23 枚中 14 枚で一致)．
    1. 紙面を**タイルに分けて** OCR する (EasyOCR は長辺 2560 px に縮めるので，
       そのまま渡すと項目名が読めない)
    2. 表頭の項目名 (「通し番号」「Lfd. Nr.」…) を目印にする
    3. 近い目印をまとめる (1 まとまり = 1 表)
    4. まとまりの中心を含む**黒画素の塊**の外接矩形を，表の箱にする
    5. 目印がほとんど無ければ**横倒し**なので，回して読み直す (s01115_04 は
       そのままで 0 個，時計回りに回すと 48 個)

OCR を使う部分は重いので，純粋な関数を分けてそちらを常に走らせ，
OCR を通す部分は偽の読み手か実データ (slow) で確かめる．
"""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from comptea import table_find as tf


# --- 2. 文字の型で目印を選ぶ (純粋) ---------------------------------------

def test_項目名と見出しと区切りを目印にする():
    got = tf.classify([('通し番号', 10, 20), ('アカマツ', 10, 50),
                       ('調査年月日', 5, 30), ('出現1回の種', 10, 90),
                       ('Begleiter:', 8, 70), ('', 0, 0), ('   ', 1, 1)])
    kinds = [(k, t) for k, _x, _y, t in got]
    assert ('item', '通し番号') in kinds
    assert ('item', '調査年月日') in kinds
    assert ('once', '出現1回の種') in kinds
    assert ('head', 'Begleiter:') in kinds
    assert all(t not in ('アカマツ', '', '   ') for _k, t in kinds)


def test_独文の語彙は目印にしない():
    """測って取り下げた．注記の「Datum」が偽のまとまりになり (12 は 2 表 → 3 箱)，
    独文の項目名が隣の表と縦に連なって橋渡しする (09 は 5 表 → 4 箱)"""
    got = tf.classify([('Lfd. Nr.', 5, 30), ('Datum d. Aufn.', 5, 60),
                       ('Ort d. Aufn.', 5, 90)])
    assert [k for k, *_ in got if k == 'item'] == []


# --- 3. 近い目印をまとめる (純粋) ------------------------------------------

def test_近い点だけをまとめる():
    pts = [(100, 100), (110, 130), (105, 160),          # 1 つ目
           (900, 100), (905, 140),                      # 2 つ目
           (100, 900)]                                  # 1 個だけ
    groups = tf.cluster(pts, dist=200)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2, 3]


def test_まとまりに要る数で1個だけの拾いを落とす():
    pts = [(100, 100), (110, 130), (100, 900)]
    assert len(tf.cluster(pts, dist=200, least=2)) == 1


def test_離れていれば別のまとまり():
    pts = [(0, 0), (0, 199), (0, 401)]                  # 199 は近い，401 は遠い
    assert len(tf.cluster(pts, dist=200)) == 2


# --- 3b. 縦に並んだ項目名だけを表とみなす (純粋) ---------------------------

def test_縦に並ぶ項目名の数を数える():
    """表頭の項目名は左端をそろえて縦に並ぶ (通し番号・調査番号・調査年月日…)"""
    pts = [(100, 100), (104, 140), (98, 180), (400, 100)]
    assert tf.vertical_run(pts, x_tol=20) == 3


def test_横に並ぶ語は縦の連なりにならない():
    """本文や文章の表頭は 1 行に流れる (kinki_026 の「Feld-Nr. 調査番号: …」)"""
    pts = [(100, 100), (400, 104), (700, 98)]
    assert tf.vertical_run(pts, x_tol=20) == 1


def test_同じ高さの語は1つと数える():
    """左端がそろっていても同じ行なら 1 つ (2 段組の見出しが並んだもの)"""
    pts = [(100, 100), (104, 102), (100, 200)]
    assert tf.vertical_run(pts, x_tol=20, y_tol=20) == 2


def test_縦に並ばないまとまりは表とみなさない():
    """`find_by_marks` の間引き．横一列の 3 語だけのまとまりは落ちる"""
    groups = [[(100, 100), (104, 140), (98, 180)],      # 縦に 3 つ
              [(900, 900), (1200, 904), (1500, 898)]]   # 横に 3 つ
    kept = tf.keep_vertical(groups, x_tol=20, least=2)
    assert len(kept) == 1 and kept[0][0] == (100, 100)


# --- 3c. まとめる距離を項目名の並びから決める (純粋) ----------------------

def test_項目名の縦の間隔から距離を決める():
    """表頭の項目名は一定の間隔で縦に並ぶ．その間隔の何倍かでまとめる"""
    pts = [(100, 100), (100, 140), (100, 180), (100, 220)]
    assert tf.mark_pitch(pts, x_tol=20) == 40


def test_縦に並ばなければ間隔は出ない():
    pts = [(100, 100), (400, 104), (700, 98)]
    assert tf.mark_pitch(pts, x_tol=20) is None


def test_間隔が出ればそれを距離に使う():
    """紙面の長辺の 1 割は，紙面ごとに合ったり合わなかったりする"""
    pts = [(100, 100), (100, 140), (100, 180)]
    assert tf.group_dist(pts, size=(4000, 6000), x_tol=20) == 40 * tf.PITCH_MUL


def test_間隔が出なければ紙面の長辺で決める():
    pts = [(100, 100), (400, 104)]
    assert tf.group_dist(pts, size=(4000, 6000), x_tol=20) == 6000 * tf.DIST


# --- 4b. 小さすぎる箱を落とす (純粋) --------------------------------------

def test_いちばん大きい箱より極端に小さい箱は落とす():
    """本文の語を拾った偽の箱は，表の箱よりずっと小さい"""
    boxes = [(0, 0, 1000, 1000), (10, 10, 60, 60)]
    assert tf.drop_small(boxes, rel=0.05) == [(0, 0, 1000, 1000)]


def test_同じくらいの大きさの箱は残す():
    """1 枚に大小の表が載る紙面もあるので，落とすのは極端なものだけ"""
    boxes = [(0, 0, 1000, 1000), (0, 0, 400, 400)]
    assert len(tf.drop_small(boxes, rel=0.05)) == 2


def test_箱が1つなら落とさない():
    boxes = [(10, 10, 60, 60)]
    assert tf.drop_small(boxes, rel=0.05) == boxes


# --- 4c. 白い縦の隙間で表の範囲を伸ばす (純粋) ----------------------------

def _dark_table(w=400, h=300, y0=0, y1=150, pitch=10, cols=(100, 160, 220, 280)):
    """上半分が表 (値が列に並ぶ)，下半分が文章の紙面の黒画素"""
    d = np.zeros((h, w), dtype=bool)
    for y in range(y0, y1, pitch):
        d[y:y + 4, 20:80] = True                    # 種名
        for x in cols:
            d[y:y + 4, x:x + 6] = True              # 値
    for y in range(y1, h, pitch):
        d[y:y + 4, 20:w - 20] = True                # 文章 (幅いっぱい)
    return d


def test_表らしい帯には白い縦の隙間が何本もある():
    d = _dark_table()
    assert len(tf.gutters(d, 0, 400, 0, 150, min_w=8)) >= 3


def test_文章の帯には縦の隙間が無い():
    d = _dark_table()
    assert len(tf.gutters(d, 0, 400, 160, 300, min_w=8)) == 0


def test_表らしい行が続く限り下へ伸ばす():
    """文章に変わったところで止まる (表の下端 150 の前後で止まること)"""
    d = _dark_table()
    box = tf.grow_box(d, (20, 0, 286, 40), pitch=10, min_w=8)
    assert 130 <= box[3] <= 170


def test_伸ばした範囲のインクの端を箱の左右にする():
    d = _dark_table()
    box = tf.grow_box(d, (100, 0, 180, 40), pitch=10, min_w=8)
    assert box[0] <= 20 and box[2] >= 286


# --- 4e. 箱の中で，表らしい範囲だけを取る (純粋) --------------------------

def test_文章が混じった箱から表の範囲だけを取る():
    """上が文章・下が表の紙面．箱が両方を覆っていても，表の側だけにする"""
    d = _dark_table(h=400, y1=200)              # 0-200 が表，200-400 が文章
    got = tf.table_span(d, (0, 0, 400, 400), pitch=10)
    assert got is not None
    a, b = got
    assert a <= 20 and 180 <= b <= 230


def test_表らしい範囲が無ければ_None():
    d = np.zeros((200, 400), dtype=bool)
    for y in range(0, 200, 10):
        d[y:y + 4, 20:380] = True               # 文章だけ
    assert tf.table_span(d, (0, 0, 400, 200), pitch=10) is None


def test_いちばん長い連なりを採る():
    """紙面に表が 2 つあっても，箱の中でいちばん長く続く方を採る"""
    d = np.zeros((400, 400), dtype=bool)
    for y in range(0, 60, 10):                  # 短い表
        for x in (20, 100, 200, 300):
            d[y:y + 4, x:x + 6] = True
    for y in range(200, 380, 10):               # 長い表
        for x in (20, 100, 200, 300):
            d[y:y + 4, x:x + 6] = True
    a, b = tf.table_span(d, (0, 0, 400, 400), pitch=10)
    assert a >= 180 and b >= 360


def test_本文の帯も1行なら隙間が空く():
    """**1 行ずつ見てはいけない**．日本語の本文は 1 行なら語間が隙間に見える

    3 行まとめると，表は隙間が同じ x に通るので残り，本文は消える
    (kinki_043・026・020 で実測: 表 3〜5 本に対し本文 0 本)．
    """
    d = np.zeros((300, 400), dtype=bool)
    for i, y in enumerate(range(0, 300, 10)):
        a = 80 + (i % 5) * 14                   # 行ごとに語の切れ目がずれる
        d[y:y + 4, 20:a] = True
        d[y:y + 4, a + 10:380] = True
    one = len(tf.gutters(d, 0, 400, 0, 10, min_w=5))
    three = len(tf.gutters(d, 0, 400, 0, 30, min_w=5))
    assert one >= 1 and three == 0


def test_行の高さは黒画素から推す():
    """目印の間隔は，目印が本文に散る紙面では当てにならない (043 は 1157 px)"""
    d = np.zeros((600, 400), dtype=bool)
    for y in range(0, 600, 20):
        d[y:y + 8, 50:350] = True
    p = tf.line_pitch(d)
    assert 12 <= p <= 30


# --- 4d. 上下端を目印で押さえる (純粋) ------------------------------------

def test_箱の上端は一番上の項目名より上へ行かない():
    """**表頭の項目名より上に表は無い**．上にあるのは表題や本文"""
    box = tf.clamp_box((100, 0, 900, 2000), marks=[(120, 500), (120, 560)],
                       once=[], pitch=40)
    assert box[1] >= 500 - 40 * 2 and box[1] <= 500


def test_出現1回の種より下は表でない():
    """**「出現 1 回の種」は表のすぐ下**にある．その上で切る"""
    box = tf.clamp_box((100, 400, 900, 2000), marks=[(120, 500)],
                       once=[(150, 1500)], pitch=40)
    assert 1400 <= box[3] <= 1500


def test_目印の上にある出現1回の種は使わない():
    """前の表の「出現 1 回の種」が上にあることがある"""
    box = tf.clamp_box((100, 400, 900, 2000), marks=[(120, 900)],
                       once=[(150, 300)], pitch=40)
    assert box[3] == 2000


def test_押さえた結果が潰れるなら元のまま():
    box = tf.clamp_box((100, 400, 900, 800), marks=[(120, 700)],
                       once=[(150, 720)], pitch=40)
    assert box == (100, 400, 900, 800)


# --- 4f. 「出現 1 回の種」「随伴種」からも表を拾う (純粋) -----------------

def test_出現1回の種の上に表を拾う():
    """**「出現 1 回の種」は表のすぐ下**にある．項目名が読めなくても，
    その上に表があると分かる (s01115_16 の 2 つ目の表)"""
    d = _dark_table(h=400, y1=250)              # 0-250 が表，その下は文章
    got = tf.boxes_from_notes(d, [(200, 300)], [], (400, 400), [], pitch=30)
    assert len(got) == 1
    x1, y1, x2, y2 = got[0]
    assert y1 <= 40 and 200 <= y2 <= 300


def test_すでに箱がある所では拾わない():
    d = _dark_table(h=400, y1=250)
    got = tf.boxes_from_notes(d, [(200, 300)], [], (400, 400),
                              [(0, 0, 400, 320)], pitch=30)
    assert got == []


def test_上に表が無ければ拾わない():
    d = np.zeros((400, 400), dtype=bool)
    for y in range(0, 400, 10):
        d[y:y + 4, 20:380] = True               # 文章だけ
    assert tf.boxes_from_notes(d, [(200, 300)], [], (400, 400), [], pitch=30) == []


# --- 5. 回した座標を元の紙面へ戻す (純粋) ---------------------------------

@pytest.mark.parametrize('how', ['cw', 'ccw'])
def test_回して読んだ位置を元に戻す(how):
    """回した画像の (x, y) は，元の画像のどこか．PIL の transpose と一致させる"""
    im = Image.new('L', (60, 40), 255)
    im.putpixel((50, 10), 0)                            # 元の画像の目印
    rot = tf.rotate(im, how)
    ys, xs = np.nonzero(np.asarray(rot) == 0)
    rx, ry = float(xs[0]), float(ys[0])
    ox, oy = tf.rotate_back(rx, ry, im.size, how)
    assert (round(ox), round(oy)) == (50, 10)


# --- 4. まとまりを含む塊を箱にする (純粋) -------------------------------

def test_目印に触れる塊を箱にする():
    blobs = [(0, 0, 100, 100), (200, 200, 500, 600)]
    group = [(300, 300), (320, 340), (310, 380)]
    assert tf.box_for(group, blobs) == (200, 200, 500, 600)


def test_触れる塊が複数なら合わせる():
    """字の段落ごとに割れた塊を 1 つだけ掴まない (12 は幅 104 px の塊になった)"""
    blobs = [(200, 200, 320, 600), (330, 200, 500, 600), (900, 900, 950, 950)]
    group = [(300, 300), (340, 340)]
    assert tf.box_for(group, blobs) == (200, 200, 500, 600)


def test_塊が無ければ目印の外接矩形を広げる():
    group = [(300, 300), (320, 340), (310, 380)]
    x1, y1, x2, y2 = tf.box_for(group, [], pad=20)
    assert (x1, y1, x2, y2) == (280, 280, 340, 400)


def test_塊が大きすぎれば目印の外接矩形にする():
    """紙面ぜんぶが 1 つの塊になることがある (膨張しすぎ)．それは箱にしない"""
    blobs = [(0, 0, 1000, 1000)]
    group = [(300, 300), (320, 340)]
    box = tf.box_for(group, blobs, pad=10, max_frac=0.5, size=(1000, 1000))
    assert box == (290, 290, 330, 350)


# --- 1・5. タイルに分けて読む・横倒しなら回す (偽の読み手で) ----------------

class FakeReader:
    """タイルの中に**横長**の黒い塊があれば項目名を返す読み手

    横倒しの紙面では塊が縦長に見えるので何も返さない．回すと横長になる．
    """

    def __init__(self):
        self.calls = 0

    def readtext(self, arr, detail=1, paragraph=False):
        self.calls += 1
        dark = np.asarray(arr)[..., 0] < 128
        if not dark.any():
            return []
        ys, xs = np.nonzero(dark)
        w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if w < 3 * h:
            return []
        cx, cy = float(xs.mean()), float(ys.mean())
        box = [[cx - 5, cy - 3], [cx + 5, cy - 3], [cx + 5, cy + 3], [cx - 5, cy + 3]]
        return [(box, '通し番号', 0.9)]


def _sheet(size=(300, 200), rect=(40, 60, 100, 68)):
    im = Image.new('RGB', size, 'white')
    ImageDraw.Draw(im).rectangle(rect, fill='black')
    return im


def test_タイルに分けて読む():
    im = _sheet()
    marks, how = tf.read_marks(im, reader=FakeReader(), tile=120)
    assert how == 'up'
    assert [k for k, *_ in marks] == ['item']
    _k, x, y, _t = marks[0]
    assert 40 <= x <= 100 and 55 <= y <= 75


def test_横倒しなら回して読む():
    """そのままでは目印が無く，回すと出る紙面 (s01115_04 の型)"""
    im = tf.rotate(_sheet(), 'cw')                     # わざと横倒しにする
    marks, how = tf.read_marks(im, reader=FakeReader(), tile=120, least=1)
    assert how in ('cw', 'ccw')
    assert len(marks) == 1
    _k, x, y, _t = marks[0]
    assert 0 <= x <= im.width and 0 <= y <= im.height   # 元の紙面の座標


def test_目印が足りていれば回さない():
    r = FakeReader()
    tf.read_marks(_sheet(), reader=r, tile=120, least=1)
    n_up = r.calls
    assert n_up <= 6                                    # 回した分は読んでいない


# --- 通し (偽の読み手) -----------------------------------------------------

def _stacked(size=(400, 300), x=(40, 100), ys=(60, 90, 120), h=8):
    """項目名らしく**縦に並ぶ**棒を描いた紙面"""
    im = Image.new('RGB', size, 'white')
    dr = ImageDraw.Draw(im)
    for y in ys:
        dr.rectangle((x[0], y, x[1], y + h), fill='black')
    return im


def test_通しで表の箱が1つ出る():
    im = _stacked()
    boxes = tf.find_by_marks(im, reader=FakeReader(), tile=40, least=1)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 40 and y1 <= 60 and x2 >= 100 and y2 >= 128


def test_横一列の目印だけでは箱を出さない():
    """表頭が文章の紙面や本文の語 (kinki_026 型) を落とす"""
    im = _sheet(size=(400, 300), rect=(40, 60, 100, 68))
    assert tf.find_by_marks(im, reader=FakeReader(), tile=150, least=1) == []


# --- 実データ (slow) -------------------------------------------------------

SCAN = os.environ.get('COMPTEA_SCAN', '')
GRIDS = os.environ.get('COMPTEA_GRIDS', '')
PARTS = os.environ.get('COMPTEA_PARTS', '')
CACHE = os.environ.get('COMPTEA_TF_CACHE', '')


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


@pytest.mark.slow
@pytest.mark.parametrize('name', ['s01114_kinki_017', 's01114_kinki_040'])
def test_本のページで組成表を囲む(name):
    """検出器で作った格子の箱 (表頭〜本体) と，7 割以上重なること"""
    pd = pytest.importorskip('pandas')
    jpg = os.path.join(SCAN, name + '.jpg')
    csv = os.path.join(GRIDS, name, 'located.csv')
    if not (os.path.exists(jpg) and os.path.exists(csv)):
        pytest.skip('実データが無い')
    d = pd.read_csv(csv)
    d = d[d.obj_name.isin(('comp', 'sname', 'species_col', 'layer',
                           'header_value', 'header_item', 'header_item_ja'))]
    truth = (float(d.x1.min()), float(d.y1.min()),
             float(d.x2.max()), float(d.y2.max()))
    boxes = tf.find_by_marks(Image.open(jpg))
    assert boxes, '表が見つからない'
    assert max(_iou(b, truth) for b in boxes) >= 0.7


@pytest.mark.slow
@pytest.mark.parametrize('num,n_tables', [('09', 5), ('12', 2)])
def test_折込で表の数が合う(num, n_tables):
    """折込 (A0 級・複数の表) で，まとまりの数が表の数と合うこと

    2026-09-11 の試行で表の数が一致した紙面を的にする (23 枚中 14 枚)．
    """
    pdf = os.path.join(SCAN, f's01115_{num}.pdf')
    if not os.path.exists(pdf):
        pytest.skip('実データが無い')
    from comptea import split_sheet
    im = split_sheet.load_page(pdf)
    boxes = tf.find_by_marks(im)
    assert len(boxes) == n_tables, f'表 {n_tables} 枚のはずが {len(boxes)} 箱'


@pytest.mark.slow
def test_横倒しの折込でも表が出る():
    """s01115_04 の 3 表は紙面上で横倒し．そのままでは項目名 0 個，回すと 48 個"""
    pdf = os.path.join(SCAN, 's01115_04.pdf')
    if not os.path.exists(pdf):
        pytest.skip('実データが無い')
    from comptea import split_sheet
    im = split_sheet.load_page(pdf)
    marks, how = tf.read_marks(im)
    assert how != 'up', '回さずに読めてしまった (横倒しのはず)'
    assert sum(1 for k, *_ in marks if k == 'item') >= 10


# --- 取り置いた目印で，全紙面の水準を守る (slow) ---------------------------
#
# OCR は 1 枚 6〜180 秒かかるので，**目印と塊を取り置いて**箱の作り方だけを測ります．
# 取り置きの置き場は `COMPTEA_TF_CACHE`．2026-09-12 に上げた水準を，ここで守ります．


def _cards():
    import glob
    import json
    out = {}
    for p in sorted(glob.glob(os.path.join(CACHE, '*.json'))):
        out[os.path.basename(p)[:-5]] = json.load(open(p, encoding='utf-8'))
    return out


def _boxes(card):
    pts = [(m[1], m[2]) for m in card['marks'] if m[0] == 'item']
    blobs = [tuple(b) for b in card['blobs']]
    return tf.boxes_from(pts, blobs, tuple(card['size']))


@pytest.mark.slow
def test_取り置きで本のページの当たりを守る():
    """格子の外接矩形と IoU >= 0.7 の枚数 (2026-09-12 に 53 → 64)

    **画像が要る分 (隙間で伸ばす・表らしい範囲だけ採る) はここでは効きません**．
    取り置きだけで測れる水準 (56 枚) を守ります．画像も使うと 64 枚．
    """
    pd = pytest.importorskip('pandas')
    if not CACHE or not GRIDS:
        pytest.skip('取り置きか格子が無い')
    cards = _cards()
    if len(cards) < 50:
        pytest.skip('取り置きが足りない')
    cls = ('comp', 'sname', 'species_col', 'layer',
           'header_value', 'header_item', 'header_item_ja')
    hit = n = 0
    for name, c in cards.items():
        f = os.path.join(GRIDS, name, 'located.csv')
        if name.startswith('s01115_') or not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d = d[d.obj_name.isin(cls)]
        if d.empty:
            continue
        t = (float(d.x1.min()), float(d.y1.min()),
             float(d.x2.max()), float(d.y2.max()))
        n += 1
        if max([_iou(b, t) for b in _boxes(c)], default=0.0) >= 0.7:
            hit += 1
    assert n >= 70, f'本のページが {n} 枚しかない'
    assert hit >= 56, f'当たりが {hit}/{n} 枚に減った (2026-09-12 は 56)'


@pytest.mark.slow
def test_取り置きで折込の表の数を守る():
    """切り分けの表の数と合った枚数 (2026-09-12 に 14 → 19)"""
    import glob
    if not CACHE or not PARTS:
        pytest.skip('取り置きか切り出しが無い')
    truth = {}
    for p in glob.glob(os.path.join(PARTS, 's01115_*.png')):
        nm = os.path.basename(p)[:-4]
        if not nm.endswith('_note'):
            truth[nm.split('_p')[0]] = truth.get(nm.split('_p')[0], 0) + 1
    cards = _cards()
    got = {k: len(_boxes(c)) for k, c in cards.items() if k.startswith('s01115_')}
    if len(got) < 20 or len(truth) < 20:
        pytest.skip('取り置きか真値が足りない')
    ok = sum(1 for k, v in truth.items() if got.get(k) == v)
    assert ok >= 19, (f'表の数が合った折込が {ok}/{len(truth)} 枚に減った '
                      '(2026-09-12 は 19)')


# --- 折込は表が横に並ぶ ----------------------------------------------------
#
# 2026-09-13 に実データで確かめた (ユーザの問い「折込では 2 段組がないとしたら，
# 学名や和名で横を区切れますか」への答え):
#
#   **折込 23 枚のうち 19 枚で，表が横に並んでいる**．表頭のまとまりの x が
#   はっきり 2 群以上に分かれる (例: s01115_01 は x = 1102 と 4742．幅 9344)．
#   同じ列に縦に積まれた表の x の差は 60〜100 px しかないので，
#   **区切りに使うのは紙面の幅の 2 割以上離れた所だけ**．
#
#   **表頭の無い表は 67 表中 1 表だけ** (11_p1 の総合表)．
#   その列は「1 回出現の種」で数える．

def test_横に並ぶ表を_x_で分ける():
    """幅の 2 割以上離れていれば別の列"""
    w = 10000
    pts = [(1000, 100), (1050, 200), (1100, 300),      # 左の表
           (5000, 100), (5050, 200), (5100, 300)]      # 右の表
    got = tf.columns(pts, w)
    assert len(got) == 2
    assert all(len(g) == 3 for g in got)


def test_近い表頭は同じ列():
    """縦に積まれた表の表頭は x が 100 px ほどしか違わない"""
    w = 10000
    pts = [(1000, 100), (1050, 200), (1100, 5000), (1090, 5100)]
    assert len(tf.columns(pts, w)) == 1


def test_列ごとに縦のまとまりを数える():
    w, h = 10000, 12000
    pts = [(1000, 100), (1050, 200), (1000, 300),      # 左上
           (1000, 7000), (1050, 7100), (1000, 7200),   # 左下
           (5000, 100), (5050, 200), (5000, 300)]      # 右
    assert tf.count_tables(pts, (w, h)) == 3


def test_表頭が無ければ出現1回の種で数える():
    """**表頭は非必須** (折込 67 表中 1 表は表頭が無い)"""
    w, h = 10000, 12000
    once = [(1000, 6000), (5000, 6000)]
    assert tf.count_tables([], (w, h), once=once) == 2


def test_表頭と出現1回の多い方を採る():
    w, h = 10000, 12000
    pts = [(1000, 100), (1050, 200), (1000, 300)]      # 表頭は 1 つ
    once = [(1000, 6000), (5000, 6000)]                # 下端は 2 つ
    assert tf.count_tables(pts, (w, h), once=once) == 2


def test_目印が無ければ_1():
    assert tf.count_tables([], (1000, 1000)) == 1


# --- 基本構造から表の箱を作る ----------------------------------------------
#
# 基本構造 (2026-09-13 ユーザ):
#   表題 → 表頭項目+値 → 学名・和名・組成 → 1 回出現の種 → 地点情報
# **表頭が表の上端**，**次の表の表頭の直前がこの表の下端**．
# 横は，**列と列の中間**で分ける (折込 23 枚中 19 枚で表が横に並ぶ)．

def test_横に並ぶ表を箱にする():
    w, h = 10000, 12000
    pts = [(1000, 100), (1050, 200), (1000, 300),
           (5000, 100), (5050, 200), (5000, 300)]
    got = tf.structure_boxes(pts, (w, h))
    assert len(got) == 2
    left, right = sorted(got, key=lambda b: b[0])
    assert left[0] == 0 and right[2] == w          # 紙面の端まで
    assert left[2] == right[0]                     # 中間で接する
    assert 1000 < left[2] < 5000


def test_縦に積まれた表を上下に分ける():
    w, h = 10000, 12000
    pts = [(1000, 100), (1050, 200), (1000, 300),        # 上の表
           (1000, 7000), (1050, 7100), (1000, 7200)]     # 下の表
    got = sorted(tf.structure_boxes(pts, (w, h)), key=lambda b: b[1])
    assert len(got) == 2
    assert got[0][3] == got[1][1]                  # 上の下端 = 下の上端
    assert got[0][1] < 100 and got[1][3] == h


def test_表頭が無ければ紙面の上から():
    """**表頭は非必須**．「1 回出現の種」しか無い列は，紙面の上から"""
    w, h = 10000, 12000
    got = tf.structure_boxes([], (w, h), once=[(1000, 6000)])
    assert len(got) == 1
    assert got[0][1] == 0 and got[0][3] == h


def test_出現1回の種は同じ表に入れる():
    """表の下端は，その表の「1 回出現の種」より下"""
    w, h = 10000, 12000
    pts = [(1000, 100), (1050, 200), (1000, 300),
           (1000, 7000), (1050, 7100), (1000, 7200)]
    once = [(1000, 6500)]                          # 上の表の下端
    got = sorted(tf.structure_boxes(pts, (w, h), once=once), key=lambda b: b[1])
    assert got[0][3] > 6500


def test_目印が無ければ紙面ぜんぶ():
    assert tf.structure_boxes([], (800, 600)) == [(0, 0, 800, 600)]


def test_箱の数は数えた枚数と合う():
    w, h = 10000, 12000
    pts = [(1000, 100), (1050, 200), (1000, 300),
           (1000, 7000), (1050, 7100), (1000, 7200),
           (5000, 100), (5050, 200), (5000, 300)]
    assert len(tf.structure_boxes(pts, (w, h))) == tf.count_tables(pts, (w, h))


# --- 箱をインクに合わせて縮める --------------------------------------------
#
# `structure_boxes` は紙面を隙間なく分ける (覆い 1.00・重なり 0.00)．
# 切り出しに使うには，**箱の中のインクの外接矩形**まで縮める．

def _ink_sheet(w, h, marks):
    """白地に黒い矩形を置いた紙面 (dark は True が黒)"""
    im = Image.new('L', (w, h), 255)
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in marks:
        d.rectangle([x1, y1, x2, y2], fill=0)
    return np.array(im) < 128


def test_インクの外接矩形まで縮める():
    dark = _ink_sheet(1000, 800, [(100, 200, 400, 600)])
    got = tf.shrink_to_ink(dark, (0, 0, 1000, 800), pad=0)
    assert got == (100, 200, 401, 601)


def test_余白を付けられる():
    dark = _ink_sheet(1000, 800, [(100, 200, 400, 600)])
    got = tf.shrink_to_ink(dark, (0, 0, 1000, 800), pad=10)
    assert got == (90, 190, 411, 611)


def test_紙面からはみ出さない():
    dark = _ink_sheet(1000, 800, [(0, 0, 50, 50)])
    got = tf.shrink_to_ink(dark, (0, 0, 1000, 800), pad=100)
    assert got == (0, 0, 151, 151)


def test_インクが無ければそのまま():
    dark = _ink_sheet(1000, 800, [])
    assert tf.shrink_to_ink(dark, (10, 20, 300, 400)) == (10, 20, 300, 400)


def test_箱の外のインクは見ない():
    dark = _ink_sheet(1000, 800, [(100, 100, 200, 200), (700, 600, 800, 700)])
    got = tf.shrink_to_ink(dark, (0, 0, 500, 500), pad=0)
    assert got == (100, 100, 201, 201)


# --- 列の境は白い隙間へ寄せる ----------------------------------------------
#
# 2026-09-13 に実データで分かった: 列の境を**表頭の中間**に置くと，
# **表頭は表の左端にある**ので，中間が左の表の本体の中に落ちる
# (19_p3 は幅 2823 px の表に対し，箱が 5627 px になった)．
# **2 つの表頭のあいだのいちばん広い白い隙間**へ寄せる．

def test_列の境を白い隙間へ寄せる():
    # 左の表 (x 100-900) と右の表 (x 1200-1900)．隙間は 900-1200
    dark = _ink_sheet(2000, 500, [(100, 50, 900, 450), (1200, 50, 1900, 450)])
    got = tf.refine_columns(dark, [0, 1000, 2000], [(100, 300), (1300, 1500)])
    assert 900 <= got[1] <= 1200
    assert got[0] == 0 and got[2] == 2000


def test_隙間が無ければ動かさない():
    dark = _ink_sheet(2000, 500, [(100, 50, 1900, 450)])
    got = tf.refine_columns(dark, [0, 1000, 2000], [(100, 300), (1300, 1500)])
    assert got[1] == 1000


def test_境が1本も無ければそのまま():
    dark = _ink_sheet(2000, 500, [(100, 50, 900, 450)])
    assert tf.refine_columns(dark, [0, 2000], [(100, 300)]) == [0, 2000]


def test_箱が白い隙間で切れる():
    """`split=True` のとき"""
    """`structure_boxes` に画像を渡すと，列の境が隙間へ寄る"""
    dark = _ink_sheet(2000, 600, [(100, 50, 900, 550), (1200, 50, 1900, 550)])
    pts = [(150, 60), (160, 100), (150, 140),
           (1250, 60), (1260, 100), (1250, 140)]
    got = sorted(tf.structure_boxes(pts, (2000, 600), dark=dark, split=True),
                 key=lambda b: b[0])
    assert len(got) == 2
    assert 900 <= got[0][2] <= 1200


# --- 表頭の無い表は，白い隙間で分ける --------------------------------------
#
# 2026-09-13 に実データで分かった: 箱の幅が広すぎるのは，**表頭の見つからない
# 表が列を作らない**ため．1 つの帯に 2 つの表が横に並んだまま入る．
# **折込に 2 段組は無い** (ユーザ確認) ので，帯の中の**縦に長い白い隙間**は
# 表と表の境とみなしてよい．

def test_帯の中の白い隙間で分ける():
    # 1 つの帯 (x 0-2000) に，2 つの表 (100-900 と 1200-1900)
    dark = _ink_sheet(2000, 1000, [(100, 50, 900, 950), (1200, 50, 1900, 950)])
    got = tf.split_at_gutters(dark, (0, 0, 2000, 1000))
    assert len(got) == 2
    assert got[0][2] <= 1200 and got[1][0] >= 900


def test_1つの表は分けない():
    dark = _ink_sheet(2000, 1000, [(100, 50, 1900, 950)])
    assert tf.split_at_gutters(dark, (0, 0, 2000, 1000)) == [(0, 0, 2000, 1000)]


def test_狭い隙間では分けない():
    """列と列のあいだ (組成の中の隙間) では分けない"""
    dark = _ink_sheet(2000, 1000, [(100, 50, 980, 950), (1000, 50, 1900, 950)])
    got = tf.split_at_gutters(dark, (0, 0, 2000, 1000), min_w=0.05)
    assert len(got) == 1


def test_短い隙間では分けない():
    """帯の高さの一部にしか無い隙間は，表の境ではない"""
    dark = _ink_sheet(2000, 1000, [(100, 50, 1900, 500), (100, 600, 900, 950),
                                   (1200, 600, 1900, 950)])
    got = tf.split_at_gutters(dark, (0, 0, 2000, 1000))
    assert len(got) == 1


def test_頼まれたときだけ隙間で分ける():
    """**隙間での分割は既定では使わない**

    2026-09-13 の実測: **紙面 23 枚中 11 枚に，端から端まで通る隙間が無い**
    (表が互い違いに並ぶため)．当てると分けすぎる (68 → 89 箱)．
    """
    dark = _ink_sheet(2000, 1000, [(100, 50, 900, 950), (1200, 50, 1900, 950)])
    pts = [(150, 60), (160, 100), (150, 140)]      # 表頭は左の表だけ
    assert len(tf.structure_boxes(pts, (2000, 1000), dark=dark)) == 1
    got = sorted(tf.structure_boxes(pts, (2000, 1000), dark=dark, split=True),
                 key=lambda b: b[0])
    assert len(got) == 2
    assert got[0][2] <= 1200 and got[1][0] >= 900


# --- 切り分けの検算 --------------------------------------------------------
#
# 2026-09-13 の実測で分かったこと: **目印だけで切り出しを作り直すのは無理**．
# 紙面 23 枚のうち 11 枚に「端から端まで通る白い隙間」が無く，
# 表は互い違いに並ぶ．**幾何の切り分け (21/23) を置き換えられない**．
# そこで**検算**に使う: 切り分けた箱に表頭がちょうど 1 つ入るかを見る．

def test_箱に表頭が1つなら_ok():
    pts = [(150, 60), (160, 100), (150, 140)]
    got = tf.check_boxes([(0, 0, 1000, 1000)], pts, (2000, 2000))
    assert got[0]['heads'] == 1 and got[0]['ok']


def test_表頭が2つなら切り足りない():
    pts = [(150, 60), (160, 100), (150, 140),
           (150, 1500), (160, 1540), (150, 1580)]
    got = tf.check_boxes([(0, 0, 1000, 2000)], pts, (2000, 2000))
    assert got[0]['heads'] == 2 and not got[0]['ok']
    assert '切り足りない' in got[0]['why']


def test_表頭が無ければ知らせる():
    got = tf.check_boxes([(0, 0, 1000, 1000)], [], (2000, 2000))
    assert got[0]['heads'] == 0 and not got[0]['ok']
    assert '表頭が無い' in got[0]['why']


def test_出現1回の種の数も返す():
    pts = [(150, 60), (160, 100), (150, 140)]
    once = [(200, 800)]
    got = tf.check_boxes([(0, 0, 1000, 1000)], pts, (2000, 2000), once=once)
    assert got[0]['once'] == 1


def test_箱の外の目印は数えない():
    pts = [(1500, 60), (1510, 100), (1500, 140)]
    got = tf.check_boxes([(0, 0, 1000, 1000)], pts, (2000, 2000))
    assert got[0]['heads'] == 0


# --- 切り足りない箱は，表頭のあいだで切る ----------------------------------
#
# 2026-09-13 の検算で見つかった 3 箱を調べると，2 つの表頭は
#   04: x も y も離れる     14・21: **y だけ離れる (縦に積まれている)**
# だった．紙面ぜんぶの隙間で切ると分けすぎる (箱 1 つが 15 分割) ので，
# **2 つの表頭のあいだ**だけを見る．

def test_縦に積まれた表頭のあいだで切る():
    box = (0, 0, 1000, 4000)
    groups = [[(500, 300), (510, 340), (500, 380)],      # 上の表頭
              [(500, 2000), (510, 2040), (500, 2080)]]   # 下の表頭
    got = tf.split_between_heads(box, groups)
    assert len(got) == 2
    assert got[0][3] == got[1][1]
    assert 380 < got[0][3] < 2000


def test_横に並んだ表頭のあいだで切る():
    box = (0, 0, 4000, 1000)
    groups = [[(300, 100), (310, 140), (300, 180)],
              [(2500, 100), (2510, 140), (2500, 180)]]
    got = tf.split_between_heads(box, groups)
    assert len(got) == 2
    assert got[0][2] == got[1][0]
    assert 310 < got[0][2] < 2500


def test_表頭が1つなら切らない():
    box = (0, 0, 1000, 1000)
    groups = [[(500, 300), (510, 340), (500, 380)]]
    assert tf.split_between_heads(box, groups) == [box]


def test_白い隙間へ寄せる():
    """画像を渡すと，境をインクの無いところへ寄せる"""
    dark = _ink_sheet(1000, 4000, [(100, 50, 900, 900),
                                   (100, 1500, 900, 3900)])
    box = (0, 0, 1000, 4000)
    groups = [[(500, 300), (510, 340), (500, 380)],
              [(500, 2000), (510, 2040), (500, 2080)]]
    got = tf.split_between_heads(box, groups, dark=dark)
    assert 900 <= got[0][3] <= 1500


def test_3つの表頭なら3つに切る():
    box = (0, 0, 1000, 6000)
    groups = [[(500, 300), (510, 340), (500, 380)],
              [(500, 2000), (510, 2040), (500, 2080)],
              [(500, 4000), (510, 4040), (500, 4080)]]
    assert len(tf.split_between_heads(box, groups)) == 3


# --- 表頭の無い箱は，切り出して読み直す ------------------------------------
#
# 2026-09-13 の検算で「表頭が無い」と出た 6 箱を調べると，
#   2 箱は**紙面の表題** (`Tab.107 日本植生誌一近畿 付表`)．表ではない
#   4 箱は**本物の表**で，格子には表頭項目が 11〜20 個ある
# だった．**紙面ぜんぶを読んだときに目印が拾えなかっただけ**なので，
# 箱を切り出して読み直す (小さくすると読める — タイルや注記と同じ)．

class _MarkReader:
    """切り出しを読むと目印を返す偽の読み手"""

    def __init__(self, marks):
        self.marks = marks
        self.calls = 0

    def readtext(self, *a, **kw):
        self.calls += 1
        return self.marks


def test_表頭の無い箱だけ読み直す(monkeypatch):
    im = Image.new('RGB', (2000, 2000), 'white')
    pts = [(100, 100), (110, 140), (100, 180)]
    seen = []

    def fake(crop, reader=None, **kw):
        seen.append(crop.size)
        # `read_marks` は (目印, 採った向き) を返す
        return ([('item', 50.0, 50.0, '通し番号'),
                 ('item', 60.0, 90.0, '調査番号'),
                 ('item', 50.0, 130.0, '調査地')], 'up')

    monkeypatch.setattr(tf, 'read_marks', fake)
    boxes = [(0, 0, 1000, 1000), (1000, 0, 2000, 1000)]
    got = tf.recheck_boxes(im, boxes, pts, (2000, 2000))
    assert len(seen) == 1                      # 表頭のある箱は読み直さない
    assert got[0]['heads'] == 1 and got[1]['heads'] == 1


def test_読み直しても見つからなければそのまま(monkeypatch):
    im = Image.new('RGB', (2000, 2000), 'white')
    monkeypatch.setattr(tf, 'read_marks', lambda *a, **k: ([], 'up'))
    got = tf.recheck_boxes(im, [(0, 0, 1000, 1000)], [], (2000, 2000))
    assert got[0]['heads'] == 0 and not got[0]['ok']


def test_読み直した目印は元の座標に戻す(monkeypatch):
    im = Image.new('RGB', (2000, 2000), 'white')

    def fake(crop, reader=None, **kw):
        return ([('item', 50.0, 50.0, '通し番号'),
                 ('item', 60.0, 90.0, '調査番号'),
                 ('item', 50.0, 130.0, '調査地')], 'up')

    monkeypatch.setattr(tf, 'read_marks', fake)
    got = tf.recheck_boxes(im, [(1000, 500, 2000, 1500)], [], (2000, 2000))
    g = got[0]['groups'][0]
    assert all(1000 <= p[0] < 2000 and 500 <= p[1] < 1500 for p in g)
