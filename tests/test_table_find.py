"""検出器を使わずに組成表全体を見つける (comptea/table_find.py)

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
    """`find_tables` の間引き．横一列の 3 語だけのまとまりは落ちる"""
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
    boxes = tf.find_tables(im, reader=FakeReader(), tile=40, least=1)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 40 and y1 <= 60 and x2 >= 100 and y2 >= 128


def test_横一列の目印だけでは箱を出さない():
    """表頭が文章の紙面や本文の語 (kinki_026 型) を落とす"""
    im = _sheet(size=(400, 300), rect=(40, 60, 100, 68))
    assert tf.find_tables(im, reader=FakeReader(), tile=150, least=1) == []


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
    boxes = tf.find_tables(Image.open(jpg))
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
    boxes = tf.find_tables(im)
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
    """格子の外接矩形と IoU >= 0.7 の枚数 (2026-09-12 に 53 → 56)

    塊が掴めない紙面を隙間で伸ばす分 (+2) は画像が要るので，ここでは数えません．
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
