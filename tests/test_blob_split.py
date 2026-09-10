"""縮小した塊で紙面を切り分ける (split_sheet.blob_boxes・split_by_blobs)

空白の帯は紙面を貫く必要があるので，表が階段状に置かれた紙面では通らない
(s01115_09 は Tab.50 の注記が右へ伸びているだけで縦の帯が消え，5 表が 3 つにしか
切れなかった)．**縮めれば表 1 つが 1 つの塊になる** (2026-09-10 ユーザ提案)．
"""
import numpy as np

from comptea import split_sheet as ss


W, H = 1600, 1200


def _table(dark, x1, y1, x2, y2, pitch=8):
    """表らしい黒画素 (行が pitch ごとに並ぶ) を置く"""
    for y in range(int(y1), int(y2), pitch):
        dark[y:y + 5, int(x1):int(x2)] = True


def _sheet():
    return np.zeros((H, W), dtype=bool)


def test_縮小は平均でなく1画素でもあれば黒():
    # 32x32 に 4 画素だけ黒 (「・」だけの組成部)．平均では白に埋もれる
    dark = np.zeros((32, 32), dtype=bool)
    dark[4:6, 4:6] = True
    small = ss.shrink_ink(dark, scale=32, min_ink=4)
    assert small.shape == (1, 1) and small[0, 0] == 1
    # 黒が足りなければ落とす (点状の汚れ)
    dark2 = np.zeros((32, 32), dtype=bool)
    dark2[4, 4] = True
    assert ss.shrink_ink(dark2, scale=32, min_ink=4)[0, 0] == 0


def test_離れた2つの表は別の塊になる():
    dark = _sheet()
    _table(dark, 100, 100, 700, 1100)
    _table(dark, 900, 100, 1500, 1100)
    boxes = ss.blob_boxes(dark)
    assert len(boxes) == 2
    assert boxes[0][2] < boxes[1][0]


def _narrow_gap():
    """表と表の隙間が，帯の下限 (120 px) より細い紙面 (s01115_09 は 64 px)"""
    dark = _sheet()
    _table(dark, 60, 60, 700, 1140)
    _table(dark, 764, 60, 1540, 700)
    return dark


def test_隙間が帯の下限より細くても塊なら切れる():
    dark = _narrow_gap()
    box = (0, 0, W, H)
    # 帯では切れない (隙間 64 px < 120 px)
    gutter = max(int(W * ss.MIN_GUTTER), ss.MIN_BAND_PX)
    gap = max(int(H * ss.MIN_GAP), ss.MIN_BAND_PX)
    assert ss._split_box(dark, box, gutter, gap) == [box]
    # 塊なら切れる (縮小の 1 ブロック = 8 px で見るため)
    out = ss.split_by_blobs(dark, box)
    assert len(out) == 2
    assert out[0][2] <= out[1][0]


def test_表の下の注記は箱に残す():
    """注記には調査地・調査年月日が書いてある (表頭に無い地点の情報の出どころ)"""
    dark = _narrow_gap()
    _table(dark, 60, 1160, 700, 1180)         # 左の表の下の注記 (表とは離れている)
    out = ss.split_by_blobs(dark, (0, 0, W, H))
    assert len(out) == 2
    left = min(out, key=lambda b: b[0])
    assert left[3] >= 1180                     # 注記まで含む
    right = max(out, key=lambda b: b[0])
    assert right[3] < 800                      # 下に何も無ければ塊のまま (範囲の下端まで伸ばさない)


def test_離れた次の表の表題は取り込まない():
    """範囲の下端まで伸ばすと，次の表の表題を巻き込む (23_p2 が Tab.149 を取り込んだ)"""
    table = (60, 60, 1540, 700)
    note = (60, 715, 1540, 760)                # 表のすぐ下の注記
    title = (60, 1050, 900, 1100)              # 離れた次の表の表題
    out = ss._reach_down([table], 1200, [table, note, title])
    assert out[0][3] == 760                    # 注記は含み，表題は含まない


def test_下に何も無ければ塊のまま():
    table = (60, 60, 1540, 700)
    assert ss._reach_down([table], 1200, [table])[0][3] == 700


def test_横に重ならない塊では止まらない():
    table = (60, 60, 700, 700)
    other = (900, 715, 1540, 760)              # 右にずれた別の表
    assert ss._reach_down([table], 1200, [table, other])[0][3] == 700


def test_真下に表があればその手前で止める():
    dark = _sheet()
    _table(dark, 60, 60, 700, 400)
    _table(dark, 60, 700, 700, 1100)
    _table(dark, 900, 60, 1540, 1100)
    out = ss.split_by_blobs(dark, (0, 0, W, H))
    assert len(out) == 3
    top = min((b for b in out if b[0] < 800), key=lambda b: b[1])
    assert 400 <= top[3] <= 700                # 下の表には食い込まない


def test_塊が覆っていれば切らない():
    # 1 つの表 (中の隙間は縮小で埋まる)
    dark = _sheet()
    _table(dark, 40, 40, 1560, 1160)
    box = (0, 0, W, H)
    assert ss.split_by_blobs(dark, box) == [box]


def test_小さい断片は切り離さない():
    # 表 1 つと，隅の札 (紙面の 1% ほど)
    dark = _sheet()
    _table(dark, 100, 200, 1500, 1100)
    _table(dark, 40, 40, 200, 100)
    box = (0, 0, W, H)
    assert ss.split_by_blobs(dark, box) == [box]


def test_字の段落ごとに割れる紙面では切らない():
    # 細い切れ端 (16_p1・21_p2 の型)．塊が上限より多い
    dark = _sheet()
    for i in range(6):
        _table(dark, 60 + i * 250, 400, 200 + i * 250, 800)
    box = (0, 0, W, H)
    assert ss.split_by_blobs(dark, box) == [box]


def test_find_tablesで塊の切り分けを外せる():
    dark = _narrow_gap()
    assert len(ss.find_tables(dark, use_blob=False)) == 1
    assert len(ss.find_tables(dark)) == 2


def test_塊は左から上への順で返す():
    dark = _sheet()
    _table(dark, 900, 100, 1500, 500)
    _table(dark, 100, 600, 700, 1100)
    boxes = ss.blob_boxes(dark)
    assert [b[0] for b in boxes] == sorted(b[0] for b in boxes)
