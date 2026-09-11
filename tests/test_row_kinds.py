"""行の種類を見分けて note に付ける (row_kinds.py．案 e の段階 3)

行は落とさない．落とすと真値との一致 (recall 1.000) が下がるので，**印だけ**を付ける．
"""
import numpy as np
import pandas as pd
from PIL import Image

from comptea import row_kinds


X_SN = [0, 300]                    # 学名
X_JA = [310, 560]                  # 和名
X_LAYER = [570, 610]               # 階層
X_COMP = [620, 660, 700, 740]      # 組成 3 列
EDGES = list(range(100, 100 + 34 * 12 + 1, 34))


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _text(px, x1, x2, y1, y2, w=7, gap=4):
    x = int(x1)
    while x + w <= int(x2):
        _fill(px, x, x + w, y1, y2)
        x += w + gap


def _grid(edges, x_edges, obj_name, block=1):
    rows = []
    for c, (xa, xb) in enumerate(zip(x_edges[:-1], x_edges[1:]), 1):
        for r, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
            rows.append(dict(x1=float(xa), x2=float(xb), y1=float(ya), y2=float(yb),
                             obj_name=obj_name, note='', block=block, row=r, col=c))
    return pd.DataFrame(rows)


def _df(edges=EDGES):
    return pd.concat([_grid(edges, X_SN, 'sname'), _grid(edges, X_JA, 'species_col'),
                      _grid(edges, X_LAYER, 'layer'), _grid(edges, X_COMP, 'comp')],
                     ignore_index=True)


def _sheet(edges=EDGES, headings=(), name_only=(), legends=(), blank_ja=(),
           size=(800, 700)):
    """種の行・見出し行・学名だけの行・凡例を描いた紙面

    種の行: 学名・和名・階層・組成の「・」がそろう．
    見出し行 (`headings`): 学名側だけに字があり，下に長い下線がある．
    学名だけの行 (`name_only`): 学名だけ (下線なし)．次の行に和名と値がある．
    凡例 (`legends`): 学名から組成まで 1 つながりの文が流れる．
    """
    img = Image.new('L', size, 255)
    px = img.load()
    for i, (ya, yb) in enumerate(zip(edges[:-1], edges[1:]), 1):
        c = int((ya + yb) // 2)
        if i in headings:
            _text(px, 10, 250, c - 8, c + 8)
            _fill(px, 10, 290, c + 11, c + 13)              # 280 px の下線
            continue
        if i in name_only:
            _text(px, 10, 250, c - 8, c + 8)
            continue
        if i in legends:
            _fill(px, 10, 730, c - 8, c + 8)                # 列をまたぐ 1 つの塊
            continue
        _text(px, 10, 250, c - 8, c + 8)
        if i not in blank_ja:
            _text(px, 320, 550, c - 8, c + 8)
        _fill(px, X_LAYER[0] + 10, X_LAYER[0] + 25, c - 8, c + 8)
        for xa, xb in zip(X_COMP[:-1], X_COMP[1:]):
            xc = (xa + xb) // 2
            _fill(px, xc - 2, xc + 3, c - 3, c + 4)
    return img


def _skewed(df, img, dys=(0, 15, 30)):
    """組成の列ごとに y を `dys` だけずらす (傾き補正後の格子の形)．紙面の「・」も同じだけずらす

    傾き補正 (`row_skew`・`row_track.fix_offsets`) のあとは同じ行でも列ごとに y が
    違う (実データでは行の高さの 1/4〜2/3．03_p2 は 35 px の行で 23 px)．
    行の全セルの y の min/max で帯を作ると隣の行の字が入る
    """
    out = df.copy()
    px = img.load()
    for c, dy in enumerate(dys, 1):
        m = (out.obj_name == 'comp') & (out.col == c)
        out.loc[m, ['y1', 'y2']] += float(dy)
    for i, (ya, yb) in enumerate(zip(EDGES[:-1], EDGES[1:]), 1):
        cy = int((ya + yb) // 2)
        for (xa, xb), dy in zip(zip(X_COMP[:-1], X_COMP[1:]), dys):
            xc = (xa + xb) // 2
            for x in range(xc - 2, xc + 3):
                for y in range(cy - 3, cy + 4):
                    if px[x, y] == 0:
                        px[x, y] = 255
            if px[10, cy] == 0 and px[320, cy] == 0:     # 種の行だけ「・」を打つ
                _fill(px, xc - 2, xc + 3, cy + dy - 3, cy + dy + 4)
    return out, img


def test_列ごとに_y_がずれても種の行に印を付けない():
    """傾き補正後の格子で，種の行が見出しや学名だけの行にされないこと

    帯 (行の全セルの min/max) だけで測ると，行の高さが水増しされて黒画素の
    中央値が上がり，本物の種の行が「本体が空」になる (14_p2 で 11 行)．
    セルごとだけで測ると，箱が字から外れている表で逆に落ちる (17_p1 で 10 行)．
    **どちらかに字があれば空でない**とするので，印は減る方向にしか動かない
    (見出しを見落とすことはあるが，種の行を落とすより安全)．
    """
    df, img = _skewed(_df(), _sheet(headings=(4,)))
    k = _kinds(img, df)
    assert all(v == '' for r, v in k.items() if r != 4)


def _kinds(img, df):
    out, _warns = row_kinds.mark_rows(img, df)
    k = (out[out.obj_name == 'sname'].sort_values('row')
         .set_index('row')['note'].fillna(''))
    return {r: v for r, v in k.items()}


def test_種の行には印を付けない():
    kinds = _kinds(_sheet(), _df())
    assert all(v == '' for v in kinds.values())


def test_見出し行に印を付ける():
    kinds = _kinds(_sheet(headings=(1, 7)), _df())
    assert 'heading' in kinds[1] and 'heading' in kinds[7]
    assert all('heading' not in kinds[r] for r in (2, 3, 8))


def test_学名だけの行は見出しと区別する():
    # 下線が無く，次の行に和名と値がある = 前の行から続く学名 (見出しではない)
    kinds = _kinds(_sheet(name_only=(5,)), _df())
    assert 'name_only' in kinds[5] and 'heading' not in kinds[5]


def test_凡例に印を付ける():
    kinds = _kinds(_sheet(legends=(12,)), _df())
    assert 'legend' in kinds[12]
    assert all('legend' not in kinds[r] for r in (2, 5, 9))


def test_複数階層で和名が空く行は種の行のまま():
    kinds = _kinds(_sheet(blank_ja=(3, 6, 9)), _df())
    assert all(v == '' for v in kinds.values())


def test_印は全クラスのセルに付く():
    out, _w = row_kinds.mark_rows(_sheet(headings=(4,)), _df())
    for cls in ('sname', 'species_col', 'layer', 'comp'):
        note = out[(out.obj_name == cls) & (out.row == 4)]['note'].fillna('')
        assert note.str.contains('heading').all()


def test_行も列も落とさない():
    df = _df()
    out, _w = row_kinds.mark_rows(_sheet(headings=(1,), legends=(12,)), df)
    assert len(out) == len(df)
    assert out['row'].nunique() == df['row'].nunique()
    assert np.array_equal(out.sort_index()[['y1', 'y2']].values,
                          df.sort_index()[['y1', 'y2']].values)


def test_警告に数を出す():
    _out, warns = row_kinds.mark_rows(_sheet(headings=(1, 7), legends=(12,)), _df())
    assert warns and any('見出し 2' in w for w in warns)


def test_組成の枠線は字と数えない():
    # 見出し (行 6) の帯に，その下の区分種 (行 7〜9) を囲む枠線の上辺と縦線が
    # 入り込んでいる紙面 (活字の 010・066 型)．枠線は段の全高で消してから見る
    img = _sheet(headings=(6,))
    px = img.load()
    ya, yb = EDGES[5], EDGES[9]
    _fill(px, X_COMP[0], X_COMP[-1], ya + 2, ya + 4)        # 枠の上辺 (横線)
    _fill(px, X_COMP[0], X_COMP[0] + 2, ya, yb)             # 枠の左辺 (縦線)
    _fill(px, X_COMP[-1] - 2, X_COMP[-1], ya, yb)           # 枠の右辺 (縦線)
    assert 'heading' in _kinds(img, _df())[6]


def test_出現1回の種の見出しを見分ける():
    """2026-09-11 ユーザ指示: 見出しはそのまま書かれている．1 は漢数字とアラビア数字"""
    from comptea.row_kinds import kind_of_text
    # **語順の揺れもある** (2026-09-11 ユーザ指示)
    for t in ('出現1回の種', '出現一回の種', '出現 1 回 の 種', '出現１回の種',
              '一回出現の種', '1回出現の種', '一 回 出現 の 種',
              # **OCR の読み違いにも当てる** (「種」を「稲」と読む，字のあいだに
              # ハイフンが入る)．「の種」は求めない
              '出現-一回の稲', '出現1回', '一回出現'):
        assert kind_of_text(t) == 'once', t
    # 独文だけの紙面もある (OCR は ss と読むことがある)
    assert kind_of_text('Ausserdem je einmal in Lfd. Nr. 1') == 'once'
    # **見出しの中には種名も並ぶので，見出しを先に見る**
    assert kind_of_text('出現1回の種 in Lfd.Nr.1: Osmunda japonica ゼンマイ K-+') == 'once'


def test_種の行と表頭の項目名():
    from comptea.row_kinds import kind_of_text
    assert kind_of_text('ヤマルリソウ') == 'species'
    assert kind_of_text('Omphalodes japonica ヤマルリソウ K') == 'species'
    assert kind_of_text('調査年月日') == 'item'
    assert kind_of_text('') == 'other'


def test_ラテン語の語は階級の略語とローマ数字を数えない():
    """長い学名の「var. intermedium」が和名の欄へはみ出しても，流し込みに見せない"""
    from comptea.row_kinds import latin_words
    assert latin_words('var, intermedium') == ['intermedium']
    assert latin_words('III IV V Carex lenta') == ['Carex', 'lenta']
    assert len(latin_words('Avena fatua カラスムギ Cardamine flexuosa')) == 4


def test_表の下の注記を見分ける():
    """調査地・出典の注記は種の行ではない (下端を多めに取ると足されていた)"""
    from comptea.row_kinds import kind_of_text
    assert kind_of_text('Lage d. Aufn. 調査地：Fluß Takeda, Hyogo-Präf. 兵庫県') == 'note'
    assert kind_of_text('Nachweis d. Vegetationsaufnahmen 既発表資料名') == 'note'
    assert kind_of_text('Quercus glauca アラカシ K') == 'species'
    # 学名の中の「datum」(cuspidatum) を注記の「Datum」と取らない (kinki_060)
    assert kind_of_text('Polygonum cuspidatum イタドリ') == 'species'
