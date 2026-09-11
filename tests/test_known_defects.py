"""ユーザ目視で見つかった崩れを真値にしたテスト (2026-09-11)

**テスト駆動で直すための的**．5 つの課題を，実データの格子で確かめる．

    1. 階層の欠落
    2. 組成の欠落
    3. 表頭の項目の分割不足
    4. 表頭の項目の行区切りの誤認識 (境が項目名の字を割る)
    5. 1 回出現の種を組成と誤認識

格子は作るのが重いので，**作り置きの格子**を読む．環境変数 `COMPTEA_GRIDS` に
`located.csv` が並ぶ置き場 (`<置き場>/<表の名前>/located.csv`) を渡すと走り，
無ければ飛ばす．画像は `COMPTEA_PARTS` (折込の切り出し) と
`COMPTEA_SCAN` (本のページ) から探す．

    $env:COMPTEA_GRIDS = "...\\grids\\work_s2"
    $env:COMPTEA_PARTS = "...\\grids\\parts4"
    $env:COMPTEA_SCAN  = "D:\\matu\\scan"
    py -3.12 -m pytest tests/test_known_defects.py -q
"""
import os

import numpy as np
import pytest

pd = pytest.importorskip('pandas')
PIL = pytest.importorskip('PIL')
from PIL import Image                           # noqa: E402

GRIDS = os.environ.get('COMPTEA_GRIDS')
PARTS = os.environ.get('COMPTEA_PARTS', '')
SCAN = os.environ.get('COMPTEA_SCAN', '')

pytestmark = pytest.mark.skipif(
    not GRIDS or not os.path.isdir(GRIDS),
    reason='実データの格子が無い (COMPTEA_GRIDS を指すと走る)')


def grid(name):
    """格子 (located.csv)．無ければ飛ばす"""
    f = os.path.join(GRIDS or '', name, 'located.csv')
    if not os.path.exists(f):
        pytest.skip(f'{name} の格子が無い')
    return pd.read_csv(f)


def image(name, d=None):
    """**格子を作ったのと同じ画像** (`comptea.source`)

    工程は傾きを直した画像や横倒しを起こした画像で検出するので，格子の座標は
    元画像とずれる．元画像と突き合わせると，20_p3 では「境が 12 本字を割る」と
    誤って出た (正しい画像では 0 本)．
    """
    from comptea import source
    work = os.path.join(GRIDS or '', name)
    back = None
    for base, ext in ((PARTS, '.png'), (SCAN, '.jpg')):
        p = os.path.join(base, name + ext)
        if base and os.path.exists(p):
            back = p
            break
    try:
        im = source.open_image(d, work, fallback=back)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    bad = source.check(d, im)
    assert not bad, bad[0]
    return im


def pitch_of(d):
    comp = d[d.obj_name == 'comp']
    return float(np.median(comp.y2 - comp.y1))


def item_lines(d, im):
    """項目名の列の**字の行** (罫線を消した投影の連なり)"""
    from comptea import ink, row_track
    ja = d[d.obj_name == 'header_item_ja']
    if ja.empty:
        pytest.skip('表頭の項目名が無い')
    dark = ink.binarize(im)
    p = pitch_of(d)
    x1, x2 = int(ja.x1.min()), int(ja.x2.max())
    y1, y2 = int(ja.y1.min()), int(ja.y2.max())
    clean = row_track.clean_rules(dark, x1, x2, y1, y2, p)
    # **項目名の字のある x の範囲だけで数える** (黒画素の 2〜98% 点)．値の側から
    # はみ出した 1 つの数字 (19_p1 の「25」) を項目名の行と数えない
    col = clean.sum(axis=0).astype(float)
    if col.sum() > 0:
        cum = np.cumsum(col) / col.sum()
        lo = max(0, int(np.searchsorted(cum, 0.02)) - 4)
        hi = min(clean.shape[1], int(np.searchsorted(cum, 0.98)) + 5)
        clean = clean[:, lo:hi]
    prof = clean.sum(axis=1).astype(float)
    if not (prof > 0).any():
        pytest.skip('項目名の列に字が無い')
    lo = max(1.0, float(np.median(prof[prof > 0])) * 0.1)
    runs, st = [], None
    for i, v in enumerate(prof):
        if v > lo:
            st = i if st is None else st
        elif st is not None:
            if i - st >= 3:
                runs.append((y1 + st, y1 + i))
            st = None
    if st is not None:
        runs.append((y1 + st, y1 + len(prof)))
    # **凡例の行は項目名ではない**．表頭の上の凡例 (22_p2 の「1: Dicranopteris
    # dichotoma …」) は値の側まで文が流れる．項目名の右端より先に字が続く行は除く
    hv = d[d.obj_name == 'header_value']
    if len(hv):
        vx1 = int(hv.x1.min())
        keep = []
        for a, b in runs:
            band = dark[int(a):int(b), x2:max(x2 + 1, vx1 + 60)]
            keep.append((a, b) if band.size == 0 or band.mean() < 0.02
                        else None)
        runs = [r for r in keep if r is not None]
    # **触れ合った 2 行は谷で割る**．20_p1 の「海抜高」と「高木第 1 層の高さ」は
    # 字が触れて 61 px の 1 つの連なりになる．中央値の 1.4 倍より高い連なりは，
    # 山の半分より深い谷で割る (工程の `_resplit_tall` と同じ考え)
    if len(runs) >= 3:
        hmed = float(np.median([b - a for a, b in runs]))
        out = []
        for a, b in runs:
            seg = prof[a - y1:b - y1]
            if b - a > hmed * 1.4 and len(seg) > 6:
                k = int(np.argmin(seg[3:-3])) + 3
                if seg[k] < 0.5 * seg.max():
                    out += [(a, a + k), (a + k, b)]
                    continue
            out.append((a, b))
        runs = out
    return runs


# --- 1. 階層の欠落 ------------------------------------------------------

@pytest.mark.parametrize('name', ['s01115_17_p1'])
def test_階層の列が出る(name):
    """階層 (B1・B2・S・K) の列が格子にあること

    17_p1 は階層の記号が和名の列の x に食い込んでおり，和名の右端と組成の
    左端の隙間に収まらないため，階層が丸ごと格子の外に出ている．
    """
    d = grid(name)
    assert (d.obj_name == 'layer').any(), '階層の列が無い'


# --- 2. 組成の欠落 ------------------------------------------------------

# (表, 最小の行数)．真値はユーザの目視
ROWS_MIN = {
    's01115_13_p3': 104,        # 2026-09-11「組成が 1 行欠落」(いま 103)
    's01115_09_p5': 119,        # 2026-09-11 に直した 4 行を守る
}


@pytest.mark.parametrize('name,least', sorted(ROWS_MIN.items()))
def test_組成の行が欠けない(name, least):
    d = grid(name)
    comp = d[d.obj_name == 'comp']
    assert comp.row.nunique() >= least, (
        f'{name} の行が {comp.row.nunique()} 行しかない (真値 {least} 以上)')


# --- 3. 表頭の項目の分割不足 --------------------------------------------

HEAD_SHORT = ['s01115_16_p2', 's01115_17_p1', 's01115_18_p1', 's01115_18_p2',
              's01115_19_p1', 's01115_20_p1', 's01115_20_p2', 's01115_20_p3',
              's01115_21_p1', 's01115_21_p4', 's01115_22_p1', 's01115_22_p2',
              's01115_22_p3']


@pytest.mark.parametrize('name', HEAD_SHORT)
def test_表頭の項目の帯が足りる(name):
    """1 つの帯に項目名が 2 行以上入っていないこと

    2026-09-11 のユーザ目視「表頭の項目の行区切りが足りない」．隣り合う
    2 項目が 1 つの帯にまとまっている (20_p1 の「通し番号」と「調査番号」)．

    **区切りすぎは数えない** (2026-09-10 方針: 正確に区切れるなら区切りすぎる
    側に倒し，段階 3 で合成する)．かけら (点・濁点) を行と数えないよう，
    高さが行の中央値の 0.4 倍に満たない連なりは除く．
    """
    d = grid(name)
    im = image(name, d)
    ja = d[d.obj_name == 'header_item_ja']
    runs = item_lines(d, im)
    if len(runs) < 2:
        pytest.skip('項目名の行が少ない')
    hmed = float(np.median([b - a for a, b in runs]))
    runs = [(a, b) for a, b in runs if b - a >= hmed * 0.4]
    bands = (ja.drop_duplicates(['y1', 'y2'])[['y1', 'y2']]
             .sort_values('y1').values)
    bad = []
    for by1, by2 in bands:
        inside = [r for r in runs if by1 <= (r[0] + r[1]) / 2 <= by2]
        if len(inside) >= 2:
            bad.append((by1, by2, len(inside)))
    assert not bad, (
        f'{name} の帯 {len(bad)} 本に項目名が 2 行以上入っている '
        f'(帯 {len(bands)} 本・字の行 {len(runs)} 本): '
        + ', '.join(f'y{a:.0f}-{b:.0f} に {n} 行' for a, b, n in bad[:4]))


# --- 4. 表頭の項目の行区切りの誤認識 ------------------------------------

@pytest.mark.parametrize('name', HEAD_SHORT)
def test_表頭の帯が項目名の字を割らない(name):
    """帯の境が項目名の文字の途中を通らないこと

    2026-09-10 のユーザ定義「正確に行が区切れている = 1 つの文字の途中に
    境が無い」．20_p3 は 11 本が字を割っている．
    """
    d = grid(name)
    im = image(name, d)
    ja = d[d.obj_name == 'header_item_ja']
    runs = item_lines(d, im)
    edges = sorted(set(ja.y1.tolist()) | set(ja.y2.tolist()))
    split = [e for e in edges for a, b in runs if a + 2 < e < b - 2]
    assert not split, f'{name} の帯の境 {len(split)} 本が項目名の字を割っている'


# --- 5. 1 回出現の種を組成と誤認識 --------------------------------------

FLOW_TABLES = ['s01114_kinki_048', 's01114_kinki_077', 's01115_05_p1',
               's01115_05_p2', 's01115_10_p1', 's01115_16_p2',
               pytest.param('s01115_17_p1', marks=pytest.mark.xfail(
                   strict=True,
                   reason='末尾の帯が行の高さの 1.25 倍を超える．本物の行と見出しが'
                          '混ざった帯と区別できないので，工程は意図的に落とさない')),
               's01115_22_p1']


@pytest.mark.slow
@pytest.mark.parametrize('name', FLOW_TABLES)
def test_最下行が1回出現の種でない(name):
    """格子の最下行が，1 回出現の種 (流し込み) でないこと

    和名の欄を読んで確かめる．本物の行の和名の欄はカタカナだけ，流し込みは
    文章が欄を突き抜けるのでラテン語 (学名) が入る．
    """
    from comptea import row_kinds
    d0 = grid(name)
    im = image(name, d0)
    if 'block' not in d0.columns:
        d0 = d0.assign(block=0)
    # **段ごとに見る**．2 段組 (kinki_048・077) を段をまたいで読むと，左右の段の
    # 本物の行が「1 行に 2 種」に見えて流し込みと誤る．箱は工程と同じ
    # (和名の左端 〜 その段の組成部の右端)
    bad = []
    for blk, d in d0.groupby('block'):
        ja = d[d.obj_name == 'species_col']
        comp = d[d.obj_name == 'comp']
        if ja.empty or comp.empty:
            continue
        p = pitch_of(d)
        g = (comp.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
             .sort_values('y1'))
        last = g.iloc[-1]
        if row_kinds.looks_flow_ja(
                im, (float(ja.x1.min()), float(last.y1),
                     float(comp.x2.max()), float(last.y2)), pad=p * 0.3):
            bad.append(f'段{blk} y{last.y1:.0f}-{last.y2:.0f}')
    assert not bad, f'{name} の最下行が流し込み (1 回出現の種) になっている: {bad}'
