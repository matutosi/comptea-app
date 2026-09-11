"""できあがった格子を，独立した尺度で検査する

行の数・列の数がそろっていても壊れていることがあるので，**別の測り方**で
確かめる．列は印字の隙間と，行は組成部の谷と突き合わせ，行の高さの
ばらつきと，組成部に食い込んだ表頭の行も見る．

**判定するだけ**で直さない．引っかかったものは警告に載せ，画像に戻って
目で確かめる．
"""

import numpy as np
from PIL import Image

from . import ink
from .body_rows import ROW_COVER_MIN, body_extent_ink, expected_row_edges
from .col_edges import plot_gaps

Image.MAX_IMAGE_PIXELS = None


CHECK_MIN_COLS = 12     # これ以上の列があるときだけ，境と隙間を突き合わせる


CHECK_NEAR_PX = 10      # 境が隙間に乗っているとみなす距離


CHECK_MIN_RATIO = 0.7   # これを下回ると知らせる


def check_source_image(image, df_loc):
    """**格子が，突き合わせている画像に収まっているか**

    工程は傾きを直した画像や横倒しを起こした画像で検出するので，格子の座標は
    元画像とずれます．収まっていなければ，違う画像を見ている印です
    (2026-09-11: 20_p3 の表頭の境が元画像では「12 本字を割る」と出た)．
    """
    from . import source
    im = Image.open(image) if isinstance(image, str) else image
    return source.check(df_loc, im)


def check_grid_columns(image, df_loc, min_cols=CHECK_MIN_COLS,
                       near=CHECK_NEAR_PX, ratio=CHECK_MIN_RATIO):
    """格子の列の境が，印字の地点の隙間に乗っているかを測る

    地点の多い表では，列の境が1つずれても目では気づけない．
    隙間の位置は**検出とは別の測り方**(黒画素)で出せるので，
    突き合わせれば食い違いを機械的に見つけられる．

    列が `min_cols` 未満の表(手元の 88 枚はすべてこちら)では何も言わない．
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty or comp['col'].nunique() < min_cols:
        return []
    edges = np.array(sorted(set(comp['x1']) | set(comp['x2'])))[1:-1]
    if len(edges) == 0:
        return []
    box = (int(comp['x1'].min()), int(comp['y1'].min()),
           int(comp['x2'].max()), int(comp['y2'].max()))
    gaps = np.array(plot_gaps(ink.binarize(Image.open(image)), box), dtype=float)
    if len(gaps) == 0:
        return ['列の境を確かめようとしたが，地点のあいだの隙間が'
                '1本も見つからなかった．組成部の位置が違うかもしれない']
    dist = np.array([np.min(np.abs(gaps - e)) for e in edges])
    hit = float(np.mean(dist <= near))
    if hit >= ratio:
        return []
    return [f'**列の境の {(1 - hit) * 100:.0f}% が，印字の地点の隙間から'
            f'{near} px 以上離れている**(境 {len(edges)} 本，隙間 {len(gaps)} 本，'
            f'ずれの中央値 {np.median(dist):.0f} px)．'
            '列の区切りが地点と合っていないおそれが強い．段階1で目で確かめる']


def check_grid_rows(image, df_loc, min_ratio=ROW_COVER_MIN, df_det=None):
    """格子の行が，種名の列の字の行をどれだけ覆っているか

    **列は測っていたのに，行は測っていなかった**．そのため行が9割落ちても
    黙って通っていた(s01115_11 の Tab.74 は種名が 110 行あるのに，
    格子は 12 行だった．列は 14 で正しかったので気づけなかった．2026-09-02)．

    期待する行数は `rows_from_body()` と同じ**行の高さの格子**で数える
    (2026-09-04)．谷の数では「・」だけの行が数行まとめて 1 つの谷になり，
    汚れで 1 つの行間が 3 つに割れるので，正しい格子にも鳴る．
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty:
        return []
    dark = ink.binarize(Image.open(image))
    ext = body_extent_ink(dark, df_loc, df_det)
    edges, _ = expected_row_edges(dark, df_loc, df_det, ext)
    n = len(edges) - 1
    rows = int(comp['row'].nunique())
    if n <= 0 or rows >= n * min_ratio:
        return []
    return [f'**格子の行が {rows} 行しかないのに，組成部には字の行が {n} ある**'
            f'({rows / n:.2f} 倍)．行が丸ごと落ちているおそれが強い．'
            '段階1で表の下の方を目で確かめる．'
            '--conf を下げると取れることがある']


ROW_HEIGHT_CV_MAX = 1.0     # 行の高さの変動係数．これを超えたら知らせる


def check_row_heights(df_loc, cv_max=ROW_HEIGHT_CV_MAX):
    """行の高さが極端に不揃いでないかを見る

    表題の上に出た `row` の誤検出 1 本で格子が上へ伸び，表頭を丸ごと
    1 行に飲み込むことがあった(s01115_07_p1 の最大の行は 1,246 px で
    他の行の 37 倍．変動係数 2.47)．行の数・列の数・行の谷の比は
    どれも基準を満たしていたので，この物差しでしか見つからなかった
    (2026-09-03)．壊れていた 4 表は 1.6-5.4，正しい表は 0.05-0.55．
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty or comp['row'].nunique() < 3:
        return []
    g = comp.groupby('row').agg(a=('y1', 'min'), b=('y2', 'max'))
    h = (g['b'] - g['a']).to_numpy(dtype=float)
    if h.mean() <= 0:
        return []
    cv = float(h.std() / h.mean())
    if cv <= cv_max:
        return []
    return [f'**行の高さが極端に不揃い**(変動係数 {cv:.2f}，最大 {h.max():.0f} px'
            f' 対 中央値 {np.median(h):.0f} px)．表題や表頭を 1 行として'
            '飲み込んでいることが多い．段階1で表の上端を目で確かめる']


def check_header_rows(df_loc):
    """表頭の値の行が組成部の中まで下りていないかを見る

    表頭の項目行(`plot_row`)は組成部の全高に散って検出されることがあり，
    残すと組成部の行が**表頭の値として二重に切られる**(s01115_05_p2 は
    組成部の中に 149 行．2026-09-04)．表頭の帯が全高に広がるので，
    階層の列の判定(表頭が空)も壊れる．
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    head = df_loc[df_loc['obj_name'] == 'header_value']
    if comp.empty or head.empty:
        return []
    top = float(comp['y1'].min())
    g = head.groupby('row').agg(a=('y1', 'min'), b=('y2', 'max'))
    inside = int((((g['a'] + g['b']) / 2) > top).sum())
    if inside == 0:
        return []
    return [f'**表頭の値の行 {inside} 行が組成部の中にある**(全 {len(g)} 行)．'
            '項目行の誤検出が組成部の行を表頭として切っている．'
            '段階1で表頭と組成部の境を目で確かめる']
