import cv2

import numpy as np


import pandas as pd


from PIL import Image


from . import col_reach, ink
from .blocks import drop_stray_anchors, split_blocks
from .axes import _shift_edges_for_header, _snap_axis, axis_notes, coord_item, ink_profile, locate_edges, locate_x_range, locate_y_range
from .filters import drop_rows_outside_body, drop_unsupported_headers, filter_results, remove_dup_ranges


def _open_image(path):
    """
    画像を開く．開けないときはNone(呼び出し側で警告する)

    Streamlitのアップロードはファイル風のオブジェクトで渡ってくるため，
    読み込み位置を先頭に戻してから開く
    """
    if path is None:
        return None
    if isinstance(path, Image.Image):
        return path
    if hasattr(path, 'seek'):
        try:
            path.seek(0)
        except (OSError, ValueError):
            return None
    try:
        return Image.open(path)
    except (OSError, ValueError, TypeError):
        return None


def locate_items(df: pd.DataFrame, source_image: str = "", item: str = "all", threth: float = 0.8, threth_col: float = None, threth_row: float = None, image=None, snap: bool = True, max_shift_ratio: float = 0.3) -> pd.DataFrame:
    """
    指定itemの位置を特定

    Args:
        df: 検出結果のデータフレーム
        source_image: ソース画像のパス
        item: 対象アイテム（現在未使用）
        threth: 重複除去の閾値（0.0-1.0、デフォルト0.8）。threth_col/threth_rowが未指定時に使用
        threth_col: col方向の重複除去の閾値
        threth_row: row方向の重複除去の閾値
        image: 境界の調整に使う画像(PILのImageかパス)．
               省略すると df の source_image から読む
        snap: 文字に重なった境界を谷へずらすか
        max_shift_ratio: ずらせる幅．1行(1列)の間隔に対する割合

    Returns:
        位置を特定したDataFrame．
        位置を決められなかったものは**行を作らない**(0,0の幽霊データを作らない)．
        決められなかった理由は `df.attrs['warnings']` にリストで入る．
        呼び出し側は次のようにして利用者に伝えること．

            for w in df_located.attrs.get('warnings', []):
                st.warning(w)
    """
    if threth_col is None:
        threth_col = threth
    if threth_row is None:
        threth_row = threth
    warnings = []
    # 項目行にも項目名の列にも裏の取れない表頭は，種群を覆った誤検出
    df, n_heads = drop_unsupported_headers(df)
    if n_heads:
        warnings.append(
            f"項目行にも項目名の列にも重ならない 'header' の検出 {n_heads} 本を捨てた"
            '(組成部の最初の種群を表頭として覆っていることがある)．')
    # 表頭や1回出現種の中にある row は，見た目が似ているだけの誤検出
    df, n_dropped = drop_rows_outside_body(df)
    if n_dropped:
        warnings.append(
            f"表頭や「1回出現種」の中にあった 'row' の検出 {n_dropped} 本を捨てた"
            '(組成部の行と見た目が似ているため誤検出されやすい)．')
    # 境界の調整に使う画像
    img = None
    if snap:
        img = _open_image(image if image is not None
                          else (source_image or df['source_image'].iloc[0]))
        if img is None:
            warnings.append(
                '画像を読めなかったため，文字に重なった境界の調整を行わなかった．'
                'locate_items(image=...) で画像を渡すと調整できる．')
    # 段の目印のうち，本物の段と縦に重ならないものは捨てる (04_p2 の表の下の偽の sname は
    # 段を 1 つ増やし，右端の列を足す限界になっていた．2026-09-10)
    df, n_stray = drop_stray_anchors(df)
    if n_stray:
        warnings.append(
            f"段の目印 ('sname' か 'species_col') のうち {n_stray} 本は，いちばん高い段と"
            '縦に重ならないので捨てた(表の下の注記などの誤検出)．')
    # 1ページに段が複数あるときは，段ごとに格子を作る(混ぜると壊れる)
    blocks = split_blocks(df)
    if len(blocks) > 1:
        warnings.append(
            f'このページには段が {len(blocks)} つある(種名のリストを折り返した組み方)．'
            '段ごとに格子を作り，行は左の段から続けて番号を振る．'
            '地点は段をまたいで同じものとして扱う．')
    # 段ごとの右の限界: 次の段の種名の左端 (右端の外の列を足すとき，隣の段へ入らないため)
    lefts = []
    for block in blocks:
        anc = block[block['obj_name'].isin(('sname', 'species_col'))]
        lefts.append(float(anc['x1'].min()) if not anc.empty else None)
    items = []
    for i, block in enumerate(blocks):
        x_max = None
        if i + 1 < len(blocks) and lefts[i + 1] is not None:
            x_max = lefts[i + 1] - REACH_MARGIN
        elif img is not None:
            x_max = float(img.width)
        res, block_warnings = _locate_block(
            block, source_image, img, max_shift_ratio, threth_col, threth_row,
            x_max=x_max)
        prefix = f'段{i + 1}: ' if len(blocks) > 1 else ''
        warnings.extend(prefix + w for w in block_warnings)
        if res is not None:
            res['block'] = i + 1
            items.append(res)
    # 文章として読む領域は，そのまま1つのセルとして渡す．
    # 格子に切れないので位置決めはしないが，OCRページで読ませる必要がある
    # (文章形式の表頭と「1回出現種」はここから plot_table / comp_table が使う)
    regions = ['once_species']
    if (df['obj_name'] == 'plot_row').sum() == 0:
        # 表頭を文章として渡すのは，項目行が無いとき(1地点の表)だけ．
        # 表形式の表頭まで渡すと，格子で読めるものを文章として解釈してしまい，
        # 黙って無意味な値が出る
        regions.append('header')
    for name in regions:
        region = df[df['obj_name'] == name]
        if not region.empty:
            r = region.iloc[0]
            items.append(pd.DataFrame([{
                'x1': float(r['x1']), 'x2': float(r['x2']),
                'y1': float(r['y1']), 'y2': float(r['y2']),
                'obj_name': name, 'note': '', 'block': 1,
            }]))
    if not items:
        return _empty_located(df, warnings)
    res = pd.concat(items, ignore_index=True)
    res['source_image'] = df['source_image'].iloc[0]
    res['model'] = df['model'].iloc[0]
    res.attrs['warnings'] = warnings
    return res


# 階層の列を隙間から補うときの条件(2026-09-01 にラベル付き33枚で測って決めた)
LAYER_GAP_MIN_W = 20      # 隙間の幅(画素)．これ未満は字が入らない


LAYER_GAP_MIN_INK = 0.45  # 隙間の黒画素 / 種名の列の黒画素(どちらも罫線を除く)


LAYER_BLOB_GAP = 0.3      # 隙間の中のかたまりをつなぐ幅 (行の高さの倍数)．
                          # 0.5 だと，階層の記号と隣の未検出の地点の列がつながる
                          # (20_p3 は K と最初の値が 1 つの 198 px のかたまりになった)


LAYER_BLOB_MIN_W = 0.15   # かたまりの幅の下限 (行の高さの倍数．縦罫線の spike を落とす)


LAYER_BLOB_MAX_W = 3.5    # 同上限 (これを超えるものは階層の列ではない．「B1,B2」の
                          # ような 2 つ組の記号は 3.2 行ぶんある: 06_p2 は 111 px / 35 px)


LAYER_BLOB_ROWS = 0.4     # 行のこの割合より少ない行にしか字が無いかたまりは，
                          # 種名のはみ出しとみなす．**距離では分けられない** (記号が
                          # 和名の枠に接している表がある: 079-1・06_p2・17_p1・20_p3 は
                          # 隙間の先頭からかたまりが始まる)．はみ出すのは長い和名だけ
                          # なので行の割合が低く，記号はほとんどの行にある


def _gap_text_blob(dark, gx1, gx2, gy1, gy2, pitch):
    """種名と組成部の隙間の中から，階層の記号のかたまりを 1 つ選んで (x1, x2) を返す

    選ぶのは**字の比がいちばん高い**かたまり．幅が広すぎるもの (検出されなかった
    地点の列を含む)，狭すぎるもの (罫線の spike)，種名に接するもの (種名のはみ出し) は
    候補から外す．縦罫線は消してから見る (罫線だけの隙間を階層の列にしていた事故がある)．
    """
    from . import row_track
    gx1, gx2 = int(gx1), int(gx2)
    gy1, gy2 = int(gy1), int(gy2)
    if gx2 - gx1 < 3 or gy2 - gy1 < 3:
        return None
    clean = row_track.clean_rules(dark, gx1, gx2, gy1, gy2, pitch)
    if clean.size == 0:
        return None
    cols = clean.sum(axis=0) > 1
    if not cols.any():
        return None
    gap = max(1, int(LAYER_BLOB_GAP * pitch))
    on = np.flatnonzero(cols)
    breaks = np.flatnonzero(np.diff(on) > gap)
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]] + 1
    lo_w = max(3, int(LAYER_BLOB_MIN_W * pitch))
    hi_w = LAYER_BLOB_MAX_W * pitch
    n_row = max(1, int(round((gy2 - gy1) / pitch)))
    best, best_r = None, -1.0
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b - a < lo_w or b - a > hi_w:
            continue
        # 行の何割に字があるか (種名のはみ出しは長い和名の行だけなので低い)
        rows = clean[:, a:b].any(axis=1)
        hit = 0
        y = 0
        while y < len(rows):
            if rows[y]:
                hit += 1
                y += max(1, int(pitch * 0.6))
            else:
                y += 1
        if hit < LAYER_BLOB_ROWS * n_row:
            continue
        r = ink.text_ratio(dark, gy1, gy2, gx1 + a, gx1 + b)
        if r > best_r:
            best, best_r = (gx1 + a, gx1 + b), r
    return best


LAYER_HEAD_MAX = 0.5      # 表頭の字が地点の列の中央値のこの倍以上なら，階層ではなく
                          # **検出されなかった地点の列**．値と記号は「行の何割に字が
                          # あるか」では分けられない (どちらもほとんどの行にある)．
                          # 表頭が空かどうかが効く (kinki_040 は最初の値の列を階層と
                          # 誤って拾った．2026-09-09)


def _guess_layer_column(img, name_ranges, x_edges, y_edges, head_y=None):
    """検出されなかった階層の列を，種名の列と組成部の隙間から補う

    `layer` はラベルが 22 件しかなく検出が育っていない．example.jpg では
    信頼度 0.05 でしか出ず，閾値を下げて拾うと他が壊れる．
    そこで**隙間に字があるか**で決める(2026-09-01)．

    ラベル付き33枚(段ごとに39件)で測った結果(**罫線を除いた黒画素**で見る)．
      階層の列がある 21 件: 隙間の黒画素/種名の黒画素は **0.562 - 3.109**
      階層の列が無い 18 件: **0.000 - 0.338**
    重なりが無いので，あいだの 0.45 で切る．
    幅だけでは分けられない(以前に測って分からなかったのはこのため)．
    **罫線を落とさないと分けられない**(kinki_010 は階層の列が無いのに，
    傾いた縦罫線の黒画素で「字がある」と見えていた)．

    間違って作っても，読んだ中身が階層として通らなければ段階2に挙がるので，
    黙って値が化けることはない．

    Returns:
        (x1, x2 を1行持つ DataFrame または None, 補ったかどうか)
    """
    if img is None or x_edges is None or y_edges is None or not name_ranges:
        return None, False
    name_x2 = max(r['x2'].iloc[0] for r in name_ranges)
    name_x1 = min(r['x1'].iloc[0] for r in name_ranges)
    gx1, gx2 = float(name_x2), float(x_edges[0])
    gy1, gy2 = float(y_edges[0]), float(y_edges[-1])
    if gx2 - gx1 < LAYER_GAP_MIN_W or gy2 - gy1 < 10:
        return None, False
    # **縦の罫線は字と数えない**(`ink.thick_ratio`)．表の境の線が隙間を通ると，
    # 字が無くても「字がある」と見えてしまう
    # (kinki_010 は階層の列が無いのに，罫線に乗った列を作っていた．2026-09-01)
    dark = ink.binarize(img)
    # **縦の罫線も字と数えない**(`ink.text_ratio`．2026-09-03)．
    # `thick_ratio` は横の連なりしか見ないので，幅 4-5 px の縦罫線が
    # 字として残る．表の左端の罫線だけの隙間が階層の列になり，
    # 読み取りで 29 セルが全滅した(s01115_14_p5 に階層の列は無い)
    base = ink.text_ratio(dark, gy1, gy2, name_x1, name_x2)
    if not base:
        return None, False
    # **隙間を丸ごと測ってはいけない** (2026-09-09)．隙間が広く記号が一部にしかない表で
    # 比が薄まって「階層なし」になり (07_p3 は 222 px の隙間に記号が 75 px だけで
    # 比 0.25．閾値 0.45 に届かない)，逆に隙間に検出されなかった地点の列が入ると
    # 階層の列が組成部まで広がる (03_p1 は枠が 561 px になり 202 行 x 3 列を巻き込んだ)．
    # 隙間の中の**字のかたまり**を見つけ，そのかたまりだけで測る
    pitch = (float(np.median(np.diff(np.asarray(y_edges, dtype=float))))
             if len(y_edges) >= 3 else (gy2 - gy1) / 10.0)
    blob = _gap_text_blob(dark, gx1, gx2, gy1, gy2, pitch)
    if blob is None:
        return None, False
    bx1, bx2 = blob
    if ink.text_ratio(dark, gy1, gy2, bx1, bx2) / base < LAYER_GAP_MIN_INK:
        return None, False
    if _looks_like_value_column(dark, bx1, bx2, x_edges, head_y):
        return None, False
    return pd.DataFrame({'x1': [float(bx1)], 'x2': [float(bx2)]}), True


def _looks_like_value_column(dark, bx1, bx2, x_edges, head_y):
    """かたまりの上の表頭に字があれば，階層ではなく**検出されなかった地点の列**"""
    if head_y is None or x_edges is None or len(x_edges) < 4:
        return False
    hy1, hy2 = float(head_y[0]), float(head_y[1])
    if hy2 - hy1 < 10:
        return False
    vals = [ink.text_ratio(dark, hy1, hy2, float(a), float(b))
            for a, b in zip(x_edges[:-1], x_edges[1:])]
    vals = [v for v in vals if v]
    if len(vals) < 3:
        return False
    med = float(np.median(vals))
    if not med:
        return False
    return ink.text_ratio(dark, hy1, hy2, bx1, bx2) / med >= LAYER_HEAD_MAX


# 段の端に行を足すときの条件(2026-09-01)
EXTEND_CLASSES = ('col', 'species_col', 'sname', 'layer')


EXTEND_MIN_GAP = 0.6      # 列の箱の端との差が行間隔のこの割合を超えたら足す


EXTEND_MAX = 2            # 片側で足す上限(行)


EXTEND_MIN_INK = 0.25     # 足す帯の黒画素 / 行の中央値．これ未満なら足さない


TRIM_MAX_INK = 0.05       # 端の帯の黒画素 / 行の中央値．これ未満なら落とす(空の帯)


def _extend_rows_to_block(y_edges, y_interp, df, source_image, img, x_lo, x_hi):
    """段の上下に，検出漏れで届いていない行を足す

    格子の縦の範囲は `row` の**検出だけ**で決まり，`locate_edges()` は
    検出と検出の**あいだ**しか内挿しない．そのため**最後の行が検出されないと
    外側へ伸ばせず，1行まるごと黙って落ちる**
    (kinki_047 の段1: 印字 29 行に対し格子 28 行．2026-09-01)．

    列の箱(`col`・種名・学名・階層)は段の端まで伸びているので，そこまでの
    空きが行間隔の半分を超えていれば，1行ぶんの帯を足す．
    **空の帯を足さないよう，その帯に字があるときだけ足す**．

    併せて，**字の無い端の帯は落とす**(`row` の誤検出で空の帯が残ることがある)．

    Returns:
        (y_edges, y_interp, 足した行数, 落とした行数)
    """
    if y_edges is None or len(y_edges) < 3 or img is None:
        return y_edges, y_interp, 0, 0
    tops, bottoms = [], []
    for name in EXTEND_CLASSES:
        r = locate_y_range(df, source_image, obj_name=name)
        if r is not None:
            tops.append(float(r['y1'].iloc[0]))
            bottoms.append(float(r['y2'].iloc[0]))
    if not tops:
        return y_edges, y_interp, 0, 0
    # 1つの箱が行き過ぎていることがあるので，中央値で段の端を決める
    y_lo, y_hi = float(np.median(tops)), float(np.median(bottoms))
    pitch = float(np.median(np.diff(y_edges)))
    if pitch <= 0:
        return y_edges, y_interp, 0, 0
    # **表頭の中へは足さない**(2026-09-04)．`col` の箱は表頭の上端から
    # 始まるので，格子が表頭の直下から始まる表では表頭の最後の 2 行が
    # 組成部の行として足されていた(s01115_07_p4 の `草本層植被率`・`出現種数`)
    heads = df[df['obj_name'].isin(('header', 'header_col'))]
    above = heads[heads['y2'] <= float(y_edges[0]) + pitch]
    if not above.empty:
        y_lo = max(y_lo, float(above['y2'].max()))

    dark = ink.binarize(img)
    base = np.median([ink.ratio(dark, a, b, x_lo, x_hi)
                      for a, b in zip(y_edges[:-1], y_edges[1:])])
    limit = max(0.002, base * EXTEND_MIN_INK)

    edges = [float(v) for v in y_edges]
    interp = [bool(v) for v in y_interp]
    added = 0

    # **字の無い端の帯は落とす**．`row` の誤検出で，段の下に空の帯が
    # 残ることがある(kinki_044・045．2026-09-01)．
    # 中身のある行を落とさないよう，**ほぼ真っ白なときだけ**にする．
    for _ in range(EXTEND_MAX):        # 下端
        if len(edges) <= 2:
            break
        if ink.ratio(dark, edges[-2], edges[-1], x_lo, x_hi) > base * TRIM_MAX_INK:
            break
        edges.pop()
        interp.pop()
        added -= 1
    for _ in range(EXTEND_MAX):        # 上端
        if len(edges) <= 2:
            break
        if ink.ratio(dark, edges[0], edges[1], x_lo, x_hi) > base * TRIM_MAX_INK:
            break
        edges.pop(0)
        interp.pop(0)
        added -= 1
    n_trim = -added if added < 0 else 0
    added = 0

    for _ in range(EXTEND_MAX):        # 下端
        if y_hi - edges[-1] <= pitch * EXTEND_MIN_GAP:
            break
        new = min(edges[-1] + pitch, y_hi)
        if ink.ratio(dark, edges[-1], new, x_lo, x_hi) < limit:
            break
        edges.append(new)
        interp.append(True)
        added += 1
    for _ in range(EXTEND_MAX):        # 上端
        if edges[0] - y_lo <= pitch * EXTEND_MIN_GAP:
            break
        new = max(edges[0] - pitch, y_lo)
        if ink.ratio(dark, new, edges[0], x_lo, x_hi) < limit:
            break
        edges.insert(0, new)
        interp.insert(0, True)
        added += 1
    if not added and not n_trim:
        return y_edges, y_interp, 0, 0
    return (np.array(edges, dtype=float), np.array(interp, dtype=bool),
            added, n_trim)


# 種名の列が格子より何行ぶん先まで伸びていたら知らせるか(2026-09-01)
REACH_MAX_ROWS = 1.0


def check_body_reach(df, source_image, y_edges):
    """格子が本文の下(上)まで届いているかを見る

    **種名の列(`species_col` / `sname`)の箱は本文の範囲をよく表す**ので，
    そこまで格子が届いていなければ，行がまるごと落ちている見込みが高い．
    `_extend_rows_to_block()` は片側2行までしか足さないので，
    それを超えて落ちている段はここで知らせる．

    手元の 74 段で測ると，残りは中央値 0.09 行・90% で 0.34 行．
    1行ぶんを超えたのは3段だけで，どれも本当に落ちていた
    (kinki_037 は 21 行，kinki_070 は 5 行，kinki_042 は 2 行．2026-09-01)．
    """
    if y_edges is None or len(y_edges) < 2:
        return []
    tops, bottoms = [], []
    for name in ('species_col', 'sname'):
        r = locate_y_range(df, source_image, obj_name=name)
        if r is not None:
            tops.append(float(r['y1'].iloc[0]))
            bottoms.append(float(r['y2'].iloc[0]))
    if not tops:
        return []
    pitch = float(np.median(np.diff(y_edges)))
    if pitch <= 0:
        return []
    below = (max(bottoms) - float(y_edges[-1])) / pitch
    above = (float(y_edges[0]) - min(tops)) / pitch
    out = []
    for gap, where in ((below, '下'), (above, '上')):
        if gap > REACH_MAX_ROWS:
            out.append(
                f'**種名の列は格子より {gap:.1f} 行ぶん{where}まで伸びている**．'
                'その範囲の行が丸ごと落ちているおそれが強い．'
                '段階1で表の端を目で確かめる．')
    return out


REACH_MARGIN = 10         # 右端の外の列を足すとき，次の段の種名の左端からこれだけ手前で止める


HEAD_COL_MIN_W = 3.0      # 項目名の領域の幅の下限 (行の高さの倍数)


HEAD_COL_MIN_BANDS = 3    # 字のある帯の数の下限


HEAD_COL_INK = 0.002      # 帯に字があるとみなす黒画素の割合 (細い縦罫線は除く)


def _guess_header_col(dark, left, value_x, bands, pitch):
    """`header_col` が無いとき，表頭の枠の左端〜値の左端を項目名の列とみなしてよいか

    項目名 (独文・和文) が紙面にあるのに枠が検出されない表がある (s01115_14_p5・15_p1・
    15_p2・15_p4・21_p3．2026-09-10 ユーザ指摘)．領域が 3 行幅以上あり，項目行の帯の
    3 つ以上に字があれば (left, value_x) を返す．細い縦罫線は字に数えない．
    """
    if value_x - left < HEAD_COL_MIN_W * pitch:
        return None
    n = sum(1 for a, b in bands
            if ink.thick_ratio(dark, a, b, left + 5, value_x - 5) > HEAD_COL_INK)
    if n < HEAD_COL_MIN_BANDS:
        return None
    return float(left), float(value_x)


def _locate_block(df: pd.DataFrame, source_image: str, img, max_shift_ratio: float,
                  threth_col: float = 0.8, threth_row: float = 0.8, x_max=None):
    """
    1つの段について格子を作る(locate_itemsから段ごとに呼ぶ)

    重複の除去は**段に分けたあと**に行う．
    行の重複判定はyの重なりだけを見てxを見ないため，
    段をまたいで一括でかけると，右の段の行が左の段の行の重複として消える．

    Args:
        x_max: 右端の外の列を足すときの限界 (次の段の左端・画像の右端)

    Returns:
        (DataFrame または None, warningsのリスト)
    """
    df = remove_dup_ranges(df, threth_col=threth_col, threth_row=threth_row)
    class_spec  = "species_col"
    class_layer = "layer"
    class_cols  = "col"
    class_rows  = "row"
    class_sname = "sname"
    class_comp  = "comp"
    warnings = []
    # itemごとにxかyの範囲を抽出(検出が無いときはNone)
    spec_x_range  = locate_x_range(df, source_image, obj_name=class_spec)
    sname_x_range = locate_x_range(df, source_image, obj_name=class_sname)
    layer_x_range = locate_x_range(df, source_image, obj_name=class_layer)
    comp_y_range  = locate_y_range(df, source_image, obj_name=class_rows)
    # rowが無いと行の位置が決まらず，何も作れない
    if comp_y_range is None:
        warnings.append(
            f"'{class_rows}' が1件も検出されなかったため，行の位置を決められない．"
            "検出の閾値(conf)を下げるか，画像の前処理を見直す．")
        return None, warnings
    # 検出そのものから行・列の境界を組み立てる
    y_edges, y_interp, y_warns = locate_edges(df, source_image, obj_name=class_rows, axis='y')
    x_edges, x_interp, x_warns = locate_edges(df, source_image, obj_name=class_cols, axis='x')
    warnings.extend(y_warns)
    warnings.extend(x_warns)
    if y_edges is None:
        warnings.append(
            f"'{class_rows}' が1件も検出されず，行の位置を決められない．"
            '何も出力しない．')
        return None, warnings
    n_rows = len(y_edges) - 1
    y_snapped = y_unresolved = None
    x_snapped = x_unresolved = None
    # 文字に重なった境界を谷へずらす
    if img is not None:
        # 行の境界は表全体を横切るので，表側も含めた幅で見る
        attr_ranges = [r for r in [spec_x_range, sname_x_range, layer_x_range]
                       if r is not None]
        x_lows = [r['x1'].iloc[0] for r in attr_ranges]
        x_highs = [r['x2'].iloc[0] for r in attr_ranges]
        if x_edges is not None:
            x_lows.append(x_edges[0])
            x_highs.append(x_edges[-1])
        if not x_lows:
            # 種名・学名・階層の列も col も1つも取れなかった段．
            # 横の範囲が決められないので，境界を谷へずらす処理は飛ばす
            # (ここで min() が空になって落ちていた)
            warnings.append(
                '横の範囲を決められる検出が1つも無いため，'
                '境界を文字の谷へずらす調整を行わなかった．'
                '種名の列・階層の列・col のいずれかが検出されているか確かめる．')
            img = None
    if img is not None:
        x_lo, x_hi = min(x_lows), max(x_highs)
        # 段の上下に，検出漏れで届いていない行を足す
        y_edges, y_interp, n_ext, n_trim = _extend_rows_to_block(
            y_edges, y_interp, df, source_image, img, x_lo, x_hi)
        if n_ext or n_trim:
            n_rows = len(y_edges) - 1
        if n_ext:
            warnings.append(
                f'段の端にある {n_ext} 行は行として検出されなかったが，'
                '列の箱がそこまで伸びていて字もあるので足した．'
                "内挿した行として note に 'interpolated' を付けてある．")
        if n_trim:
            warnings.append(
                f'段の端にある {n_trim} 行は**字が1つも無かった**ので落とした'
                "('row' の誤検出とみなした)．")
        warnings.extend(check_body_reach(df, source_image, y_edges))
        y_edges, y_warns, y_snapped, y_unresolved = _snap_axis(
            y_edges, img, (x_lo, y_edges[0], x_hi, y_edges[-1]),
            axis='y', label='行', max_shift_ratio=max_shift_ratio)
        warnings.extend(y_warns)
        if x_edges is not None:
            x_edges, x_warns, x_snapped, x_unresolved = _snap_axis(
                x_edges, img, (x_edges[0], y_edges[0], x_edges[-1], y_edges[-1]),
                axis='x', label='列', max_shift_ratio=max_shift_ratio)
            warnings.extend(x_warns)
            # **右端の外の列を足す** (2026-09-10 ユーザ指摘: 006・08_p2・04_p2・15_p3)．
            # `locate_edges` は検出のあいだしか内挿しないので，端の列は丸ごと落ちる．
            # 本体にも表頭にも字のある帯だけ (常在度の字のはみ出しは表頭が空)
            head_rows = filter_results(df, source_image, 'plot_row')
            y_head = ((float(head_rows['y1'].min()), float(head_rows['y2'].max()))
                      if not head_rows.empty else None)
            x_edges, n_reach = col_reach.reach_right(
                ink.binarize(img), x_edges, y_edges, y_head=y_head, x_max=x_max)
            if n_reach:
                x_interp = np.r_[x_interp, np.ones(n_reach, dtype=bool)]
                x_snapped = np.r_[x_snapped, np.zeros(n_reach, dtype=bool)]
                x_unresolved = np.r_[x_unresolved, np.zeros(n_reach, dtype=bool)]
                warnings.append(
                    f'組成部の右端の外に，本体にも表頭にも字のある列が {n_reach} 列あったので'
                    "足した(検出の箱が届いていない)．内挿した列として note に 'interpolated' を"
                    '付けてある．段階1で右端の列を目で確かめる．')
    if y_interp.any():
        warnings.append(
            f'行のうち {int(y_interp.sum())} 行は検出されず，前後の間隔から内挿した'
            f'(全 {n_rows} 行)．内挿した行は位置がずれていることがある．')
    # セルごとの気になる点(内挿・スナップ・文字に重なったまま)
    y_notes = axis_notes(y_interp, y_snapped, y_unresolved, n=n_rows)
    # 階層の列が検出されなかったときは，種名の列と組成部の隙間から補う
    layer_guessed = False
    if layer_x_range is None:
        name_ranges = [r for r in (spec_x_range, sname_x_range) if r is not None]
        head_box = df[df['obj_name'] == 'header'] if 'obj_name' in df else None
        head_y = ((float(head_box['y1'].min()), float(head_box['y2'].max()))
                  if head_box is not None and len(head_box) else None)
        layer_x_range, layer_guessed = _guess_layer_column(
            img, name_ranges, x_edges, y_edges, head_y=head_y)
        if layer_guessed:
            warnings.append(
                f"'{class_layer}' は検出されなかったが，種名の列と組成部の隙間に"
                '字があるため，そこを階層の列として補った．'
                '読んだ中身が階層として通らなければ段階2に挙がる．')
    # itemごとのx,y座標．範囲が決まらないものは作らない
    items = []
    for x_range, obj_name in [
        (spec_x_range , class_spec),
        (sname_x_range, class_sname),
        (layer_x_range, class_layer),
    ]:
        if x_range is None:
            if obj_name == class_layer:
                # 階層列を持たない組成表は珍しくない(草本群落など)．
                # ラベル付き33枚のうち15枚にしかlayerが無かった．
                # **隙間の黒画素で有無を見分けられる**ようになった(2026-09-01)ので，
                # ここまで来たら「無い」と言い切ってよい(_guess_layer_column)．
                warnings.append(
                    '階層の列は**無い**とみなした'
                    '(検出が無く，種名の列と組成部の隙間にも字が無い)．'
                    '草本群落などでは普通のこと．'
                    '階層の列があるはずなら，段階1で隙間を目で確かめる．')
            else:
                warnings.append(
                    f"'{obj_name}' が1件も検出されなかったため，この列は出力しない．")
            continue
        col_edges = np.array([x_range['x1'].iloc[0], x_range['x2'].iloc[0]], dtype=float)
        items.append(coord_item(col_edges, y_edges, obj_name=obj_name, y_notes=y_notes))
    # 組成部のセル
    if x_edges is None:
        warnings.append(
            f"'{class_cols}' が1件も検出されず，組成部のセル('{class_comp}')を"
            '出力できない．検出の閾値(conf)を下げると改善することが多い．')
    else:
        n_cols = len(x_edges) - 1
        if x_interp.any():
            warnings.append(
                f'列のうち {int(x_interp.sum())} 列は検出されず，前後の間隔から内挿した'
                f'(全 {n_cols} 列)．')
        x_notes = axis_notes(x_interp, x_snapped, x_unresolved, n=n_cols)
        items.append(coord_item(x_edges, y_edges, obj_name=class_comp,
                                x_notes=x_notes, y_notes=y_notes))
    # 表頭のセル
    items += _locate_header(df, source_image, x_edges, warnings,
                            img=img, max_shift_ratio=max_shift_ratio)
    if not items:
        return None, warnings
    return pd.concat(items, ignore_index=True), warnings


def _header_bands_from_names(img, box, min_gap_ratio: float = 0.15):
    """項目名の列の字の行から，表頭の帯の境界を作る

    項目名は**1項目1行**．値は2行にわたることがある(調査番号の 'MS' と '162'，
    調査月日の '6' と '11')ので，帯は
    **項目名の行の始まりから，次の項目名の行の始まりまで**とする．
    こうすると，2行にわたる値もその項目の帯に収まる．

    `plot_row` の箱をそのまま帯にすると，2項目が1つの帯に入ったり，
    値の2行目だけが1つの帯になったりしていた (2026-09-01)．

    Returns:
        境界の並び(np.ndarray)．作れないときは None
    """
    profile, origin, _ = ink_profile(img, box, axis='y')
    if profile is None or profile.max() <= 0:
        return None
    on = profile >= max(1.0, profile.max() * 0.06)
    # 字のある区間の始まりを集める
    starts, ends = [], []
    for i, v in enumerate(on):
        if v and (i == 0 or not on[i - 1]):
            starts.append(i)
        if v and (i == len(on) - 1 or not on[i + 1]):
            ends.append(i)
    if len(starts) < 2:
        return None
    # 1行の中の切れ目でちぎれた区間はつなぐ．
    # つなぐ幅は**行間より狭く**する(行間は行の間隔のおよそ4分の1しかない)．
    # 広くとると全部が1つにつながる (2026-09-01 に 0.45 で失敗した)
    pitch = float(np.median(np.diff(starts)))
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] < pitch * min_gap_ratio:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if len(merged) < 2:
        return None
    # 罫線は字の行と別の区間になるので落とす(項目ごとに下線を引いた資料がある)．
    # 字の行より薄い区間を罫線とみなす (2026-09-01．kinki_045 で 1 行が 2 本になった)
    heights = [e - s + 1 for s, e in merged]
    min_h = max(3.0, float(np.median(heights)) * 0.35)
    merged = [b for b, h in zip(merged, heights) if h >= min_h]
    if len(merged) < 2:
        return None
    edges = [origin + merged[0][0] - 3]
    for s, _ in merged[1:]:
        edges.append(origin + s - 3)
    edges.append(float(box[3]))
    return np.array(edges, dtype=float)


def _locate_header(df: pd.DataFrame, source_image: str, x_edges, warnings: list,
                   img=None, max_shift_ratio: float = 0.3):
    """
    表形式の表頭のセルを作る

    項目行は `plot_row`，地点の列は組成部と同じ `col` の境界を使う
    (`col` の箱は表頭まで伸びているので，列の位置は共通)．
    項目名は `header_col` の中にある．

    **境界は本文の行と同じように字の谷へ寄せる**(2026-09-01)．
    `plot_row` の箱をそのまま帯にすると，数字が上下に割れて切り出され，
    EasyOCR でも AI でも読めない画像になっていた．

    文章形式の表頭(1地点の表)には項目行が無いので，何も作らない．

    Returns:
        coord_item() の戻り値のリスト(作れないときは空)
    """
    class_plot_row = 'plot_row'
    class_hdr_col = 'header_col'
    class_hdr = 'header'
    rows = filter_results(df, source_image, class_plot_row).sort_values(by='y1')
    if rows.empty:
        return []
    if x_edges is None:
        warnings.append(
            f"'{class_plot_row}' はあるが列の位置が決まらないため，"
            '表頭のセルを出力できない．')
        return []
    # 項目行は高さがまちまち(値が複数行にわたる項目がある)ので，
    # 等間隔を前提にした locate_edges は使わず，箱をそのまま帯として使う
    bands = [[float(r.y1), float(r.y2)] for r in rows.itertuples()]
    heights = [b - a for a, b in bands]
    median_h = float(np.median(heights))
    # 表頭の上下に，項目行として拾えていない帯が残っていたら足す
    # (kinki_004 は「通し番号」の行にラベルが無く，先頭の項目が落ちていた)
    head = filter_results(df, source_image, class_hdr)
    n_added = 0
    if not head.empty:
        top, bottom = float(head['y1'].min()), float(head['y2'].max())
        if bands[0][0] - top > median_h * 0.5:
            bands.insert(0, [top, bands[0][0]])
            n_added += 1
        if bottom - bands[-1][1] > median_h * 0.5:
            bands.append([bands[-1][1], bottom])
            n_added += 1
    if n_added:
        warnings.append(
            f"表頭の端にある {n_added} 行は '{class_plot_row}' として検出されず，"
            "'header' の範囲から補った．")
    # 隣り合う帯のあいだを中点で分ける(隙間に値が落ちないように)
    h_edges = [bands[0][0]]
    for i in range(len(bands) - 1):
        h_edges.append((bands[i][1] + bands[i + 1][0]) / 2)
    h_edges.append(bands[-1][1])
    h_edges = np.array(h_edges, dtype=float)
    # **境が下がらないようにする**(2026-09-03)．`plot_row` の箱が重なって
    # 並ぶと，隣り合う帯の中点が前の境より上に来て**高さが負のセル**が
    # できる(s01115_18_p2 の表頭の 1 行)．そのまま渡すと切り出しで落ちる
    h_edges = np.maximum.accumulate(h_edges)
    n_items = len(h_edges) - 1
    name_dy = 0.0        # 項目名の側の帯を値の帯からずらす量 (px)
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
    n_wide = sum(1 for g in gaps if g > median_h * 0.8)
    if n_wide:
        warnings.append(
            f'表頭の項目行のあいだに，1行ぶん以上あいた箇所が {n_wide} つある．'
            '拾えていない項目があるかもしれない．')
    hc = locate_x_range(df, source_image, obj_name=class_hdr_col)
    hc_guessed = False
    if hc is None and img is not None and not head.empty:
        # **枠が無くても項目名の字があれば補う** (2026-09-10 ユーザ指摘: 14_p5・15_p1・
        # 15_p2・15_p4・21_p3 は項目名が紙面にあるのに `header_col` が検出されず，
        # 値だけになっていた)．領域は表頭の枠の左端から値の左端まで．
        # 独文と和文の境は対策 B が黒画素で決める
        guess = _guess_header_col(ink.binarize(img), float(head['x1'].min()),
                                  float(x_edges[0]), bands, median_h)
        if guess is not None:
            hc = pd.DataFrame({'x1': [guess[0]], 'x2': [guess[1]]})
            hc_guessed = True
            warnings.append(
                f"'{class_hdr_col}' は検出されなかったが，表頭の枠の左端 {guess[0]:.0f} px から"
                f'値の左端 {guess[1]:.0f} px までに項目名の字があるので，そこを項目名の列として'
                '補った．段階1で項目名の列を目で確かめる．')
    # **項目名の行から帯を作り直す**(できるとき)．
    # `plot_row` の箱は項目行と1対1に対応しておらず，2項目を1つの箱に
    # まとめたり，値の2行目だけを1つの箱にしたりする．
    # 項目名は1項目1行なので，そちらの字の行を数える方が確かめやすい．
    if img is not None and hc is not None:
        name_x1 = float(hc['x1'].iloc[0])
        name_x2 = float(x_edges[0])      # 項目名(独語・和名)から値の手前まで
        # **まず OCR の検出器の箱から行を作る**(2026-09-09)．投影の区間分けは
        # タイプ打ちで崩れる(行間に谷が無く 16 行が 3 行に，かけらが行になる，
        # 字数の少ない先頭行が落ちる)．検出器は文字を語や行にまとめた箱を返すので，
        # 箱の y を束ねればそのまま行になる(代表 6 表で真値と一致)．
        # 領域は表頭の枠の 1 行上から(先頭行の取りこぼし対策)．表題・凡例の帯は
        # 値側に字が無いことで落とす．3 行未満なら投影へ戻す(複数方式の候補から選ぶ)
        from . import header_lines
        ocr_top = float(h_edges[0])
        if not head.empty:
            ocr_top = min(ocr_top, float(head['y1'].min()))
        ocr_top = max(0.0, ocr_top - median_h)
        # 行を束ねる閾値は**組成部の行の高さ**(検出の `row` の箱の高さの中央値)の半分．
        # 表頭の項目行は組成部と同じ行送りで組まれている
        body_rows = filter_results(df, source_image, 'row')
        pitch = (float((body_rows['y2'] - body_rows['y1']).median())
                 if len(body_rows) >= 3 else float(median_h))
        hinfo = {}
        ocr_edges = header_lines.header_bands(
            img, (name_x1, ocr_top, name_x2, float(h_edges[-1])),
            value_x=(float(x_edges[0]), float(x_edges[-1])), pitch=pitch, info=hinfo)
        if ocr_edges is not None:
            # **項目名の側の帯は，値との印字のずれだけ上下にずらす** (2026-09-10)．
            # 組成部の「行番号は共有し，y は列ごとに持つ」と同じ
            name_dy = header_lines.name_offset(
                hinfo.get('spans', []), hinfo.get('values', []), pitch)
            if abs(name_dy) >= 3:
                warnings.append(
                    f'表頭の項目名は値より {name_dy:+.0f} px ずれて組まれているので，'
                    '項目名の帯だけそのぶんずらした(帯の対応は同じ)．')
            if len(ocr_edges) != len(h_edges):
                warnings.append(
                    f'表頭の項目行は，検出({len(h_edges) - 1} 行)と'
                    f'OCR の検出器の箱の行({len(ocr_edges) - 1} 行)で数が違う．'
                    'OCR の側を採った．段階1で行の対応を目で確かめる．')
            h_edges = ocr_edges
            n_items = len(h_edges) - 1
        else:
            new_edges = _header_bands_from_names(
                img, (name_x1, h_edges[0], name_x2, h_edges[-1]))
            if new_edges is not None and len(new_edges) >= 3:
                if len(new_edges) != len(h_edges):
                    warnings.append(
                        f'表頭の項目行は，検出({len(h_edges) - 1} 行)と'
                        f'項目名の字の行({len(new_edges) - 1} 行)で数が違う．'
                        '項目名の側を採った(OCR の箱は 3 行未満)．段階1で行の対応を目で確かめる．')
                h_edges = new_edges
                n_items = len(h_edges) - 1
    h_notes = axis_notes(n=n_items)
    # **表頭は組成部と横位置がずれることがある**(2026-09-04 ユーザ指摘)．
    # s01115_22_p2(常在度の総合表)は表頭の値が組成部より 18 px 左に組まれており，
    # 組成部の境をそのまま使うと表頭の値を割る．**列の数は変えず**，
    # 表頭の黒画素がいちばん合う位置へ境ごとずらす(数が変わると
    # 表頭の項目と地点の対応が崩れる)
    hx_edges, shift = _shift_edges_for_header(
        x_edges, img, [(float(a), float(b))
                       for a, b in zip(h_edges[:-1], h_edges[1:])])
    if shift:
        warnings.append(
            f'表頭は組成部と横位置が {shift:+d} px ずれているので，'
            '表頭の列の境だけずらした．段階1で表頭の値の切れ目を目で確かめる')
    out = [coord_item(hx_edges, h_edges, obj_name='header_value',
                      y_notes=h_notes)]
    if hc is None:
        warnings.append(
            f"'{class_hdr_col}' が検出されず，表頭の項目名を出力できない．"
            '値だけでは，どの行が何の項目か決められない．')
        return out
    left, right = float(hc['x1'].iloc[0]), float(hc['x2'].iloc[0])
    # **独文と和文の境は，検出枠でなく黒画素で決める**(対策 B．2026-09-09)．
    # 枠は 2 つ検出されることがあり，その和が境になると和文にかかって和名が割れるか，
    # 和文の右に出て独文と和文が 1 列にまとまる(146 表で 43 表の境が字を横切っていた)．
    # 枠は**領域の左端を決めるためだけ**に使う(col_edges.fix_column_edges と同じ考え)
    from . import header_cols
    split = header_cols.split_x(
        ink.binarize(img), (left, float(x_edges[0])),
        [(float(a), float(b)) for a, b in zip(h_edges[:-1], h_edges[1:])])
    if split is not None and left + 20 < split < float(x_edges[0]) - 20:
        if abs(split - right) > 5 and not hc_guessed:
            warnings.append(
                f'表頭の独文と和文の境を，検出枠の右端 {right:.0f} px から'
                f'黒画素の谷 {split:.0f} px へ移した．'
                '段階1で和名の列に字がそろって入っているかを目で確かめる')
        right = split
    n_edges = np.asarray(h_edges, dtype=float) + name_dy
    # ずらしたあと，境を**項目名の列の字の間の空白**へ寄せる (組成部の
    # `fit_edges` と同じ考え．中央値のずらしだけでは行ごとの差が残り，003 では
    # 境が項目名の下端の 1〜16 px 内側を通っていた．2026-09-10)
    if img is not None and len(n_edges) >= 3:
        from . import header_lines as _hl
        n_edges = _hl.snap_to_gap(n_edges, ink.binarize(img), (left, float(x_edges[0])),
                                  float(median_h))
    out.append(coord_item(np.array([left, right]), n_edges,
                          obj_name='header_item', y_notes=h_notes))
    # 項目名は2言語で入っていることが多い(header_col はドイツ語で，
    # その右に和名の列がある)．和名の方がOCRも突き合わせも確実なので，
    # 独文と和文の境から値の左端までを別に切り出す
    gap = float(x_edges[0]) - right
    if gap > (right - left) * 0.2:
        out.append(coord_item(np.array([right, float(x_edges[0])]), n_edges,
                              obj_name='header_item_ja', y_notes=h_notes))
    return out


def number_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    セルに行番号と列番号を振る

    段(block)が複数あるときの決まりは次の2つ．
      - 行番号は**段をまたいで通し**にする(右の段は左の段の続きなので)
      - 列番号は**段ごと**に振り直す(同じ地点が段ごとに現れるので)

    Args:
        df: locate_items()の出力(block列が無ければ1つの段とみなす)
    Returns:
        col, row を足したDataFrame
    """
    if 'block' not in df.columns:
        df = df.assign(block=1)
    rows = df[['block', 'y1']].drop_duplicates().sort_values(['block', 'y1'])
    rows['row'] = range(1, len(rows) + 1)
    cols = df[['block', 'x1']].drop_duplicates().sort_values(['block', 'x1'])
    cols['col'] = cols.groupby('block').cumcount() + 1
    return (df.merge(rows, on=['block', 'y1'], how='left')
              .merge(cols, on=['block', 'x1'], how='left'))


def _empty_located(df: pd.DataFrame, warnings: list) -> pd.DataFrame:
    """
    位置を1つも決められなかったときの空の戻り値

    列の構成はlocate_itemsの通常の戻り値に合わせる
    """
    res = pd.DataFrame(columns=['x1', 'x2', 'y1', 'y2', 'obj_name', 'note',
                                'block', 'source_image', 'model'])
    res.attrs['warnings'] = warnings
    return res
