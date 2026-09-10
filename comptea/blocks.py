"""1 枚の紙面を，表と段に分ける

**表**(`split_tables`)と**段**(`split_blocks`)は別のもの．表が 2 つ縦に
並ぶ紙面は，表頭も地点も別なので**決して 1 つにまとめない**．
段は 1 つの表を左右に折り返した組み方で，行番号は段をまたいで続ける．
"""

import numpy as np
import pandas as pd


# 1ページに表が2つ以上ある資料の見分け方(2026-09-01 に手元の全90枚で測って決めた)．
# 表頭(header)・種名の列(sname / species_col)が縦に2つ以上に分かれる画像は
# kinki_037(Tab.64 と Tab.65)の1枚だけで，他の82枚はどれも1つだった．
TABLE_ANCHOR = 'header'                        # 表の始まりの目印


TABLE_CONFIRM = ('sname', 'species_col')       # 裏を取る目印(表頭の誤検出よけ)


def _vertical_groups(df: pd.DataFrame):
    """y で重なる箱をひとかたまりにし，かたまりごとの上端を返す"""
    if df.empty:
        return []
    tops, bottom = [], None
    for _, r in df.sort_values(by='y1').iterrows():
        if bottom is None or r['y1'] > bottom:
            tops.append(float(r['y1']))
            bottom = float(r['y2'])
        else:
            bottom = max(bottom, float(r['y2']))
    return tops


def split_tables(df: pd.DataFrame):
    """
    1ページに表が2つ以上あるとき，表ごとに分ける

    `split_blocks()` が分ける「段」とは別物なので混同しない．

    - **段**は横に並ぶ．同じ表を折り返したもので，**地点は同じ**．
    - **表**は縦に並ぶ．**別の表**なので表頭も地点も違う．

    したがって表は最後まで別々に扱う(1つの組成表にまとめてはいけない)．
    切る位置は2つ目以降の表頭の上端で，箱は中心の y で振り分ける．

    Args:
        df: 検出結果
    Returns:
        (表ごとのDataFrameのリスト(上から下の順), 警告のリスト)．
        表が1つなら要素1つのリスト
    """
    tops = _vertical_groups(df[df['obj_name'] == TABLE_ANCHOR])
    if len(tops) < 2:
        return [df], []
    # 表頭だけで切ると，本文の見出しを拾った誤検出で表が割れる．
    # 種名の列も同じ数に分かれているときだけ切る
    if not any(len(_vertical_groups(df[df['obj_name'] == name])) == len(tops)
               for name in TABLE_CONFIRM):
        return [df], [
            f'表頭が縦に {len(tops)} つに分かれているが，種名の列は分かれていない．'
            '1ページに表が2つある形とは見なさず，1つの表として扱った．'
            'もし表が2つあるなら，種名の列が拾えていない']
    cuts = np.array(tops[1:], dtype=float)
    centers = (df['y1'].to_numpy(dtype=float) + df['y2'].to_numpy(dtype=float)) / 2
    which = np.searchsorted(cuts, centers)
    tables = [df[which == i] for i in range(len(tops))]
    return tables, [
        f'このページには表が {len(tables)} つある(表頭と種名の列が縦に分かれている)．'
        '**別々の表**なので，1つずつ格子を作り，別々の組成表として出力する']


STRAY_OVERLAP = 0.5       # 段の目印は，いちばん高い段と縦にこの割合 (自分の高さ比) 以上重なること


def _anchor_groups(found):
    """x が重なる目印をまとめ，段ごとの範囲 (x1, x2, y1, y2, 元の index) を左から返す"""
    groups = []
    for idx, r in found.sort_values(by='x1').iterrows():
        if groups and r['x1'] <= groups[-1]['x2']:
            g = groups[-1]
            g['x2'] = max(g['x2'], r['x2'])
            g['y1'] = min(g['y1'], r['y1'])
            g['y2'] = max(g['y2'], r['y2'])
            g['idx'].append(idx)
        else:
            groups.append(dict(x1=r['x1'], x2=r['x2'], y1=r['y1'], y2=r['y2'], idx=[idx]))
    return groups


def drop_stray_anchors(df: pd.DataFrame, anchors=('sname', 'species_col'),
                       overlap=STRAY_OVERLAP):
    """段の目印のうち，いちばん高い段と縦に重ならないものを捨てる

    表の下の注記などが `sname` として検出されると，それが段を 1 つ増やし
    (s01115_04_p2 の y 6447〜6596 の 149 px)，本物の段の右の限界になって右端の
    地点の列が足せなくなる (指摘 13・35．2026-09-10)．折り返した段は左の段と同じ
    高さから始まるので，縦に重ならない目印は段ではない．
    **同じ列の目印が縦に分かれて検出されたもの (x が重なる) は 1 つの段にまとめて
    から見る** (01_p3・12_p1 などは 1 列が 2〜3 個の箱になっている)．

    Returns:
        (df, 捨てた目印の数)
    """
    for anchor in anchors:
        found = df[df['obj_name'] == anchor]
        if not found.empty:
            break
    else:
        return df, 0
    groups = _anchor_groups(found)
    if len(groups) < 2:
        return df, 0
    tallest = max(groups, key=lambda g: g['y2'] - g['y1'])
    drop = []
    for g in groups:
        if g is tallest:
            continue
        h = g['y2'] - g['y1']
        ov = min(g['y2'], tallest['y2']) - max(g['y1'], tallest['y1'])
        if h <= 0 or ov < h * overlap:
            drop.extend(g['idx'])
    if not drop:
        return df, 0
    return df.drop(index=drop), len(drop)


ALIGN_MIN_ROWS = 0.5      # 右の段がこれ (行) 以上短いときだけ，左の段の下端まで足す


BODY_CLASSES = ('comp', 'sname', 'species_col', 'layer')


def _renumber_rows(df):
    """段ごとに y1 の順で行番号を振り直す(`locate.number_cells` と同じ決まり)"""
    rows = df[['block', 'y1']].drop_duplicates().sort_values(['block', 'y1'])
    rows['row'] = range(1, len(rows) + 1)
    return df.drop(columns=['row']).merge(rows, on=['block', 'y1'], how='left')


def align_block_bottoms(df: pd.DataFrame):
    """右の段の下端を左の段にそろえる (2026-09-10 ユーザ指示)

    種名のリストを折り返した組み方では，右の段の行は左の段と同じ高さに並ぶ．
    右の段の箱 (`row`・`species_col`) が最後の行まで届かないと，その行が丸ごと落ちる
    (kinki_060 の「シナダレススメガヤ +・2」)．左の段の行のうち右の段の下端より下に
    あるものを，右の段に写す (note は 'interpolated')．
    右の段が本当に短い表 (084・056・068・043) では空の行が付くだけで，OCR が
    空白を返す (ユーザ了承)．右の段が長いときは触らない (047・035 は左が本当に短い)．
    **行の高さの後処理 (row_heights) の後に呼ぶ**．前に足すと，下端の詰めが
    「組成に字が無い末尾の行」として落とす (060 で 1 行足しても最終の格子に残らなかった)．

    Args:
        df: 格子 (block・row・col が付いたもの)
    Returns:
        (格子, 足した行の数)
    """
    if 'block' not in df.columns or 'row' not in df.columns:
        return df, 0
    body = df[df['obj_name'].isin(BODY_CLASSES)]
    if body.empty or body['block'].nunique() < 2:
        return df, 0
    blocks = sorted(body['block'].unique())
    first = body[body['block'] == blocks[0]]
    rows1 = (first.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
             .sort_values('y1'))
    bottom1 = float(rows1['y2'].max())
    ys1 = rows1['y1'].to_numpy(dtype=float)
    pitch1 = float(np.median(np.diff(ys1))) if len(ys1) >= 2 else 0.0
    added, total = [], 0
    for b in blocks[1:]:
        bb = body[body['block'] == b]
        bottom = float(bb['y2'].max())
        ys = np.sort(bb['y1'].unique())
        pitch = float(np.median(np.diff(ys))) if len(ys) >= 2 else pitch1
        if not pitch > 0 or bottom1 - bottom < pitch * ALIGN_MIN_ROWS:
            continue
        # 左の段の行のうち，右の段の下端より下にあるもの (中心で見る) を写す
        cand = rows1[(rows1['y1'] + rows1['y2']) / 2 > bottom]
        if cand.empty:
            continue
        bands = [[float(a), float(c)] for a, c in zip(cand['y1'], cand['y2'])]
        bands[0][0] = bottom                    # 最後の行に続ける (隙間も重なりも作らない)
        last = bb[bb['row'] == bb['row'].max()]
        for a, c in bands:
            for r in last.to_dict('records'):
                cell = dict(r)
                cell.update(y1=a, y2=c, note='interpolated')
                added.append(cell)
        total += len(bands)
    if not added:
        return df, 0
    out = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    return _renumber_rows(out), total


def split_blocks(df: pd.DataFrame, anchors=('sname', 'species_col')):
    """
    1ページに複数の段(表のかたまり)があるとき，段ごとに分ける

    1地点だけの組成表は縦に長くなるので，紙面を節約するために
    種名のリストを左右2段に折り返して組むことがある
    (ラベル付き33枚のうち11枚がこの形)．
    右の段は左の段の**続き**で，地点は同じ．

    段の左端は種名の列(sname，無ければspecies_col)なので，
    その位置で切る．

    Args:
        df: 検出結果
        anchors: 段の左端とみなすクラス．先に見つかった方を使う
    Returns:
        段ごとのDataFrameのリスト(左から右の順)．
        段が1つなら要素1つのリスト
    """
    for anchor in anchors:
        found = df[df['obj_name'] == anchor]
        if not found.empty:
            break
    else:
        return [df]
    # 重なる検出は同じ段のものとしてまとめる
    found = found.sort_values(by='x1')
    lefts, right = [], None
    for _, r in found.iterrows():
        if right is None or r['x1'] > right:
            lefts.append(r['x1'])
            right = r['x2']
        else:
            right = max(right, r['x2'])
    if len(lefts) < 2:
        return [df]
    cuts = np.array(lefts[1:], dtype=float)
    centers = (df['x1'].to_numpy(dtype=float) + df['x2'].to_numpy(dtype=float)) / 2
    which = np.searchsorted(cuts, centers)
    return [df[which == i] for i in range(len(lefts))]
