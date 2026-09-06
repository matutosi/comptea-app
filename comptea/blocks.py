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
