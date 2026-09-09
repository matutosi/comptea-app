"""検出の選別

**表として組む前に，組んではいけないものを外す**．紙面の隅の札や凡例の帯
(表頭も種名の列も無い)，項目行にも項目名の列にも重ならない `header`
(組成部の最初の種群を覆っていることがある)，「1回出現種」の中に出た `row`，
そして同じものを二度検出した箱．

残すと**黙って誤ったデータ**になる: 偽の表頭は種群を丸ごと飲み込み，
重複した列は地点の番号をずらす．
"""

import pandas as pd


# 組成表でないページとみなす 'row' の上限(これ未満なら表ではない)．
# 手元の組成表は，最も少ないページでも行が数十本ある
NO_TABLE_MAX_ROWS = 5    # 種名の列のセルがこれ未満なら，組成表の載っていないページ


FRAGMENT_CLASSES = ('header', 'plot_row', 'header_col', 'sname',
                    'species_col', 'layer', 'once_species')


def looks_like_fragment(df_det) -> bool:
    """検出が `table`・`row`・`col` だけで，表頭も種名の列も無いなら切れ端

    折り込みを切り分けると，シートの隅の札と表題・凡例(流し込み)だけの
    細い帯が「表」として残ることがある(s01115_21_p2，3041 x 527 px)．
    そこに `row` 3 本と `col` 2 本が出て 5 x 2 の格子ができ，
    表の数が 1 つ多く数えられていた(2026-09-03)．
    組成表なら表頭か種名の列のどれかは必ず検出される．
    """
    names = set(df_det['obj_name']) if len(df_det) else set()
    return not (names & set(FRAGMENT_CLASSES))


def looks_like_no_table(df_loc: pd.DataFrame) -> bool:
    """組成表の載っていないページか(位置決めの結果から見る)

    手元の資料には，本文・写真・隣のページから続く流し込みだけの
    ページが混ざっている(2026-09-01 に 88 枚中 9 枚を確認)．
    これを失敗に数えると切り出しの実力を過小に見せ，閾値の比較もゆがむ．

    **本物の表なのに 'col' が取れない段**(kinki_047 の型)を
    巻き込んで隠さないよう，判定は次の3つをすべて満たすときだけにする．
      - ページ全体で組成セルが1つも無い(段が1つでも作れていれば表とみなす)
      - 'col' が1本も無い
      - **種名の列(学名・和名)のセルが NO_TABLE_MAX_ROWS 未満**(本物の表なら数十行ある)

    3つ目は 'row' の本数で見ていたが，`locate_items()` は `obj_name` に 'row' を
    **一度も作らない**ので常に 0 で，例外は一度も発動していなかった(2026-09-09)．
    そのため 1 調査区の 2 段組(kinki_047 は種名の列 120 行，053 は 100 行)が
    「組成表が見あたらない」として格子ごと失われていた．

    検出が0件のときも表ではないので，空の表を渡せば True を返す．

    規則をここに1つだけ置き，scan_blocks.py とスキルの
    run_pipeline.py の両方から呼ぶ(2か所に書くとずれる)．
    """
    if df_loc is None or len(df_loc) == 0 or 'obj_name' not in df_loc:
        return True
    name = df_loc['obj_name']
    n_body = int(name.isin(('sname', 'species_col')).sum())
    return (not (name == 'comp').any()
            and not (name == 'col').any()
            and n_body < NO_TABLE_MAX_ROWS)


def filter_results(df: pd.DataFrame, 
    source_image: str = "",
    obj_name: str = "") -> pd.DataFrame:
    """ filter by file name and obj_name """
    if source_image:
        df = df[df['source_image'] == source_image]
    if obj_name:
        df = df[(df['obj_name'] == obj_name)]
    return df


HEADER_SUPPORT = ('plot_row', 'header_col')


def drop_unsupported_headers(df: pd.DataFrame):
    """項目行にも項目名の列にも裏を取れない `header` を捨てる

    表頭の箱が組成部の最初の種群を覆うことがある(s01115_01_p1 で
    conf 0.32 の 2 つ目の `header` が y 794-1328 に出た．本物は y 10-719，
    conf 0.93)．表頭の帯がそこまで広がると，`drop_rows_outside_body()` が
    その種群の行を「表頭の中の row」として捨て，16 行が格子から落ちた
    (2026-09-03)．

    本物の表頭には項目行(`plot_row`)か項目名の列(`header_col`)が重なる．
    どちらの中心もその箱に入らない `header` は偽物とみなす．
    ただし**裏の取れた `header` が1つも無いときは何もしない**
    (1地点の表の表頭は流し込みで，項目行として検出されない)．

    Returns:
        (残した検出, 捨てた本数)
    """
    heads = df[df['obj_name'] == 'header']
    sup = df[df['obj_name'].isin(HEADER_SUPPORT)]
    if len(heads) < 2 or sup.empty:
        return df, 0
    centers = (sup['y1'].to_numpy(dtype=float) + sup['y2'].to_numpy(dtype=float)) / 2
    supported = {
        i: bool(((centers >= r.y1) & (centers <= r.y2)).any())
        for i, r in heads.iterrows()}
    if not any(supported.values()):
        return df, 0
    drop = [i for i, ok in supported.items() if not ok]
    return df.drop(index=drop), len(drop)


def drop_rows_outside_body(df: pd.DataFrame, overlap: float = 0.5):
    """
    表頭や「1回出現種」の中にある row の検出を捨てる

    どちらも組成部の行と見た目が似ていて，`row` として誤検出される．
    領域そのものは別クラスで検出できるので，位置で捨てられる．
      kinki_008: 1回出現種の1行目を row と誤検出し，行が1つ増えていた
      kinki_071: 表頭の項目行を row と誤検出していた

    Args:
        df: 検出結果
        overlap: 行の高さのこの割合以上が領域に入っていたら捨てる
    Returns:
        (残した検出, 捨てた本数)
    """
    regions = df[df['obj_name'].isin(['header', 'once_species'])]
    rows = df[df['obj_name'] == 'row']
    if regions.empty or rows.empty:
        return df, 0
    drop = []
    # **表頭より上に出た `row` も捨てる**(2026-09-03)．表題や凡例の字を
    # 行と見た誤検出で，1 本あるだけで格子が上へ伸び，表頭を丸ごと 1 行に
    # 飲み込む(s01115_07_p1 は行の高さが 1,246 px と，他の行の 37 倍)．
    # 領域の中でなく**外(上)**なので，下の重なりの判定では拾えない
    # 表頭の目印には `header_col` も数える(2026-09-04)．`header` が出ない
    # 表頭があり(s01115_23_p1_t2 の上段)，下段の表の `header` だけで上端を
    # 決めると，上段の行 116 本が「表頭より上」として全部捨てられた
    head = df[df['obj_name'].isin(['header', 'header_col'])]
    if not head.empty:
        top = float(head['y1'].min())
        above = rows[(rows['y1'] + rows['y2']) / 2 < top]
        drop.extend(above.index.tolist())
    for r in rows.itertuples():
        height = r.y2 - r.y1
        if height <= 0:
            continue
        if r.Index in drop:
            continue
        for g in regions.itertuples():
            inside = min(r.y2, g.y2) - max(r.y1, g.y1)
            if inside > 0 and inside / height >= overlap:
                drop.append(r.Index)
                break
    return df.drop(index=drop), len(drop)


def remove_dup_ranges(df, threth=0.8, threth_col=None, threth_row=None):
    """
    remove_dup_rangeのラッパー

    colとrowの両方で重複の多い推定を削除する
    """
    if threth_col is None:
        threth_col = threth
    if threth_row is None:
        threth_row = threth
    df_col = df[(df['obj_name'] == "col")]
    df_row = df[(df['obj_name'] == "row")]
    df_col = remove_dup_range(df_col, item="col", threth=threth_col)
    df_row = remove_dup_range(df_row, item="row", threth=threth_row)
    df_others = df[(df['obj_name'] != 'col') & (df['obj_name'] != 'row')]
    df = pd.concat([df_col, df_row, df_others], ignore_index=True)
    return df


def remove_dup_range(df, item="col", threth=0.8):
    """
    dfの重複する範囲から，confidenceの小さい方を削除
    
    Args:
        df (pd.DataFrame): 処理対象のデータフレーム。
        item (str): "col" or "row"
        threth (float): 範囲が重複しているとみなす閾値。
    
    Returns:
        pd.DataFrame: 重複行が削除されたデータフレーム。
    
    Examples
        data = {
            'obj_name': ['col', 'col', 'col', 'col', 'col', 'col', 'col', 'col'],
            'obj_class': [3, 3, 3, 3, 3, 3, 3, 3],
            'confidence': [0.73, 0.39, 0.38, 0.33, 0.33, 0.31, 0.31, 0.27],
            'x1': [1839, 1743, 1930, 2023, 1639, 2002, 2124, 2039],
            'y1': [537, 490, 610, 776, 539, 480, 630, 690],
            'x2': [1921, 1836, 2020, 2113, 1747, 2098, 2218, 2119],
            'y2': [2894, 2560, 2794, 2714, 2779, 2892, 2775, 2871],
            'source_image': ['images/example.jpg'] * 8,
            'model': ['comptea.pt'] * 8
        }
        df = pd.DataFrame(data)
        df
        df_removed = remove_dup_range(df)
        df_removed
    """
    if item=="col":
        ranges=['x1', 'x2']
    if item=="row":
        ranges=['y1', 'y2']
    df = df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
    to_drop = set()
    for i in range(len(df)):
        if i in to_drop:
            continue
        loc1 = df.iloc[i]
        val1_1, val1_2 = loc1[ranges[0]], loc1[ranges[1]]
        for j in range(i + 1, len(df)):
            if j in to_drop:
                continue
            loc2 = df.iloc[j]
            val2_1, val2_2 = loc2[ranges[0]], loc2[ranges[1]]
            # 範囲の重複を計算
            intersection_start = max(val1_1, val2_1)
            intersection_end = min(val1_2, val2_2)
            intersection_len = max(0, intersection_end - intersection_start)
            range1_len = val1_2 - val1_1
            range2_len = val2_2 - val2_1
            #一方の範囲に対する重複率がthrethを超えるか?
            if range1_len > 0 and intersection_len / range1_len > threth:
                to_drop.add(j)
            elif range2_len > 0 and intersection_len / range2_len > threth:
                to_drop.add(j)
    return df.drop(index=list(to_drop)).reset_index(drop=True)
