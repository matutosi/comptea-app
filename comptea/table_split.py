"""1 枚の紙面に載る，別々の表を見分ける

空白の帯では切れない切れ目を，**検出の手掛かり**から見つける
(左右に並ぶ表の仕切りは 23 px しかないことがあり，1 つの表の内部にできる
隙間より狭い)．種名の列が横に 2 つあって表頭も 2 つなら別々の表，
種名の列が 2 つでも表頭が 1 つなら 1 つの表を折り返したもの．

表頭の外に出た検出(凡例や地の文に付いた `plot_row`・種名の列)を捨てるのも
ここ．残すと組成部の行を二重に切り，表頭の帯が紙面の全高に広がる．
"""

import numpy as np


def _x_groups(df, anchors=('sname', 'header_col', 'species_col')):
    """横に離れたかたまりの (x1, x2) を，左から順に返す"""
    for a in anchors:
        found = df[df['obj_name'] == a]
        if not found.empty:
            break
    else:
        return []
    found = found.sort_values('x1')
    out = []
    for _, r in found.iterrows():
        if out and r['x1'] <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], r['x2']))
        else:
            out.append((r['x1'], r['x2']))
    return out


def split_side_by_side(df_det):
    """1枚に**左右に並んだ別々の表**があるなら，検出を表ごとに分ける

    `s01115_02` は Tab.8 と Tab.9 が左右に並ぶ．**仕切りは 23 px しかなく**，
    地点と地点のあいだの隙間(20 px)と見分けが付かないので，
    `split_sheet.py` の空白の帯では切れない．検出で見分けるしかない．

    見分け方は「**種名の列が横に離れて2つ以上あり，表頭もその数だけある**」．
    1つの表を左右2段に折り返した紙面(手元の 33 枚中 11 枚)は，
    表頭が1つしかないので，この条件では切れない．

    Returns:
        (表ごとの DataFrame のリスト(左から右), 警告のリスト)
    """
    names = _x_groups(df_det)
    if len(names) < 2:
        return [df_det], []
    heads = _x_groups(df_det, anchors=('header',))
    if len(heads) != len(names):
        return [df_det], [
            f'種名の列が横に {len(names)} つに分かれているが，表頭は {len(heads)} つ．'
            '**1つの表を左右に折り返した紙面**とみなし，分けなかった']
    # 切れ目は**次の表の種名の列の左端**．2つの種名の列の中間で切ると，
    # 左の表の組成部(種名の列よりずっと右まで伸びている)を割ってしまう
    # (s01115_02 では中間 x2284 で切り，左の表が 11 列に減った．2026-09-02)
    cuts = [names[i + 1][0] for i in range(len(names) - 1)]
    centers = (df_det['x1'].to_numpy(dtype=float)
               + df_det['x2'].to_numpy(dtype=float)) / 2
    which = np.searchsorted(np.array(cuts, dtype=float), centers)
    out = [df_det[which == i] for i in range(len(names))]
    return out, [
        f'**この画像には表が左右に {len(out)} つ並んでいる**'
        '(種名の列も表頭もその数だけある)．別々の表として1つずつ組み立てる．'
        '仕切りが細くて画像の段階では切れなかったので，段階1で境目を目で確かめる']


def side_by_side_cuts(df_det):
    """左右に並んだ表の切れ目(x)を返す．`split_side_by_side()` と同じ判定"""
    names = _x_groups(df_det)
    if len(names) < 2:
        return []
    heads = _x_groups(df_det, anchors=('header',))
    if len(heads) != len(names):
        return []
    return [int(names[i + 1][0]) for i in range(len(names) - 1)]


NAME_BAND_K = 3.0       # 種名の列とみなす黒画素の量．地点の列の中央値に対する比


NAME_BAND_MIN_COLS = 2  # 密な列がこれ以上つづくと帯とみなす


NAME_BAND_SIDE = 4      # 帯の両側に，地点の列がこれ以上ある(端の帯は階層や常在度)


NAME_BAND_HEAD_K = 1.5  # 帯の上の表頭の黒画素．表頭の列の中央値に対する比


def name_band_cuts(dark, df_det, k=NAME_BAND_K, min_cols=NAME_BAND_MIN_COLS,
                   side=NAME_BAND_SIDE, head_k=NAME_BAND_HEAD_K):
    """組成部の中に**字の密な帯**(隠れた種名の列)があれば，その左端 x を返す

    表と表の間隔が 50 px しかないと(s01115_23 の Tab.141 と Tab.148)，
    画像の空白でも検出でも分けられない．右の表の種名の列は `sname` として
    検出されず，左の表の組成部の続きに見える．
    ただし**黒画素からは見える**．地点の列は「・」や 1-2 字なので薄く，
    種名の列は行ごとに字が詰まって濃い．23_p1_t1 では格子の列 29-31 が
    中央値の 3.5-4.4 倍で，他の列は 1 倍前後だった(2026-09-03)．
    66 表で数えると，この帯があるのは 23_p1_t1 と，左端の列 0-1 が濃い
    05_p1_t2 だけ．**両側に地点の列がある内側の帯**に限れば 23 だけに当たる．

    列の幅は `col` の検出の幅の中央値で刻む(格子はまだ無い)．

    Returns:
        帯の左端 x の並び(左から)．無ければ []
    """
    col = df_det[df_det['obj_name'] == 'col']
    if len(col) < side * 2 + min_cols:
        return []
    w = float(np.median(col['x2'] - col['x1']))
    if w <= 0:
        return []
    x1, x2 = int(col['x1'].min()), int(col['x2'].max())
    rows = df_det[df_det['obj_name'] == 'row']
    y1 = int(rows['y1'].min()) if len(rows) else int(col['y1'].min())
    y2 = int(rows['y2'].max()) if len(rows) else int(col['y2'].max())
    n = int((x2 - x1) // w)
    if n < side * 2 + min_cols:
        return []
    fr = np.array([dark[y1:y2, int(x1 + i * w):int(x1 + (i + 1) * w)].mean()
                   for i in range(n)])
    med = float(np.median(fr))
    if med <= 0:
        return []
    # **帯の上の表頭にも字があること**．隠れた表の種名の列の上には，その表の
    # 項目名(調査番号・標高…)が並ぶ．常在度の列(`III(+-2)` のように字が詰まる)
    # も組成部では密だが，その上の表頭は空に近い．実測(2026-09-03)
    #   23_p1 の種名の帯: 表頭の黒 1.7-2.0 倍   21_p3 の常在度の帯: 0.9-1.2 倍
    # 21_p3 は地点 26 の 1 表なのに，常在度の列で 16 + 10 列に割れていた
    head = df_det[df_det['obj_name'] == 'header']
    if head.empty or int(head['y1'].min()) >= y1:
        return []
    hy1 = int(head['y1'].min())
    hb = np.array([dark[hy1:y1, int(x1 + i * w):int(x1 + (i + 1) * w)].mean()
                   for i in range(n)])
    hmed = float(np.median(hb))
    if hmed <= 0:
        return []
    dense = fr >= med * k
    cuts, i = [], 0
    while i < n:
        if not dense[i]:
            i += 1
            continue
        j = i
        while j < n and dense[j]:
            j += 1
        if (j - i >= min_cols and i >= side and n - j >= side
                and float(hb[i:j].mean()) >= hmed * head_k):
            cuts.append(int(x1 + i * w))
        i = j
    return cuts


STRAY_TOL_ROWS = 1.5        # 表頭の帯のゆとり(項目行の高さの倍数)


NAME_CLASSES = ('sname', 'species_col')


def drop_stray_name_cols(df):
    """表頭の中にある種名の列の検出を捨てる(2026-09-05)

    表頭の凡例や表題の字が `sname` として検出されることがある．
    `split_blocks()` は種名の列で段を分けるので，それが**段を 1 つ増やし**，
    本物の種名の列がその段に取られて，本体の段から和名が消える
    (s01115_17_p1 は x926-1635 y244-491 の偽の `sname` で段が 2 つになり，
     和名が 1 つも出なかった)．

    Returns:
        (残した検出, 捨てた本数)
    """
    heads = df[df['obj_name'] == 'header']
    names = df[df['obj_name'].isin(NAME_CLASSES)]
    if heads.empty or len(names) < 2:
        return df, 0
    # **表頭より上に出たものも捨てる**．表題の欄に出ることがあり
    # (17_p1 の偽の `sname` は y244-491 で，表頭 y1187-1509 の上だった)，
    # 帯の中だけを見ていては拾えない
    hi = float(heads['y2'].max())
    center = (names['y1'] + names['y2']) / 2
    inside = names[center <= hi]
    # 全部が表頭の中なら，表頭の判定の方を疑う(捨てない)
    if inside.empty or len(inside) == len(names):
        return df, 0
    return df.drop(index=inside.index), len(inside)


def drop_stray_plot_rows(df, ref=None, heads=True):
    """表頭の外に出た `plot_row`(と `header`)を捨てる

    `heads=False` なら `plot_row` だけを捨てる．紙面全体の検出でも項目行は
    組成部の全高に散る(s01115 の 68 表中 16 表．18_p2 は 62 本のうち 58 本が
    表頭の外で，`header_value` が 1,386 セル・紙面の全高に広がり，
    組成部の行が表頭の値として二重に切られていた．2026-09-04)．
    そこでは `header` は残す(凡例の表頭を捨てると，行の選別や
    組成部の範囲の決め方が変わる)．

    表頭の項目行は組成部の行と見た目が似ているので，短冊のどこででも
    検出される．表の全高に散らばったまま渡すと，`locate.py` が
    **表頭が表の全高にある**とみなし，項目行が 141 行になる
    (2026-09-02 に p3 で起きた．本当は 17 項目)．
    `locate.py` が `row` について同じ選別をしているのと対になる処理．

    **表頭の帯は，表全体を1度検出した結果(`ref`)から決める**(2026-09-03)．
    短冊の検出は組成部の中に `header` そのものを作ることがあり
    (s01115_22_p2 は y 2088-3027 と 3608-4721 に偽の表頭)，短冊の `header`
    で帯を決めると帯が表の全高に広がって，この選別が効かない．
    偽の表頭は `_locate_header()` が表頭の格子にし，組成部が表の下 1/4 を
    失った(145 行 → 63 行)．表全体の検出は縮尺が学習時と同じで，
    表頭の位置は安定している．`ref` に表頭が無ければ短冊の表頭で決める
    """
    src = ref if ref is not None and (ref['obj_name'] == 'header').any() else df
    head = src[src['obj_name'] == 'header']
    if head.empty:
        return df, []
    # **項目名の列に重なる表頭だけを帯にする**(2026-09-03)．
    # 表頭は表の下の凡例や地の文にも出るので，全部の和を取ると帯が紙面の
    # 全高に広がり，この選別そのものが効かなくなる
    # (19_p1 は y 0-171・237-522・657-1394・11089-11919 の 4 つが出て
    #  帯が y 0-11919 になり，本体の行を拾った `plot_row` が 62 本残った)．
    # 項目名の列(`header_col`)は表頭にしか無く，1 つしか出ない
    hcol = src[src['obj_name'] == 'header_col']
    if not hcol.empty:
        a, b = float(hcol['y1'].min()), float(hcol['y2'].max())
        near = head[(head['y1'] <= b) & (head['y2'] >= a)]
        if not near.empty:
            head = near
    lo, hi = float(head['y1'].min()), float(head['y2'].max())
    # **帯には項目行の高さ 1.5 倍のゆとりを持たせる**(2026-09-04)．`header` の
    # 箱は表頭の最初の 1-2 行(通し番号・調査番号)を取りこぼすことがあり
    # (kinki_008・072)，中心で厳密に切ると本物の項目行を捨てる．
    # 組成部に散った偽の項目行は表頭から何行も離れているので，これで
    # 拾い直すことはない
    rows = df[df['obj_name'] == 'plot_row']
    if not rows.empty:
        tol = float((rows['y2'] - rows['y1']).median()) * STRAY_TOL_ROWS
        lo, hi = lo - tol, hi + tol
    center = (df['y1'] + df['y2']) / 2
    outside = (center < lo) | (center > hi)
    bad_row = (df['obj_name'] == 'plot_row') & outside
    bad_head = (df['obj_name'] == 'header') & outside & heads
    out, warn = df, []
    if bad_row.any():
        warn.append(f"表頭の外にあった 'plot_row' の検出 {int(bad_row.sum())} 本を"
                    '捨てた(組成部の行と見た目が似ているため誤検出されやすい)．')
    if bad_head.any():
        warn.append(f"組成部の中に出た 'header' の検出 {int(bad_head.sum())} 本を"
                    '捨てた(表全体の検出では表頭は '
                    f'y {int(lo)}-{int(hi)} だけ)．')
    if bad_row.any() or bad_head.any():
        out = df[~(bad_row | bad_head)]
    return out, warn
