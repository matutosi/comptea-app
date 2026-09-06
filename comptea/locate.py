import cv2
import numpy as np
import pandas as pd
from PIL import Image

import ink

# 組成表でないページとみなす 'row' の上限(これ未満なら表ではない)．
# 手元の組成表は，最も少ないページでも行が数十本ある
NO_TABLE_MAX_ROWS = 5


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

    **本物の表なのに 'col' が取れない段**(旧 kinki_047 の型)を
    巻き込んで隠さないよう，判定は次の3つをすべて満たすときだけにする．
      - ページ全体で組成セルが1つも無い(段が1つでも作れていれば表とみなす)
      - 'col' が1本も無い
      - 'row' が NO_TABLE_MAX_ROWS 本未満(本物の表なら数十本ある)

    検出が0件のときも表ではないので，空の表を渡せば True を返す．

    規則をここに1つだけ置き，scan_blocks.py とスキルの
    run_pipeline.py の両方から呼ぶ(2か所に書くとずれる)．
    """
    if df_loc is None or len(df_loc) == 0 or 'obj_name' not in df_loc:
        return True
    name = df_loc['obj_name']
    return (not (name == 'comp').any()
            and not (name == 'col').any()
            and int((name == 'row').sum()) < NO_TABLE_MAX_ROWS)


def filter_results(df: pd.DataFrame, 
    source_image: str = "",
    obj_name: str = "") -> pd.DataFrame:
    """ filter by file name and obj_name """
    if source_image:
        df = df[df['source_image'] == source_image]
    if obj_name:
        df = df[(df['obj_name'] == obj_name)]
    return df

def locate_x_range(df: pd.DataFrame,
    source_image: str = "",
    obj_name: str = "col"):
    """
    obj_nameのx1とx2の範囲

    Returns:
        検出があるとき: x1, x2を1行持つDataFrame
        検出が無いとき: None(0,0の範囲を作らない．幅ゼロの領域をでっちあげないため)
    """
    results = filter_results(df, source_image, obj_name)
    if results.empty:
        return None
    return pd.DataFrame({'x1': [results['x1'].min()], 'x2': [results['x2'].max()]})

def locate_y_range(df: pd.DataFrame,
    source_image: str = "",
    obj_name: str = "row"):
    """
    obj_nameのy1とy2の範囲

    Returns:
        検出があるとき: y1, y2を1行持つDataFrame
        検出が無いとき: None(0,0の範囲を作らない)
    """
    results = filter_results(df, source_image, obj_name)
    if results.empty:
        return None
    return pd.DataFrame({'y1': [results['y1'].min()], 'y2': [results['y2'].max()]})

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


def _merge_close(centers: np.ndarray, min_gap: float) -> np.ndarray:
    """
    近すぎる中心をまとめる

    同じ行(列)を2回検出したものを1つにする．
    重なりが小さい重複はremove_dup_range()では残るため，ここで拾う．
    """
    merged = []
    group = [centers[0]]
    for c in centers[1:]:
        if c - group[-1] < min_gap:
            group.append(c)
        else:
            merged.append(np.mean(group))
            group = [c]
    merged.append(np.mean(group))
    return np.array(merged)


def _drop_isolated(starts: np.ndarray, ends: np.ndarray, pitch: float,
                   max_gap_pitches: float = 5):
    """
    他から大きく離れた検出を捨てる

    ページの隅に1本でも誤検出があると，そこまでの隙間を内挿で埋めてしまう．
    いちばん大きな塊(隣との隙間が max_gap_pitches 以内で繋がる並び)だけを残す．

    **離れ具合は箱と箱の隙間(前の下端 → 次の上端)で測る**．箱の中心の間隔で
    測ってはいけない(2026-09-03)．組成部の谷から作った行の箱は，字の詰まった
    数行(谷が取れない区間)をまたいで 150-500 px の高さになることがある．
    中心で測るとその箱の前後が「5行分より離れている」になり，塊が切れて
    最大の塊以外が丸ごと捨てられていた(s01115_10_p1 は 335 行のうち 113 行しか
    残らず，残り 6653 px を1つの巨大な行が飲み込んでいた)．
    隙間で測れば，連続した箱は隙間 0 なので切れず，本来の用途(kinki_007 の
    2584 px 離れた1本)はそのまま捨てられる．

    Returns:
        (残す箱の bool マスク, 捨てた本数)
    """
    n = len(starts)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep, 0
    limit = pitch * max_gap_pitches
    reach = np.maximum.accumulate(ends)     # ここまでの箱の下端の最大
    groups, current = [], [0]
    for i in range(1, n):
        if starts[i] - reach[i - 1] > limit:
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)
    if len(groups) == 1:
        return keep, 0
    best = max(groups, key=len)
    keep[:] = False
    keep[best] = True
    return keep, n - len(best)


def locate_edges(df: pd.DataFrame, source_image: str = "",
                 obj_name: str = "row", axis: str = "y"):
    """
    検出された1本ずつの位置から，分割の境界を組み立てる

    「表全体を等分する」のをやめ，検出そのものを境界に使う．
    隣り合う検出の間隔を代表値(中央値)で割った値を丸めて，
    その区間に何行(何列)あるかを数える．
      1 -> 隣接．そのまま
      2以上 -> 検出漏れ．その区間だけ内挿する
      0 -> 同じ行の重複検出．まとめる(_merge_close)
    行数を「表の高さ ÷ 平均行高」で決めるより，
    検出の増減に強く，ドリフト(途中からのずれ)にも追従する．

    Args:
        df: 検出結果．重複除去済みであること
        source_image: 対象の画像
        obj_name: 'row' または 'col'
        axis: 'y'(行) または 'x'(列)
    間隔の代表値は2通りで見積もる．
      - 検出どうしの間隔の中央値: 検出漏れが多いと「2行分」を1行と見誤る
      - 検出の箱の大きさの中央値: 漏れの影響を受けないが，
        ラベルが行より小さく付けられていると過小になる
    小さい方を採る(漏れによる過大を避ける)．
    2つが食い違うときは，どちらが正しいか決められないので警告する．

    Returns:
        (edges, interpolated, warnings)
        edges       : 長さ n+1 の境界．n が行数(列数)
        interpolated: 長さ n の bool．内挿で作ったセルが True
        warnings    : 気づいたことの文字列のリスト
        検出が2本未満で間隔を出せないときは (None, None, warnings)
    """
    a, b = ('y1', 'y2') if axis == 'y' else ('x1', 'x2')
    warnings = []
    res = filter_results(df, source_image, obj_name)
    if res.empty:
        return None, None, warnings
    if len(res) == 1:
        # 1地点だけの組成表はcolが1本しかない．間隔は要らないのでそのまま使う
        warnings.append(
            f"'{obj_name}' の検出が1本しかないため，分割せず1つとして扱う．"
            '本当に1つかどうかは目で確かめる．')
        edges = np.array([res[a].iloc[0], res[b].iloc[0]], dtype=float)
        return edges, np.array([False]), warnings
    res = res.sort_values(by=a)
    starts = res[a].to_numpy(dtype=float)
    ends = res[b].to_numpy(dtype=float)
    centers = (starts + ends) / 2

    pitch_size = np.median(ends - starts)
    pitch_diff = np.median(np.diff(centers))
    if not np.isfinite(pitch_diff) or pitch_diff <= 0:
        return None, None, warnings
    if not np.isfinite(pitch_size) or pitch_size <= 0:
        pitch_size = pitch_diff
    pitch = min(pitch_size, pitch_diff)

    # 遠く離れた検出を捨てる．
    # ページの隅の誤検出が1本混じるだけで，そこまでの隙間を内挿で埋めてしまう
    # (kinki_007 で本来の行から 2584px = 46行分 離れた1本があり，
    #  6行のはずが52行になった)．
    # 箱の隙間で測るので，中心をまとめる前に箱のまま行う
    keep, dropped = _drop_isolated(starts, ends, pitch_size, max_gap_pitches=5)
    if dropped:
        warnings.append(
            f"'{obj_name}' の検出 {dropped} 本が他から大きく離れていたため捨てた．"
            '誤検出とみなしたが，表が離れて2つある資料なら見直す．')
        centers = centers[keep]
        # **両端の境にも使わない**(2026-09-04)．捨てたのは中心の並びだけで，
        # 最後の境は `ends.max()` から作っていたので，流し込みの中の 1 本の
        # 下端が最後の行の下端になり，最後の行が 1,487 px になっていた
        # (s01115_02_p1_s1．14_p1 は 923 px)
        starts, ends = starts[keep], ends[keep]
    if len(centers) < 2:
        return None, None, warnings

    # 代表値で重複をまとめ，間隔を取り直す．
    # **列は箱の大きさでまとめる**(2026-09-03)．`pitch`(2つの見積もりの
    # 小さい方)で決めると，同じ地点を何本も検出しているときに
    # 中心の間隔の中央値がその重なりの幅まで潰れ(実測 88 px の列に対し 6 px)，
    # `pitch/2` では重複がまとまらない．まとまらないまま間隔を取るので
    # 「1つの列に何本もある」を「列がたくさんある」と数えてしまう
    # (s01115_19_p2_t1 は 17 地点の表が 42 列，05_p2 は 17 地点が 53 列)．
    # **行には同じことをしない**．行の箱は隣どうしが重なって並ぶのが普通で，
    # 幅を広げると `_merge_close` が数珠つなぎに繋いで行を失う
    # (全 66 表で測ると 19 表・569 行が減った)．列は地点ごとに離れて並ぶ
    merge_gap = (pitch_size if axis == 'x' else pitch) / 2
    centers = _merge_close(centers, min_gap=merge_gap)
    if len(centers) < 2:
        return None, None, warnings

    diffs = np.diff(centers)
    pitch_diff = np.median(diffs)
    pitch = min(pitch_size, pitch_diff)

    # 区間ごとに何行あるかを数え，足りない分を内挿する
    counts = np.maximum(1, np.round(diffs / pitch)).astype(int)
    # 2通りの見積もりで行数が変わるなら，どちらが正しいか決められない．
    # 見積もりの値そのものは検出漏れが多いと当然ずれるので，
    # 「結果が変わるか」で判断する
    other = pitch_diff if pitch == pitch_size else pitch_size
    n_other = int(np.maximum(1, np.round(diffs / other)).sum()) + 1
    n_used = int(counts.sum()) + 1
    if n_other != n_used:
        warnings.append(
            f"'{obj_name}' の数を決められない．"
            f'検出の間隔から見ると {n_used if pitch == pitch_diff else n_other}，'
            f'検出の箱の大きさから見ると {n_used if pitch == pitch_size else n_other}．'
            f'間隔の狭い方をとって {n_used} とした．検出結果を目で確かめる．')
    full = [centers[0]]
    interpolated = [False]
    for i, k in enumerate(counts):
        step = (centers[i + 1] - centers[i]) / k
        for j in range(1, k + 1):
            full.append(centers[i] + step * j)
            interpolated.append(j != k)  # 区間の最後だけが実測
    full = np.array(full)

    # 境界は中心どうしの中点．両端だけは検出の外側をそのまま使う
    mid = (full[:-1] + full[1:]) / 2
    edges = np.concatenate([[starts.min()], mid, [ends.max()]])
    return edges, np.array(interpolated), warnings


def ink_profile(image, box, axis: str = "y"):
    """
    領域を二値化して，軸に沿ったインクの量を数える

    行の境界(axis='y')なら，y の位置ごとに横一列のインク画素数を数える．
    文字のある高さは山，行と行の隙間は谷になる．

    Args:
        image: PILのImage
        box: (x1, y1, x2, y2) の集計範囲
        axis: 'y'(行の境界を見る) または 'x'(列の境界を見る)
    Returns:
        (profile, origin, cross_len)
        profile  : 位置ごとのインク画素数
        origin   : profile[0] が画像のどの座標にあたるか
        cross_len: 足し合わせた方向の長さ(谷の判定に使う)
        領域が取れないときは (None, 0, 0)
    """
    w, h = image.size
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, 0, 0
    gray = np.asarray(image.crop((x1, y1, x2, y2)).convert('L'), dtype=np.uint8)
    # 大津の方法で二値化(古い資料は明るさが一定でないため固定閾値にしない)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if axis == 'y':
        return bw.sum(axis=1) / 255.0, y1, x2 - x1
    return bw.sum(axis=0) / 255.0, x1, y2 - y1


def _peak_run(profile: np.ndarray, pos: int, tol: float):
    """
    posを含む「インクがtolを超えて続く区間」を (始まり, 終わり) で返す
    """
    start = pos
    while start > 0 and profile[start - 1] > tol:
        start -= 1
    end = pos
    while end < len(profile) - 1 and profile[end + 1] > tol:
        end += 1
    return start, end


def snap_edges(edges: np.ndarray, profile: np.ndarray, origin: int,
               cross_len: int, max_shift: int, ink_ratio: float = 0.02,
               max_rule_width: int = 6):
    """
    文字に重なっている境界を，近くの谷へずらす

    等分でも検出どおりでも，境界が文字の上に乗ると，
    1文字が2つのセルに割れてOCRが失敗する．
    インクの少ない位置へ寄せることでこれを避ける．
    両端は表の外枠なので動かさない．

    **罫線は例外**．組成表は実線や点線で一部を囲むことがあり，
    罫線は表を横切るのでプロファイルでは山になる．
    しかし罫線のある位置は「文字」ではなく，むしろ正しい境界そのものなので，
    谷へ逃がしてはいけない．山の幅が細いもの(max_rule_width以下)を罫線とみなし，
    その中央へ合わせる．文字の帯は太い山になるので区別できる．

    Args:
        edges: 境界の並び(長さ n+1)
        profile, origin, cross_len: ink_profile()の戻り値
        max_shift: ずらせる最大の画素数
        ink_ratio: これ以下のインク量を「谷」とみなす割合(cross_lenに対する)
        max_rule_width: この幅以下の山は罫線とみなす(画素)
    Returns:
        (new_edges, snapped, unresolved, on_rule)
        snapped   : 境界ごとのbool．動かしたものがTrue
        unresolved: 境界ごとのbool．ずらしても谷に入らなかったものがTrue
        on_rule   : 境界ごとのbool．罫線に合わせたものがTrue
    """
    new_edges = np.array(edges, dtype=float)
    snapped = np.zeros(len(edges), dtype=bool)
    unresolved = np.zeros(len(edges), dtype=bool)
    on_rule = np.zeros(len(edges), dtype=bool)
    tol = max(1.0, ink_ratio * cross_len)
    # 谷はプロファイル全体から取る．窓の中だけで見ると，
    # 窓に切られた谷の「端」を中央と勘違いする
    runs = _valley_runs(profile, tol)
    run_centers = np.array([(s + e) / 2 for s, e in runs]) if runs else np.array([])
    for i in range(1, len(edges) - 1):
        pos = int(round(edges[i])) - origin
        if pos < 0 or pos >= len(profile):
            continue
        if profile[pos] <= tol:
            continue  # すでに谷にある
        run_start, run_end = _peak_run(profile, pos, tol)
        if run_end - run_start + 1 <= max_rule_width:
            # 罫線に乗っている．逃がさず，罫線の中央に合わせる
            pick = (run_start + run_end) / 2
            on_rule[i] = True
            if pick != pos:
                new_edges[i] = pick + origin
                snapped[i] = True
            continue
        lo = max(0, pos - max_shift)
        hi = min(len(profile), pos + max_shift + 1)
        window = profile[lo:hi]
        if not len(window):
            continue
        # 窓に掛かる谷を探す
        reachable = [j for j, (s, e) in enumerate(runs) if s <= hi - 1 and e >= lo]
        if reachable:
            # 谷の端では文字に近すぎるので，いちばん近い谷の中央へ寄せる
            centers = run_centers[reachable]
            target = centers[np.argmin(np.abs(centers - pos))]
            pick = float(np.clip(target, lo, hi - 1))  # ずらせる幅は超えない
        else:
            # 谷が無い．いちばんインクの少ない位置まで寄せて，直せなかったと伝える
            pick = float(np.argmin(window) + lo)
            unresolved[i] = True
        if pick != pos:
            new_edges[i] = pick + origin
            snapped[i] = True
    return new_edges, snapped, unresolved, on_rule


def _valley_runs(window: np.ndarray, tol: float):
    """
    インクがtol以下で続く区間(谷)を (始まり, 終わり) の並びで返す
    """
    is_valley = window <= tol
    if not is_valley.any():
        return []
    edges = np.diff(np.concatenate([[0], is_valley.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts, ends))


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


def axis_notes(interpolated=None, snapped=None, unresolved=None, n: int = 0):
    """
    1つの軸について，セルごとの気になる点を並べる

    境界のフラグ(長さ n+1)は，その境界に接する両側のセルに配る．

    Args:
        interpolated: セルごとのbool(長さ n)．検出漏れを内挿したもの
        snapped: 境界ごとのbool(長さ n+1)．文字を避けてずらしたもの
        unresolved: 境界ごとのbool(長さ n+1)．ずらしても文字に重なるもの
        n: セルの数
    Returns:
        長さ n のリスト．各要素は文字列のリスト
    """
    notes = [[] for _ in range(n)]
    for i in range(n):
        if interpolated is not None and interpolated[i]:
            notes[i].append('interpolated')
        if snapped is not None and (snapped[i] or snapped[i + 1]):
            notes[i].append('snapped')
        if unresolved is not None and (unresolved[i] or unresolved[i + 1]):
            notes[i].append('on_text')
    return notes


def coord_item(x_edges: np.ndarray, y_edges: np.ndarray, obj_name: str = "",
               x_notes=None, y_notes=None) -> pd.DataFrame:
    """
    境界の並びからセルの座標を作る

    分割しない軸は，端から端までの2要素を渡す．

    Args:
        x_edges: 長さ n_cols+1 の境界
        y_edges: 長さ n_rows+1 の境界
        obj_name: 付けるクラス名
        x_notes, y_notes: axis_notes()の戻り値．セルのnote列に合わせて入れる
    Returns:
        x1, x2, y1, y2, obj_name, note を持つDataFrame(列ごとに行が並ぶ)
        note は ';' 区切り．気になる点が無ければ空文字
    """
    x1, x2 = x_edges[:-1], x_edges[1:]
    y1, y2 = y_edges[:-1], y_edges[1:]
    n_cols, n_rows = len(x1), len(y1)
    if x_notes is None:
        x_notes = [[] for _ in range(n_cols)]
    if y_notes is None:
        y_notes = [[] for _ in range(n_rows)]
    # np.repeat / np.tile と同じ並び(列ごとに行が並ぶ)にする
    notes = [';'.join(dict.fromkeys(x_notes[c] + y_notes[r]))
             for c in range(n_cols) for r in range(n_rows)]
    return pd.DataFrame({
        'x1': np.repeat(x1, n_rows),
        'x2': np.repeat(x2, n_rows),
        'y1': np.tile(y1, n_cols),
        'y2': np.tile(y2, n_cols),
        'obj_name': obj_name,
        'note': notes,
    })


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
    # 1ページに段が複数あるときは，段ごとに格子を作る(混ぜると壊れる)
    blocks = split_blocks(df)
    if len(blocks) > 1:
        warnings.append(
            f'このページには段が {len(blocks)} つある(種名のリストを折り返した組み方)．'
            '段ごとに格子を作り，行は左の段から続けて番号を振る．'
            '地点は段をまたいで同じものとして扱う．')
    items = []
    for i, block in enumerate(blocks):
        res, block_warnings = _locate_block(
            block, source_image, img, max_shift_ratio, threth_col, threth_row)
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


def _guess_layer_column(img, name_ranges, x_edges, y_edges):
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

    間違って作っても，読んだ中身が階層として通らなければ関門2に挙がるので，
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
    if ink.text_ratio(dark, gy1, gy2, gx1, gx2) / base < LAYER_GAP_MIN_INK:
        return None, False
    return pd.DataFrame({'x1': [gx1], 'x2': [gx2]}), True


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
                '関門1で表の端を目で確かめる．')
    return out


def _edge_ink(edges, dark, y1, y2):
    """境の線上の黒画素(帯の中央値との比)．小さいほど字を割っていない"""
    x1, x2 = int(edges[0]), int(edges[-1])
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    prof = dark[int(y1):int(y2), x1:x2].sum(axis=0).astype(float)
    if not (prof > 0).any():
        return None
    med = float(np.median(prof[prof > 0]))
    vals = [prof[int(e) - x1] for e in edges[1:-1] if x1 < e < x2]
    return (float(np.mean(vals)) / med) if vals and med else None


HEADER_SHIFT_MAX = 0.4      # 表頭をずらす範囲(列の間隔に対する比)
HEADER_SHIFT_GAIN = 0.5     # 線上の黒画素がこの倍以下になるときだけずらす
HEADER_SHIFT_MIN = 5        # これ未満のずれは動かさない(誤差の範囲)
HEADER_SHIFT_TOL = 5        # 行ごとのずれがこの差以内なら，同じずれとみなす


def _shift_edges_for_header(x_edges, img, bands, max_ratio=HEADER_SHIFT_MAX,
                            gain=HEADER_SHIFT_GAIN, min_shift=HEADER_SHIFT_MIN,
                            tol=HEADER_SHIFT_TOL):
    """表頭の列の境を，**地点の値の行の合意**で左右にずらす

    帯ぜんたいの黒画素で決めてはいけない(2026-09-05 に測って分かった)．
    表頭には群落名や凡例の**地の文**が混ざり，そちらの字数が多いので
    ずらし量がそれに引っぱられる(s01115_22_p2 は帯ぜんたいで +41 px だが，
    通し番号の行で測ると 0 px = ずらす必要なし．読んだ通し番号の一致は
    0.56 → 0.04 と落ちていた)．

    **地点ごとの値が並ぶ行**(塊が列の数の半分以上)だけを対象にし，
    行ごとの最良のずれを求めて，`tol` px 以内で最も多くの行が賛成する値を採る．

    Args:
        bands: 表頭の行の (y1, y2) のリスト
    Returns:
        (ずらした境, ずらした量 px．ずらさなければ (元の境, 0))
    """
    import count_plots

    if img is None or x_edges is None or len(x_edges) < 3 or not bands:
        return x_edges, 0
    dark = ink.binarize(img)
    n_col = len(x_edges) - 1
    pitch = float(np.median(np.diff(x_edges)))
    votes = []
    for y1, y2 in bands:
        if y2 - y1 < 5:
            continue
        bl = count_plots.blobs(dark, x_edges[0], x_edges[-1], y1, y2,
                               max(6, int(pitch * 0.25)))
        if len(bl) < n_col * 0.5:
            continue                      # 地の文の行(値が並んでいない)
        _, s = _shift_edges_to_band(x_edges, img, y1, y2, max_ratio,
                                    gain=1.0, min_shift=1)
        votes.append(s)
    if not votes:
        return x_edges, 0
    # いちばん多くの行が賛成するずれ(±tol px を同じ票とみなす)
    best, best_n = 0, 0
    for v in votes:
        n = sum(1 for w in votes if abs(w - v) <= tol)
        if n > best_n or (n == best_n and abs(v) < abs(best)):
            best, best_n = v, n
    agree = [w for w in votes if abs(w - best) <= tol]
    shift = int(round(float(np.median(agree))))
    if abs(shift) < min_shift or best_n < max(2, len(votes) * 0.5):
        return x_edges, 0
    lo, hi = float(bands[0][0]), float(bands[-1][1])
    e = np.asarray(x_edges, dtype=float).copy()
    e[1:-1] += shift
    return np.maximum.accumulate(e), shift


def _shift_edges_to_band(x_edges, img, y1, y2, max_ratio=HEADER_SHIFT_MAX,
                         gain=HEADER_SHIFT_GAIN, min_shift=HEADER_SHIFT_MIN):
    """列の境を，`y1`-`y2` の帯の黒画素に合わせて左右にずらす

    列の数は変えない．ずらして線上の黒画素が `gain` 倍以下になるときだけ採る．

    Returns:
        (ずらした境, ずらした量 px．ずらさなければ (元の境, 0))
    """
    if img is None or x_edges is None or len(x_edges) < 3 or y2 - y1 < 5:
        return x_edges, 0
    dark = ink.binarize(img)
    base = _edge_ink(x_edges, dark, y1, y2)
    if base is None:
        return x_edges, 0
    pitch = float(np.median(np.diff(x_edges)))
    reach = max(1, int(pitch * max_ratio))

    def shifted(s):
        # **外側の境は動かさない**．端ごとずらすと，端の値が枠の外に出る
        # (22_p2 は +41 px で通し番号の「1」が左端の外に落ちた．2026-09-04)
        e = np.asarray(x_edges, dtype=float).copy()
        e[1:-1] += s
        return np.maximum.accumulate(e)

    best, best_v = 0, base
    for s in range(-reach, reach + 1):
        v = _edge_ink(shifted(s), dark, y1, y2)
        if v is not None and v < best_v:
            best, best_v = s, v
    # **しきい値は 0.5 倍・5 px 以上**(2026-09-05 に 3 案を測って決めた)．
    # 0.8 倍だと 6 表 420 セルのうち動きが 1 表増えるだけで，読みは同じだった．
    # 発動する表が少ないほど，意図しない副作用が減る
    if abs(best) < min_shift or best_v > base * gain:
        return x_edges, 0
    return shifted(best), best


def _locate_block(df: pd.DataFrame, source_image: str, img, max_shift_ratio: float,
                  threth_col: float = 0.8, threth_row: float = 0.8):
    """
    1つの段について格子を作る(locate_itemsから段ごとに呼ぶ)

    重複の除去は**段に分けたあと**に行う．
    行の重複判定はyの重なりだけを見てxを見ないため，
    段をまたいで一括でかけると，右の段の行が左の段の行の重複として消える．

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
        layer_x_range, layer_guessed = _guess_layer_column(
            img, name_ranges, x_edges, y_edges)
        if layer_guessed:
            warnings.append(
                f"'{class_layer}' は検出されなかったが，種名の列と組成部の隙間に"
                '字があるため，そこを階層の列として補った．'
                '読んだ中身が階層として通らなければ関門2に挙がる．')
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
                    '階層の列があるはずなら，関門1で隙間を目で確かめる．')
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
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
    n_wide = sum(1 for g in gaps if g > median_h * 0.8)
    if n_wide:
        warnings.append(
            f'表頭の項目行のあいだに，1行ぶん以上あいた箇所が {n_wide} つある．'
            '拾えていない項目があるかもしれない．')
    hc = locate_x_range(df, source_image, obj_name=class_hdr_col)
    # **項目名の行から帯を作り直す**(できるとき)．
    # `plot_row` の箱は項目行と1対1に対応しておらず，2項目を1つの箱に
    # まとめたり，値の2行目だけを1つの箱にしたりする．
    # 項目名は1項目1行なので，そちらの字の行を数える方が確かめやすい．
    if img is not None and hc is not None:
        name_x1 = float(hc['x1'].iloc[0])
        name_x2 = float(x_edges[0])      # 項目名(独語・和名)から値の手前まで
        new_edges = _header_bands_from_names(
            img, (name_x1, h_edges[0], name_x2, h_edges[-1]))
        if new_edges is not None and len(new_edges) >= 3:
            if len(new_edges) != len(h_edges):
                warnings.append(
                    f'表頭の項目行は，検出({len(h_edges) - 1} 行)と'
                    f'項目名の字の行({len(new_edges) - 1} 行)で数が違う．'
                    '項目名の側を採った．関門1で行の対応を目で確かめる．')
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
            '表頭の列の境だけずらした．関門1で表頭の値の切れ目を目で確かめる')
    out = [coord_item(hx_edges, h_edges, obj_name='header_value',
                      y_notes=h_notes)]
    if hc is None:
        warnings.append(
            f"'{class_hdr_col}' が検出されず，表頭の項目名を出力できない．"
            '値だけでは，どの行が何の項目か決められない．')
        return out
    left, right = float(hc['x1'].iloc[0]), float(hc['x2'].iloc[0])
    out.append(coord_item(np.array([left, right]), h_edges,
                          obj_name='header_item', y_notes=h_notes))
    # 項目名は2言語で入っていることが多い(header_col はドイツ語で，
    # その右に和名の列がある)．和名の方がOCRも突き合わせも確実なので，
    # header_col の右端から値の左端までを別に切り出す
    gap = float(x_edges[0]) - right
    if gap > (right - left) * 0.2:
        out.append(coord_item(np.array([right, float(x_edges[0])]), h_edges,
                              obj_name='header_item_ja', y_notes=h_notes))
    return out


def _snap_axis(edges, img, box, axis: str, label: str, max_shift_ratio: float):
    """
    1つの軸の境界をまとめてずらす(locate_itemsから使う)

    Returns:
        (new_edges, warnings)
    """
    warnings = []
    none_flags = np.zeros(len(edges), dtype=bool)
    profile, origin, cross_len = ink_profile(img, box, axis=axis)
    if profile is None:
        warnings.append(f'{label}の境界を調べる領域が画像の外にあり，調整を行わなかった．')
        return edges, warnings, none_flags, none_flags
    pitch = np.median(np.diff(edges))
    max_shift = max(1, int(round(pitch * max_shift_ratio)))
    # 文字の帯より細い山を罫線とみなす
    max_rule_width = max(3, int(round(pitch * 0.15)))
    new_edges, snapped, unresolved, on_rule = snap_edges(
        edges, profile, origin, cross_len, max_shift,
        max_rule_width=max_rule_width)
    n_snapped = int(snapped.sum())
    n_unresolved = int(unresolved.sum())
    n_on_rule = int(on_rule.sum())
    if n_snapped - n_on_rule > 0:
        warnings.append(
            f'{label}の境界 {n_snapped - n_on_rule} 本が文字に重なっていたため，'
            f'最大 {max_shift} px の範囲で隙間へずらした．')
    if n_on_rule:
        warnings.append(
            f'{label}の境界 {n_on_rule} 本は罫線に乗っていたため，'
            'その中央に合わせた(罫線は正しい区切りなので，動かさない)．')
    if n_unresolved:
        warnings.append(
            f'{label}の境界 {n_unresolved} 本は，ずらしても文字に重なったまま．'
            'そのセルは文字が割れて読めていないおそれがある．'
            f'{label}数が実際と違う可能性を疑う．')
    # ずらした先が隣の隙間だと，セルの大きさが偏る．格子そのものを疑う手がかりになる
    sizes = np.diff(new_edges)
    n_odd = int((np.abs(sizes - pitch) > pitch * 0.25).sum())
    if n_odd:
        warnings.append(
            f'{label}のうち {n_odd} 件は，大きさが代表値({pitch:.0f} px)から'
            f'25%以上ずれている．{label}の区切りが1つずれているおそれがある．')
    # 罫線に合わせたものは問題ではないので，セルの印には含めない
    return new_edges, warnings, snapped & ~on_rule, unresolved


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
