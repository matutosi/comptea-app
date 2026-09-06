"""地点の多い横長の表を，短冊に分けて検出する

`s01115` の付表には**地点が 37 もある表**がある．学習した表はどれも 10 地点以下で，
行の箱が横に長すぎて `row` が1本も検出できない．**縮尺の問題ではない**
(2026-09-02 に測った．長辺に合わせて `imgsz` を 2560・4288 まで上げ，
検出数の上限 `max_det` も外したが，`row` は 0 件のまま)．
行の**横幅**が学習時から外れているのが原因で，

    種名の列 + 8 地点 (2200 px)   → row 102 件，格子 95 行 x 8 列
    種名の列 + 17 地点 (3000 px)  → row  13 件，格子  8 行 x 17 列

と，幅だけで結果が変わる．

そこで，**種名の列を頭に付けたまま，組成部を地点のかたまりに分けて**検出する．
短冊は学習した表と同じ姿(種名 + 10 地点ほど)になるので，行がふつうに取れる．
切れ目は**地点と地点のあいだの隙間**に置くので，1地点が2つに割れることはない．

検出した箱は**元の画像の座標に戻して1つにまとめる**ので，
`locate.py` から先は横長の表であることを知らなくてよい．
"""

import numpy as np
import pandas as pd
from PIL import Image

import ink
from col_edges import column_edges_from_gaps, plot_gaps
from table_split import drop_stray_plot_rows

Image.MAX_IMAGE_PIXELS = None


# 種名の側の列(短冊の頭に付ける)
# **`layer`・`header_col` は入っていない**(2026-09-06 に分割して分かった)．
# 元は 4 つ組で書いてあったが，同じ名前の定数が同じファイルの後ろで 2 つ組に
# 上書きされており，**走っていた値は 2 つ組**だった(`map_back()` の 1 か所が
# これを見ている)．分けるにあたっては振る舞いを変えないため，走っていた値を
# そのまま採る．4 つ組が正しいかは，物差しを回して別に決める
NAME_CLASSES = ('sname', 'species_col')


# 表の幅いっぱいに広がるもの．短冊では途中で切れるので，戻すときに広げ直す
FULL_WIDTH = ('row', 'plot_row', 'header', 'table', 'once_species')


TARGET_W = 2237         # 短冊の幅の目安(学習画像の幅の中央値)


MIN_PLOTS = 4           # これ未満になる分け方はしない(短すぎると行が取れない)


OVERLAP_PITCH = 1       # 隣り合う短冊を，地点いくつぶん重ねるか


WIDE_COL_MIN = 20       # 「横長」と判定するのに要る `col` の検出数


WIDE_ROW_MAX = 5        # 「横長」と判定する `row` の検出数の上限


COL_MAX_PITCH = 2.5     # 地点の間隔の何倍までを `col` として認めるか


# **隙間の求め方と組み合わせて初めて効く**．2026-09-02 に 58 表で測った
# (物差しは，表頭の値の行から数えた地点数と，格子の列数のずれ)．
#
#   隙間の求め方        格子   表  ±1一致   ずれの中央値  ずれの合計
#   黒画素の合計(旧)    なし   52  20 (38%)      2           506
#   黒でない割合(新)    なし   58  22 (38%)      3           693
#   黒でない割合(新)    あり   58  28 (48%)      2           333  ← これ
#
# 共通の 52 表では 良化 15 / 悪化 6 / 変化なし 31，ずれの合計 506 → 289．
# **どちらか片方では効かない**．隙間が粗いまま格子を当てると間隔を
# 読み違え，隙間が良くても格子を当てなければ列の境に反映されない
USE_LATTICE = True


# **近すぎる谷を間引く案は，測って採らなかった**(2026-09-02)．
# 字の内側で黒画素が薄くなり 1 行が 2-3 に割れることがあるので，
# 間隔の中央値の 0.6 倍より近い谷を落としてみた．
#   格子の行の合計  7,952 -> 7,876   セルの合計 271,176 -> 268,977
# s01115_10_p1 は 114 -> 193 行と大きく良くなるが，12_p1(162 -> 135)と
# 11_p1(93 -> 84)の悪化で相殺され，全体では微減だった
DEDUPE_IOU = 0.5        # 短冊どうしで同じものとみなす縦の重なり


MERGE_GAP_RATIO = 0.02  # 縦に切れた箱を繋ぐ隙間．画像の高さに対する比


# **表の高さいっぱいに伸びる列**だけを和にしてまとめる．
# `header` や `table` を入れてはいけない．表頭は短冊ごとに何本にも分かれて
# 出るので，隙間を許して繋ぐと**数珠つなぎで表の全高に広がり**，
# 表頭の項目行が 174 行になった(2026-09-02 に p3 で起きた)
MERGE_CLASSES = ('sname', 'species_col', 'layer', 'header_col')


def needs_strips(df_det):
    """短冊に分けないと読めない表か

    `col` はたくさん取れているのに `row` がほとんど無い，という形で起きる．

    **`row` が 0 件であることを条件にしてはいけない**．重みを変えると
    同じ表で 1 件だけ取れることがあり，そのとき条件が外れて，
    短冊に分けないまま「1 行 x 35 列」の格子ができてしまう
    (2026-09-02 に学習し直した重みで起きた)．
    `col` の検出が多い表で `row` が数本しかないのは，どのみち壊れている．
    地点の少ない小さな表は `col` の検出も少ないので，ここには来ない．
    """
    if df_det.empty:
        return False
    n = df_det['obj_name'].value_counts()
    return (int(n.get('row', 0)) < WIDE_ROW_MAX
            and int(n.get('col', 0)) >= WIDE_COL_MIN)


def layout(df_det, width):
    """種名の側と組成部の境目を，検出から決める

    Returns:
        (name_x2, body_x1, body_x2)．決められなければ None
    """
    col = df_det[df_det['obj_name'] == 'col']
    if col.empty:
        return None
    # **種名の列の上に出た `col` は組成部に数えない**(2026-09-03)．
    # 部分画像に切り出した Tab.8(s01115_02)では，種名の列に `col` が
    # 1本出て組成部の左端が x 1236 → 50 になり，種名の列まで地点として
    # 格子を当てて 29 地点の表が 42 列になった
    names = df_det[df_det['obj_name'].isin(('sname', 'species_col', 'header_col'))]
    if len(names):
        name_right = float(names['x2'].max())
        inside = ((col['x1'] + col['x2']) / 2 < name_right)
        if inside.any() and not inside.all():
            col = col[~inside]
    body_x1, body_x2 = int(col['x1'].min()), int(col['x2'].max())
    # **組成部の左は，まるごと種名の側とする**．
    # 検出された `sname` などの右端で切ると，取れなかった列が短冊から
    # 外れてしまう(p3 では和名の列が conf 0.24 で落ち，そのぶん抜けた)．
    # 階層の列も種名と組成部の隙間にあるので，この取り方なら一緒に入る
    name_x2 = body_x1
    if body_x2 <= body_x1 or name_x2 <= 0 or body_x2 > width:
        return None
    return name_x2, body_x1, body_x2


def plan_strips(gaps, name_w, body_x1, body_x2, target_w=TARGET_W,
                min_plots=MIN_PLOTS, overlap=OVERLAP_PITCH):
    """組成部を切る位置を決める．返すのは (x1, x2) の並び

    短冊の幅が `target_w` を超えない範囲で，**できるだけ多くの地点**を入れる．
    切れ目は隙間の中央にしか置かない．

    隣り合う短冊は**地点 `overlap` 個ぶん重ねる**．短冊の端に来た地点は
    検出が弱く，そのままだと継ぎ目のたびに列を落とす
    (p3 では 37 地点のうち 32 列しか取れなかった．2026-09-02)．
    重なったぶんは `locate.py` の列の重複除去が落とす．
    """
    room = max(target_w - name_w, 0)
    inner = [g for g in gaps if body_x1 < g < body_x2]
    if room <= 0 or not inner:
        return [(body_x1, body_x2)]
    pitch = np.median(np.diff([body_x1] + inner + [body_x2])) if inner else room
    if room < pitch * min_plots:
        # 種名の列が広すぎて地点が数えるほどしか入らない．幅を譲る
        room = pitch * min_plots

    cuts, start = [], body_x1
    while start < body_x2:
        limit = start + room
        cand = [g for g in inner if start < g <= limit]
        if not cand or body_x2 <= limit:
            cuts.append((start, body_x2))
            break
        end = max(cand)
        cuts.append((start, end))
        start = end
    if len(cuts) < 2:
        return cuts
    pad = pitch * overlap
    return [(int(max(body_x1, a - pad if i else a)),
             int(min(body_x2, b + pad if i < len(cuts) - 1 else b)))
            for i, (a, b) in enumerate(cuts)]


def cols_from_edges(edges, y1, y2, like):
    """列の境から `col` の検出を作り直す(検出の代わりに使う)"""
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = like.copy()
        r.update(dict(obj_name='col', confidence=1.0,
                      x1=int(round(a)), x2=int(round(b)),
                      y1=int(y1), y2=int(y2)))
        rows.append(r)
    return pd.DataFrame(rows)


def make_strip(im, table_x1, name_x2, x1, x2, y1=0, y2=None):
    """種名の列 + 組成部の一部 を横につないだ画像を作る

    `table_x1` はこの表の左端．**1枚に表が左右に並んでいる**ことがあるので，
    いつも x=0 から切ると隣の表を巻き込む(s01115_02 の Tab.8 と Tab.9)．

    `y1`・`y2` はこの表の上端と下端．**表が縦に重なっている**ことがあり，
    全高で切ると下(上)の表まで巻き込んで，先に分けた表がまた1つに戻る
    (s01115_19_p2 の右側．下の表の行が 1回出現種の帯で切れ，丸ごと
    捨てられていた．2026-09-03)．渡されなければ全高
    """
    if y2 is None:
        y2 = im.height
    name_w = name_x2 - table_x1
    left = im.crop((table_x1, y1, name_x2, y2))
    right = im.crop((x1, y1, x2, y2))
    strip = Image.new(im.mode, (name_w + (x2 - x1), y2 - y1), 255)
    strip.paste(left, (0, 0))
    strip.paste(right, (name_w, 0))
    return strip


def _back(v, table_x1, name_w, x1):
    """短冊の x を元の画像の x に戻す"""
    return table_x1 + v if v < name_w else x1 + (v - name_w)


def map_back(df, table_x1, name_x2, x1, body_x1, body_x2, pitch=None,
             y1=0):
    """短冊の検出を元の画像の座標に戻す(`y1` は短冊の上端．縦にも戻す)

    表の幅いっぱいに広がるもの(`row` など)は，短冊では途中で切れているので
    **元の幅に広げ直す**．そうしないと段の幅を測り違える．

    種名の側に出た `col` は捨てる．短冊では和名の列のすぐ隣に値が並ぶので，
    和名の列そのものが地点の列に見えることがある(元の表では地点ではない)．
    """
    if df.empty:
        return df
    d = df.copy()
    d['y1'] = d['y1'] + y1
    d['y2'] = d['y2'] + y1
    # 組成部を丸ごと1列とみなした `col` は，短冊の継ぎ目をまたいで
    # 種名の側から始まることがある．そのまま戻すと表の幅いっぱいの
    # 巨大な列になり，格子が数本に潰れる(2026-09-02 に p3 で起きた)．
    # 左端を組成部の始まりに揃えてから戻す
    name_w = name_x2 - table_x1
    is_col = d['obj_name'] == 'col'
    d.loc[is_col, 'x1'] = d.loc[is_col, 'x1'].clip(lower=name_w)
    d['x1'] = d['x1'].map(lambda v: _back(v, table_x1, name_w, x1))
    d['x2'] = d['x2'].map(lambda v: _back(v, table_x1, name_w, x1))
    center = (d['x1'] + d['x2']) / 2
    # 短冊では和名の列のすぐ隣に値が並ぶので，和名の列が地点の列に見えたり，
    # 値の並びが種名の列に見えたりする．**どちら側にあるか**で捨てる
    d = d[~((d['obj_name'] == 'col') & (center < body_x1))]
    center = (d['x1'] + d['x2']) / 2
    d = d[~(d['obj_name'].isin(NAME_CLASSES) & (center >= body_x1))]
    if pitch:
        # 地点の間隔は隙間から測ってある．それより大幅に広い `col` は，
        # 組成部をまとめて1列と見た誤検出なので落とす
        wide = ((d['obj_name'] == 'col')
                & ((d['x2'] - d['x1']) > pitch * COL_MAX_PITCH))
        d = d[~wide]
    full = d['obj_name'].isin(FULL_WIDTH)
    d.loc[full, 'x1'] = table_x1
    d.loc[full, 'x2'] = body_x2
    return d


def _y_iou(a, b):
    """縦の重なり具合(1次元の IoU)"""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / ((a[1] - a[0]) + (b[1] - b[0]) - (hi - lo))


def _merge_pieces(g, gap):
    """縦に重なるか接している箱を，1つの箱にまとめる(和をとる)"""
    g = g.sort_values('y1')
    out = []
    for _, r in g.iterrows():
        if out and r['y1'] <= out[-1]['y2'] + gap:
            cur = out[-1]
            cur['y2'] = max(cur['y2'], r['y2'])
            cur['x1'] = min(cur['x1'], r['x1'])
            cur['x2'] = max(cur['x2'], r['x2'])
            cur['confidence'] = max(cur['confidence'], r['confidence'])
        else:
            out.append(r.copy())
    return pd.DataFrame(out)


def dedupe(df, height, iou=DEDUPE_IOU, gap_ratio=MERGE_GAP_RATIO):
    """短冊の数だけ重なった検出を，1枚ぶんの姿に戻す

    まとめ方はクラスで違う．

    - `col` はそのまま．短冊ごとに**別の地点**なので重なっていない
    - 種名の列・表頭など**縦に長い1つのもの**は，重なるか接している箱を
      和にしてまとめる．短冊ごとに上下に切れて写るので，そのまま繋ぐと
      「縦に2つある」ことになり，`split_tables()` が
      **1つの表を2つと読み違える**(2026-09-02 に p3 で起きた)
    - 行は縦の重なりで重複だけを落とす(別々の行はまとめない)
    """
    if df.empty:
        return df
    gap = max(4, int(height * gap_ratio))
    keep = []
    for name, g in df.groupby('obj_name', sort=False):
        if name == 'col':
            keep.append(g)
        elif name in MERGE_CLASSES:
            keep.append(_merge_pieces(g, gap))
        else:
            g = g.sort_values('confidence', ascending=False)
            chosen = []
            for _, r in g.iterrows():
                span = (r['y1'], r['y2'])
                if any(_y_iou(span, c) >= iou for c in chosen):
                    continue
                chosen.append(span)
                keep.append(g.loc[[r.name]])
    return pd.concat(keep, ignore_index=True)


def detect_wide(image, df_det, detect_fn, target_w=TARGET_W,
                lattice=USE_LATTICE, y_range=None):
    """短冊ごとに検出し，元の座標に戻した検出結果を1つにまとめて返す

    Args:
        image   : 表の画像の場所
        df_det  : 表ぜんたいを1度検出した結果(組成部の位置を決めるのに使う)
        detect_fn: (PIL 画像, 元の名前) -> 検出の DataFrame
        y_range : この表の (上端, 下端)．表が縦に重なっているときに渡す．
                  無ければ全高(1つの表しかない紙面はこれまでどおり)
    Returns:
        (検出の DataFrame, 警告のリスト)．分けられなければ (None, 警告)
    """
    warnings = []
    im = Image.open(image)
    lay = layout(df_det, im.width)
    if lay is None:
        warnings.append('横長の表を短冊に分けようとしたが，'
                        '種名の列と組成部の境目を決められなかった')
        return None, warnings
    name_x2, body_x1, body_x2 = lay
    table_x1 = max(0, int(df_det['x1'].min()))

    dark = ink.binarize(im)
    ys = df_det[df_det['obj_name'] == 'col']
    box = (body_x1, int(ys['y1'].min()), body_x2, int(ys['y2'].max()))
    gaps = plot_gaps(dark, box)
    inner = [g for g in gaps if body_x1 < g < body_x2]
    pitch = (float(np.median(np.diff([body_x1] + inner + [body_x2])))
             if inner else 0.0)
    chunks = plan_strips(gaps, name_x2 - table_x1, body_x1, body_x2,
                         target_w)
    warnings.append(
        f'**地点が多すぎて行が取れないので，組成部を {len(chunks)} つに分けて'
        f'検出した**(種名の列 x0-{name_x2} を頭に付け，'
        f'地点のあいだの隙間 {len(gaps)} 本から切れ目を選んだ)．'
        '短冊の境目で列が重複していないか，段階1で目で確かめる')

    y1, y2 = (0, im.height) if y_range is None else (
        max(0, int(y_range[0])), min(im.height, int(y_range[1])))
    parts = []
    for i, (x1, x2) in enumerate(chunks, 1):
        strip = make_strip(im, table_x1, name_x2, x1, x2, y1, y2)
        d = detect_fn(strip, f'{image}#strip{i}')
        if d.empty:
            warnings.append(f'短冊 {i}/{len(chunks)} (x {x1}-{x2}) は検出が0件だった')
            continue
        parts.append(map_back(d, table_x1, name_x2, x1, body_x1, body_x2,
                              pitch, y1))
    if not parts:
        warnings.append('短冊に分けても検出が0件だった')
        return None, warnings
    out = dedupe(pd.concat(parts, ignore_index=True), im.height)
    out, stray = drop_stray_plot_rows(out, ref=df_det)
    warnings += stray

    # **列の境は印字の隙間から決める**．短冊に分けた表では，`col` の検出が
    # 継ぎ目や値の粗密で崩れ，列が足りなくなったり多すぎたりする
    # (2026-09-02 に測ったとき，落ちた 22 表のうち 11 表がこれだった)．
    # 地点の列は等間隔に組まれているので，隙間に格子を当てるほうが確か
    col = out[out['obj_name'] == 'col']
    edges = (column_edges_from_gaps(gaps, body_x1, body_x2, dark=dark,
                                    y1=box[1], y2=box[3])
             if lattice else None)
    if edges is not None and not col.empty:
        like = col.iloc[0].to_dict()
        made = cols_from_edges(edges, col['y1'].min(), col['y2'].max(), like)
        out = pd.concat([out[out['obj_name'] != 'col'], made],
                        ignore_index=True)
        warnings.append(
            f'列の境を**印字の地点の隙間**から決め直した'
            f'({len(col)} 件の検出 → {len(made)} 列)．'
            '地点の列は等間隔に組まれているので，検出より隙間の方が確か．'
            '段階1で列の中身を目で確かめる')
    elif lattice:
        warnings.append(
            '列の境を印字の隙間から決めようとしたが，**等間隔に乗らなかった**．'
            '検出から決めた境をそのまま使う．段階1で列を目で確かめる')

    out['source_image'] = str(image).replace('\\', '/')
    return out, warnings
