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

Image.MAX_IMAGE_PIXELS = None

# 種名の側の列(短冊の頭に付ける)．`layer` は隙間にあって検出されないことが多い
NAME_CLASSES = ('sname', 'species_col', 'layer', 'header_col')
# 表の幅いっぱいに広がるもの．短冊では途中で切れるので，戻すときに広げ直す
FULL_WIDTH = ('row', 'plot_row', 'header', 'table', 'once_species')

TARGET_W = 2237         # 短冊の幅の目安(学習画像の幅の中央値)
MIN_PLOTS = 4           # これ未満になる分け方はしない(短すぎると行が取れない)
OVERLAP_PITCH = 1       # 隣り合う短冊を，地点いくつぶん重ねるか
GAP_MIN_PX = 20         # 隙間とみなす最小の幅(300 dpi のスキャンで測った)
BLANK_RATIO = 0.98      # 「空白」とみなす，黒でない画素の割合
WIDE_COL_MIN = 20       # 「横長」と判定するのに要る `col` の検出数
WIDE_ROW_MAX = 5        # 「横長」と判定する `row` の検出数の上限
COL_MAX_PITCH = 2.5     # 地点の間隔の何倍までを `col` として認めるか
LATTICE_TOL = 0.25      # 格子と隙間のずれがこの割合(間隔比)を超えたら当てない
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
CHECK_MIN_COLS = 12     # これ以上の列があるときだけ，境と隙間を突き合わせる
CHECK_NEAR_PX = 10      # 境が隙間に乗っているとみなす距離
CHECK_MIN_RATIO = 0.7   # これを下回ると知らせる
ROW_COVER_MIN = 0.8     # 格子の行 / 組成部の字の行．これを下回ると知らせる
TOP_GAP_ROWS = 1.5      # 格子の上端が表頭の下端からこの行数より下なら，行を決め直す
ROW_VALLEY_K = 0.2      # 行の谷とみなす黒画素の量．組成部の中央値に対する比
ROW_VALLEY_MIN = 2      # 谷とみなす最小の幅(px)
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


def plot_gaps(dark, box, min_gap=GAP_MIN_PX, blank_ratio=BLANK_RATIO):
    """組成部の中の，地点と地点のあいだの隙間(中央の x)を返す

    **空白を「線」として捉える**．x ごとに「黒でない画素の割合」を出し，
    それが `blank_ratio` 以上つづく帯を，罫線の代わりの区切りとみなす
    (この資料には縦罫線が1本も無い．組成部の高さの 60% 以上つながる
     縦線を測ったが，最長でも 49%．2026-09-02)．

    **割合を絶対値で見るのが要**．前は黒画素の合計を**中央値との比**で
    見ていたが，それだと表の混み具合で基準が動く．50 表で測った結果
    (表頭から数えた地点数とのずれの合計)．

        黒でない画素の割合 0.95・最小幅 20 px   443   ← これ
        黒画素の合計 / 中央値比・最小幅  8 px   908
        黒画素の合計 / 中央値比・最小幅 20 px  1013

    最小幅だけを上げても**悪くなる**ので，効いているのは統計量の側．
    """
    x1, y1, x2, y2 = box
    sub = dark[y1:y2, x1:x2]
    if sub.size == 0:
        return []
    frac = (~sub).mean(axis=0)
    out, start = [], None
    for i, v in enumerate(frac):
        if v >= blank_ratio:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_gap:
                out.append(x1 + (start + i) // 2)
            start = None
    if start is not None and len(frac) - start >= min_gap:
        out.append(x1 + (start + len(frac)) // 2)
    return out


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


def _robust_pitch(gaps):
    """地点の間隔を，隙間の並びから頑健に見積もる

    隙間には**地点の中で割れたもの**が混ざるので，間隔をそのまま
    中央値にすると小さくなる．いちど中央値を取り，その 0.7-1.3 倍に
    入る間隔だけでもう一度中央値を取る．
    """
    d = np.diff(np.asarray(gaps, dtype=float))
    d = d[d > 0]
    if len(d) == 0:
        return 0.0
    m = float(np.median(d))
    keep = d[(d > m * 0.7) & (d < m * 1.3)]
    return float(np.median(keep)) if len(keep) else m


def column_edges_from_gaps(gaps, body_x1, body_x2, tol_ratio=LATTICE_TOL,
                           dark=None, y1=None, y2=None, steps=20):
    """地点の隙間から，等間隔の列の境を組み立てる

    組成部の地点の列は**等間隔に組まれている**．隙間はそれを写したものだが，
    地点の中で割れた偽の隙間が混ざり，逆に値の詰まった所では隙間が出ない．
    そこで**格子を当てはめる**．間隔を頑健に見積もり，位置は観測された
    隙間との距離が最小になるところへ合わせる．

    こうすると，検出の善し悪しに関係なく列の境が決まる
    (`col` の検出は短冊の継ぎ目や値の粗密で崩れる)．

    **位置合わせは，線上の黒画素が最小になるところ**で決める(2026-09-03)．
    `dark` を渡さなければ従来どおり「隙間から格子までの距離」で決める．
    地点の中で割れた偽の隙間が混ざると距離では位相が引っぱられ，
    格子の線が印字を割る(19_p1 は「90」「3.5」が線で切れていた)．
    35 表で測ると**列の数は変わらない**(±1 一致 33/35，ずれの合計 18 で同じ)が，
    **線上の黒画素が 58.2 → 41.3 に下がる**．セルが字を割らなくなる分だけ
    読み取りが良くなる．

    Returns:
        境の並び(両端を含む)．組み立てられなければ None
    """
    inner = sorted(g for g in gaps if body_x1 < g < body_x2)
    if len(inner) < 3:
        return None
    pitch = _robust_pitch([body_x1] + inner + [body_x2])
    if pitch <= 0:
        return None
    n = int(round((body_x2 - body_x1) / pitch))
    if n < 2:
        return None
    if dark is not None and y1 is not None and y2 is not None:
        prof = dark[int(y1):int(y2), int(body_x1):int(body_x2)].sum(axis=0).astype(float)
        best, best_cost = 0.0, None
        for k in range(steps):
            off = pitch * k / steps
            lat = body_x1 + off + pitch * np.arange(n + 1)
            idx = [int(v - body_x1) for v in lat if 0 < v - body_x1 < len(prof)]
            if len(idx) < 2:
                continue
            cost = float(np.mean([prof[max(0, i - 1):i + 2].max() for i in idx]))
            if best_cost is None or cost < best_cost:
                best, best_cost = off, cost
        if best_cost is not None:
            return _lattice_edges(body_x1, body_x2, best, pitch, n)

    # 位置合わせ．格子の刻みを 1/10 ずつずらし，隙間に最も近い所を採る
    best, best_cost = 0.0, None
    obs = np.asarray(inner, dtype=float)
    for k in range(10):
        off = pitch * k / 10.0
        lat = body_x1 + off + pitch * np.arange(n + 1)
        lat = lat[(lat >= body_x1 - pitch * 0.5) & (lat <= body_x2 + pitch * 0.5)]
        if len(lat) < 2:
            continue
        # **観測された隙間が格子に乗っているか**を見る(逆向きにしない)．
        # 格子の点ごとに隙間を探すと，値が詰まっていて隙間の出ない所が
        # あるだけで当てはめに失敗する(s01115_02 は 88 列に対し隙間が 48 本)
        cost = float(np.mean([np.min(np.abs(lat - g)) for g in obs]))
        if best_cost is None or cost < best_cost:
            best, best_cost = off, cost
    if best_cost is None or best_cost > pitch * tol_ratio:
        return None                     # 等間隔に乗らない．格子は当てない
    return _lattice_edges(body_x1, body_x2, best, pitch, n)


def _lattice_edges(body_x1, body_x2, off, pitch, n):
    """位相 `off` の等間隔の格子から，列の境の並びを作る"""
    lat = body_x1 + off + pitch * np.arange(n + 1)
    lat = [float(v) for v in lat if body_x1 - pitch * 0.5 <= v <= body_x2 + pitch * 0.5]
    edges = sorted({float(body_x1), *lat, float(body_x2)})
    # 端で潰れた区間はまとめる
    out = [edges[0]]
    for v in edges[1:]:
        if v - out[-1] < pitch * 0.4:
            out[-1] = v
        else:
            out.append(v)
    return out if len(out) >= 3 else None


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
        '仕切りが細くて画像の段階では切れなかったので，関門1で境目を目で確かめる']


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


EDGE_INK_GAIN = 0.7      # 隙間から作った格子を採るのは，線上の黒画素がこの倍以下のとき
EDGE_SWAP_MIN_COLS = 12  # 列がこれ未満の表では入れ替えない(隙間が少なく当てはめが利かない)
EDGE_SWAP_MIN_KEEP = 0.8  # 入れ替え後の列が元のこの割合を下回るなら採らない


def fix_column_edges(img, df_loc, gain=EDGE_INK_GAIN, min_cols=EDGE_SWAP_MIN_COLS,
                     min_keep=EDGE_SWAP_MIN_KEEP):
    """**格子ができたあとに**，組成部の列の境だけを印字の隙間から組み直す

    検出から作った境は，地点の数を1つ余分に取ると間隔が足りなくなり，
    右へ行くほど値を割る(s01115_22_p2 は 26 列・118 px．印字は 25 列・125 px で，
    線上の黒画素 1.49 対 0.20．2026-09-04 ユーザ指摘)．

    **行が決まったあとに列だけ組み直す**．格子を作る途中で入れ替えると，
    組成部の左端が動いて行の高さの測り方まで変わり，行が半分になる表が出た
    (s01115_06_p1 は 50 行 → 25 行)．外側の境も検出のまま動かさない．

    **2026-09-02 の決定(短冊に分けない表の列は検出のまま)は変えない**．
    一律に置き換えると悪くなるので，**線上の黒画素で測って良いときだけ**替える．

    Returns:
        (格子, 警告のリスト)
    """
    import locate

    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty or comp['col'].nunique() < min_cols:
        return df_loc, []
    edges = sorted(set(comp['x1']) | set(comp['x2']))
    y1, y2 = float(comp['y1'].min()), float(comp['y2'].max())
    x1, x2 = float(edges[0]), float(edges[-1])
    dark = ink.binarize(img)
    gaps = plot_gaps(dark, (int(x1), int(y1), int(x2), int(y2)))
    lat = column_edges_from_gaps(gaps, x1, x2, dark=dark, y1=y1, y2=y2)
    if lat is None or len(lat) < 3:
        return df_loc, []
    lat = sorted({x1, *[float(v) for v in lat[1:-1]], x2})
    if len(lat) - 1 < (len(edges) - 1) * min_keep:
        return df_loc, []
    cur = locate._edge_ink(edges, dark, y1, y2)
    new = locate._edge_ink(lat, dark, y1, y2)
    if cur is None or new is None or new > cur * gain:
        return df_loc, []

    def rebuild(src, edges):
        """行の帯はそのまま，列だけ作り直す"""
        bands = (src.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
                 .sort_index())
        like = src.iloc[0].to_dict()
        cells = []
        for row, b in bands.iterrows():
            for i, (a, c) in enumerate(zip(edges[:-1], edges[1:]), start=1):
                r = dict(like)
                r.update(dict(row=row, col=i, note='', x1=float(a), x2=float(c),
                              y1=float(b.y1), y2=float(b.y2)))
                cells.append(r)
        return pd.DataFrame(cells), bands

    cells, _ = rebuild(comp, lat)
    keep = df_loc[~df_loc['obj_name'].isin(('comp', 'header_value'))]
    parts = [keep, cells]
    # **表頭も同じ境で作り直す**(2026-09-05)．組成部だけ組み直すと，表頭は
    # 検出から作った古い境のままで**列の数が食い違う**(22_p2 は 26 対 25)．
    # 表頭は本体と横位置がずれることがあるので，新しい境で測り直してずらす
    head = df_loc[df_loc['obj_name'] == 'header_value']
    if len(head):
        hb = (head.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
              .sort_index())
        hx, _ = locate._shift_edges_for_header(
            np.array(lat, dtype=float), img,
            [(float(r.y1), float(r.y2)) for _, r in hb.iterrows()])
        hcells, _ = rebuild(head, list(hx))
        parts.append(hcells)
    out = pd.concat(parts, ignore_index=True)
    if 'cell_id' in out.columns:
        out['cell_id'] = range(1, len(out) + 1)
    return out, [
        f'列の境を**印字の隙間から**組み直した({len(edges) - 1} 列 → {len(lat) - 1} 列，'
        f'線上の黒画素 {cur:.2f} → {new:.2f})．検出から作った境より，'
        '線上の黒画素が少ない．関門1で列の中身を目で確かめる']


# 境をずらせる幅(列の幅に対する比)．**広げると値が別の地点へ移る**(2026-09-05)．
# 49 表で測った得失(切れなくなった値 / 新たに切れる値 / 別の列へ移る値)
#   ±0.08  2,337 / 414 / 135    ← これ
#   ±0.12  2,886 / 480 / 422
#   ±0.22  3,462 / 595 / 962
# 0.08 なら，利得の 67% を取り違えの 14% で得られる
CROSS_SHIFT = 0.08
CROSS_MIN_BLOB = 2        # これより細い黒の連なりは字とみなさない(汚れ)
CROSS_MIN_GAIN = 0.9      # 割る回数がこの倍より減るときだけ入れ替える
# **列の少ない表には当てない**(2026-09-05)．kinki_047(3 列・幅 90 px)は
# 境が 2 本しかなく，ずらせる幅が広いので値が別のセルへ移り，
# 正解表の組成部が 1.000 → 0.625 に落ちた．地点の多い表でだけ効かせる
CROSS_MIN_COLS = 12


def crossing_counts(dark, bands, width):
    """x を境にしたとき，**字のかたまりを割る回数**を x ごとに数える

    行ごとに黒画素の連なり(字のかたまり)を拾い，その内側の x に 1 ずつ足す．
    「線上の黒画素」との違いは，**濃さで重みが付かない**こと．組成部は
    大半が非出現の `・` なので，濃さで測ると `・` の位置だけで決まり，
    数の少ない値の行が無視される(s01115_17_p1 は線上の黒画素が中央値の
    0.56 倍と良い値なのに，`2・3` の `3` が境で切れていた．2026-09-05)．
    """
    cross = np.zeros(int(width) + 1, dtype=np.int32)
    for y1, y2 in bands:
        prof = dark[int(y1):int(y2)].any(axis=0)
        if not prof.any():
            continue
        idx = np.flatnonzero(np.diff(
            np.concatenate(([False], prof, [False])).astype(np.int8)))
        for a, b in zip(idx[0::2], idx[1::2]):
            if b - a >= CROSS_MIN_BLOB:
                cross[a + 1:b] += 1
    return cross


def fix_edges_by_crossings(img, df_loc, shift=CROSS_SHIFT, gain=CROSS_MIN_GAIN,
                           min_cols=CROSS_MIN_COLS):
    """組成部の列の境を，**字を割る回数がいちばん少ない位置**へずらす

    列の数は変えず，外側の境も動かさない(`fix_column_edges()` と同じ用心)．
    ずらせる幅は列の幅の `shift` まで．同じ回数なら元の位置に近い方を採る．

    値は `・` を軸に `2・3` と組まれるので，境が数 px ずれるだけで
    右の 1 字が隣のセルに落ち，どちらのセルも値にならない．
    49 表で割る回数は 9,008 → 6,473 に減り，切れなくなった値は 2,337 件．
    **ずらせる幅を広げてはいけない**(`CROSS_SHIFT` の注記を見る)．

    Returns:
        (格子, 警告のリスト)
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty or comp['col'].nunique() < min_cols:
        return df_loc, []
    xs = np.array(sorted(set(comp['x1']) | set(comp['x2'])), dtype=float)
    if len(xs) < 4:
        return df_loc, []
    dark = ink.binarize(img)
    bands = [(r.y1, r.y2) for r in
             comp.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max')).itertuples()]
    cross = crossing_counts(dark, bands, dark.shape[1])
    pitch = float(np.median(np.diff(xs)))
    room = max(1, int(pitch * shift))
    out = [xs[0]]
    for v in xs[1:-1]:
        v = int(v)
        lo, hi = max(1, v - room), min(len(cross) - 1, v + room)
        seg = cross[lo:hi + 1]
        best = np.flatnonzero(seg == seg.min())
        out.append(float(lo + min(best, key=lambda i: abs(lo + i - v))))
    out.append(xs[-1])
    now = int(sum(cross[int(v)] for v in xs[1:-1]))
    aft = int(sum(cross[int(v)] for v in out[1:-1]))
    if now == 0 or aft > now * gain:
        return df_loc, []
    bandsr = (comp.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
              .sort_index())
    like = comp.iloc[0].to_dict()
    cells = []
    for row, b in bandsr.iterrows():
        for i, (a, c) in enumerate(zip(out[:-1], out[1:]), start=1):
            r = dict(like)
            r.update(dict(row=row, col=i, note='', x1=float(a), x2=float(c),
                          y1=float(b.y1), y2=float(b.y2)))
            cells.append(r)
    res = pd.concat([df_loc[df_loc['obj_name'] != 'comp'], pd.DataFrame(cells)],
                    ignore_index=True)
    if 'cell_id' in res.columns:
        res['cell_id'] = range(1, len(res) + 1)
    moved = int(sum(1 for a, b in zip(xs[1:-1], out[1:-1]) if a != b))
    return res, [
        f'列の境を**字を割らない位置**へずらした({moved} 本，最大 ±{room} px．'
        f'割る回数 {now} → {aft})．列の数は変えていない']


def align_header_columns(df_loc, img=None):
    """表頭の列を，本体の列の境から作り直す

    **表頭は本体より列が多いことがある**(2026-09-06 に 68 表中 33 表)．
    `layer_col.fix_columns()` が本体の先頭から「地点でない列」
    (種名の領域にかかる列・空の隙間・階層の列)を捨てるのに対し，
    表頭は捨てないためで，差は 1-3 列．

    **これは黙って誤ったデータになる**．`plot_table.py` は地点番号を
    **表頭の値の x の並び順**で振るので(`plots = {x1: i + 1 …}`)，
    表頭に余分な列が 3 本あると，項目がすべて 3 つずれた地点に付く
    (s01115_19_p2_t1 は本体 14 列に対し表頭 17 列)．

    **捨てるのではなく作り直す**．外れたセルを落とすだけでは，本体より
    細かく割れた表頭が残って数が合わない(33 表のうち 16 表が残った)．
    行の帯はそのままに，列の境だけ本体のものに置き換えれば，
    数は必ず一致する．表頭は本体より数十 px 横にずれることがあるので，
    `locate._shift_edges_for_header()` で境をずらしてから当てる．

    Returns:
        (格子, 警告のリスト)
    """
    import locate

    comp = df_loc[df_loc['obj_name'] == 'comp']
    head = df_loc[df_loc['obj_name'] == 'header_value']
    if comp.empty or head.empty:
        return df_loc, []
    n_body = int(comp['col'].nunique())
    n_head = int(head['col'].nunique())
    if n_body == n_head:
        return df_loc, []
    edges = np.array(sorted(set(comp['x1']) | set(comp['x2'])), dtype=float)
    bands = (head.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
             .sort_index())
    if img is not None:
        edges, _ = locate._shift_edges_for_header(
            edges, img, [(float(r.y1), float(r.y2)) for _, r in bands.iterrows()])
        edges = np.asarray(edges, dtype=float)
    like = head.iloc[0].to_dict()
    cells = []
    for row, b in bands.iterrows():
        for i, (a, c) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            r = dict(like)
            r.update(dict(row=row, col=i, note='', x1=float(a), x2=float(c),
                          y1=float(b.y1), y2=float(b.y2)))
            cells.append(r)
    out = pd.concat([df_loc[df_loc['obj_name'] != 'header_value'],
                     pd.DataFrame(cells)], ignore_index=True)
    if 'cell_id' in out.columns:
        out['cell_id'] = range(1, len(out) + 1)
    return out, [f'表頭の列を本体の境から作り直した({n_head} 列 → {n_body} 列)．'
                 '本体で捨てた列が表頭に残ると，表頭の項目が地点とずれる']


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
            '列の区切りが地点と合っていないおそれが強い．関門1で目で確かめる']


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


def body_row_valleys(dark, x1, x2, y1, y2, k=ROW_VALLEY_K, min_w=ROW_VALLEY_MIN):
    """組成部の**黒画素の谷**から，行の境を数えて返す

    **種名の列で数えてはいけない**(2026-09-02 ユーザ指摘)．
    1つの種が複数の階層に出るとき，種名は最初の行にしか印字されず，
    下の階層の行は空白になる．種名で数えると行を数え落とす．
    **組成部は非出現でも「・」があるので，どの行にも必ず字がある**．

    数え方は**黒画素の量の谷**．横方向に使った「空白の割合」は
    行方向では効かない(組成部は字がまばらで，行の内側でも 95% が空白に
    なり，行と行が繋がってしまう)．中央値の `k` 倍を下回る所を谷とする．
    実測(2026-09-02)．

        正しく組めている表   格子の行 / 谷 = 0.92 - 1.03
        行が落ちている表                    0.10 - 0.26

    Returns:
        谷の中央の y の並び
    """
    sub = dark[int(y1):int(y2), int(x1):int(x2)]
    if sub.size == 0:
        return []
    prof = sub.sum(axis=1).astype(float)
    thr = float(np.median(prof)) * k
    out, start = [], None
    for i, v in enumerate(prof):
        if v <= thr:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_w:
                out.append(int(y1) + (start + i) // 2)
            start = None
    return out


GUTTER_MIN_W = 20           # 種名の列と組成部のあいだの隙間とみなす最小の幅(px)
GUTTER_BLANK = 0.002        # 隙間とみなす，x ごとの黒画素の割合の上限
GUTTER_INK = 0.3            # 帯の中で隙間の x のこの割合に字が来たら，隙間が埋まった
BOTTOM_BLANK_MIN = 0.5      # 列の境がこの割合より埋まったら，組成部の外
BOTTOM_RUN = 3              # その帯がこれだけ続いたら，そこで打ち切る
RULE_RUN = 2.0              # 行間隔のこの倍より長く横に続く黒画素は罫線．境の判定から外す
# 流し込みを「字と字の間隔」で見分ける(2026-09-05)．横に太らせたときの伸びが，
# 値の行は 2.4-5.7 倍，文章の行は 1.5-1.7 倍
GROW_RATIO = 2.0            # 太らせても伸びが小さければ，文章の行
GROW_WIDTH = 0.3            # 太らせる幅(列の幅に対する比)
GROW_DENS = 0.8             # 連なりの中で文章の行が占める割合の下限
GROW_MIN_RUN = 3            # 連なりがこの帯数に満たなければ切らない
GROW_MIN_INK = 0.02         # 字がこれより薄い帯は判定しない


def _strip_long_runs(band, min_len):
    """横に `min_len` px 以上続く黒画素(罫線)を消した写しを返す"""
    out = band.copy()
    for i in range(out.shape[0]):
        row = out[i]
        if not row.any():
            continue
        pad = np.concatenate(([False], row, [False]))
        d = np.diff(pad.astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for a, b in zip(starts, ends):
            if b - a >= min_len:
                row[a:b] = False
    return out


def body_bottom(dark, x_edges, y1, y2, pitch, blank_min=BOTTOM_BLANK_MIN,
                run=BOTTOM_RUN, rule=RULE_RUN, gutter=None,
                gutter_ink=GUTTER_INK):
    """組成部の下端を，**列の境に字が来る**ところで決める

    表の下には「1回出現種」が**流し込み**で組まれる．`once_species` が
    検出されないと，列の検出がそこまで伸び，行の谷が文章の行を拾って
    格子が流し込みまで広がる(s01115_13_p1 は 192 行のうち下の 60 行ほどが
    文章だった．2026-09-03)．

    見分け方は**列の境**．組成部では地点と地点のあいだが空いているが，
    流し込みは全幅に字が続くので境が埋まる．
    境の `blank_min` 未満しか空いていない帯が `run` 行つづいたら，そこで切る．

    **罫線の行は帯から外してから見る**(2026-09-04)．表頭の下の罫線や
    種群を囲む枠線(2-4 px)は全部の境を一度に埋めるので，その直下で
    打ち切っていた(上端を表頭の直下まで広げたとたん，s01115_11_p1 が
    147 行 → 12 行，17_p1 が 474 → 114 に落ちた)．
    「境の帯の中で字のある行の割合」で見分ける案は測って採らなかった:
    罫線の帯は 0.00-0.12，流し込みの帯は 0.1-0.5 で重なり，0.15 にすると
    13_p1 の流し込みが 20 行ぶん組成部に混じる．罫線は**横に長い**ので，
    行間隔の `rule` 倍より長く横に続く黒画素を消してから見る．
    「幅の 3 割以上が黒い行を外す」案はスキャンの傾きで罫線が数行に
    またがると外れないので採らなかった(11_p1 が 2 行で打ち切られた)．

    **境の埋まりだけでは，全地点に値の入った密な行を流し込みと取り違える**
    (17_p1 の `5-4 5-4 …` の行は境の 97% が埋まる)．そこで，種名の列と
    組成部のあいだの隙間 `gutter` が取れたときは，**境と隙間の両方が
    埋まる帯**だけを数える．流し込みは幅いっぱいに字が流れるので両方を
    満たすが，密な行は隙間が空いたままで，種群の見出し(隙間をまたぐ)は
    境が空いたまま(2026-09-04．隙間だけで見る案は 22_p2 が見出しで切れた)．

    Returns:
        下端の y(打ち切る所が無ければ `y2`)
    """
    inner = [int(v) for v in x_edges if 0 <= int(v) < dark.shape[1]]
    if pitch <= 0 or y2 <= y1:
        return y2
    # 列が 1-2 本の表(1 地点の表)は境の空きを測れない．隙間が取れていれば
    # 隙間だけで見る(kinki_024 は「Lage d. Aufn.」の流し込みが 1 行分
    # 組成部に入っていた．2026-09-04)
    use_edges = len(inner) >= 3
    if not use_edges and gutter is None:
        return y2
    bad, y = 0, int(y1)
    step = max(1, int(pitch))
    while y + step <= y2:
        band = dark[y:y + step]
        if band.size == 0:
            break
        band = _strip_long_runs(band, int(step * rule))
        filled = True
        if use_edges:
            blank = 0
            for x in inner:
                a, b = max(0, x - 2), min(dark.shape[1], x + 3)
                if not band[:, a:b].any():
                    blank += 1
            filled = blank / len(inner) < blank_min
        if filled and gutter is not None:
            g = dark[y:y + step, int(gutter[0]):int(gutter[1])]
            filled = g.size > 0 and g.any(axis=0).mean() >= gutter_ink
        if filled:
            bad += 1
            if bad >= run:
                return float(y - step * (run - 1))
        else:
            bad = 0
        y += step
    return y2

def _dilate_h(prof, k):
    """1 次元の真偽の並びを，左右に `k` だけ太らせる"""
    out = prof.copy()
    for i in range(1, int(k) + 1):
        out[i:] |= prof[:-i]
        out[:-i] |= prof[i:]
    return out


def running_text_top(dark, x0, x1, y1, y2, pitch, cell_w,
                     ratio=GROW_RATIO, width=GROW_WIDTH, dens=GROW_DENS,
                     min_run=GROW_MIN_RUN, min_ink=GROW_MIN_INK, rule=RULE_RUN,
                     gutter=None, gutter_ink=GUTTER_INK):
    """いちばん下の**流し込み**が始まる y を返す(見つからなければ `y2`)

    `body_bottom()` の「列の境の埋まり」では足りない表がある(2026-09-05)．
    地点が 116 もある s01115_17_p1 では，流し込みでも境の 62% は字と字の
    あいだに当たって空くので，しきい値 0.5 に届かない．一方で本体は 80% 空く
    ので差はあるが，表ごとの相対値にすると**密な行の続く所**で早く切れる
    (17_p1 は y 9531 で切れて 128 行を失った)．

    分けられるのは**字と字の間隔**．文章は字が続けて並ぶので，横に少し
    太らせるとすぐ埋まる(伸び 1.5-1.7 倍)が，値の行は値と値のあいだが
    広く空いているので大きく伸びる(2.4-5.7 倍)．密な本体(05_p2 の
    `4.3 4.3 …`)でも 2.4 倍で，文章と重ならない．

    **下から遡って**連なりを探す．流し込みは必ず表のいちばん下にあるので，
    途中の紛らわしい帯で切らずに済む．連なりの中の文章らしい帯の割合が
    `dens` を下回ったら止める(緩めると，本体の中を数珠つなぎに遡って
    12_p1 が 152 行を失った)．

    Returns:
        流し込みの先頭の y(無ければ `y2`)
    """
    if pitch <= 0 or y2 <= y1 or x1 <= x0:
        return y2
    k = max(4, int(cell_w * width))
    step = max(1, int(pitch))
    flags, ys = [], []
    y = int(y1)
    while y + step <= int(y2):
        band = _strip_long_runs(dark[y:y + step, int(x0):int(x1)], int(step * rule))
        prof = band.any(axis=0) if band.size else np.zeros(1, dtype=bool)
        raw = float(prof.mean())
        # 字の薄い帯(空行)は判定できないので，文章でないことにする
        grow = float(_dilate_h(prof, k).mean()) / raw if raw > min_ink else np.inf
        ok = grow < ratio
        # **常在度の表は，本体の行も字が連なって見える**(2026-09-05)．
        # `III(+-2)` は 8 字あり，字と字の間隔だけでは文章と区別できない
        # (s01115_11_p1 は本体を 35 行切り落とした)．種名の列と組成部の
        # あいだの隙間は，文章なら埋まり，本体なら空くので，これを足す
        # (11_p1 の本体は 0.000，17_p1 の流し込みは 0.54-1.00)
        if ok and gutter is not None:
            g = dark[y:y + step, int(gutter[0]):int(gutter[1])]
            ok = g.size > 0 and float(g.any(axis=0).mean()) >= gutter_ink
        flags.append(ok)
        ys.append(y)
        y += step
    if not any(flags):
        return y2
    end = len(flags) - 1
    while end >= 0 and not flags[end]:
        end -= 1
    best, hit = None, 0
    for j in range(end, -1, -1):
        hit += bool(flags[j])
        length = end - j + 1
        if hit / length < dens:
            break
        if length >= min_run and flags[j]:
            best = j
    return float(ys[best]) if best is not None else y2


def body_extent(df_loc, df_det=None):
    """組成部の**本当の縦の範囲**を返す (y1, y2)

    格子の範囲で数えてはいけない(2026-09-03)．行の検出が表の一部にしか
    出ないと格子もそこだけになり，**その中で谷を数えても比が 1.0 になって
    検査が鳴らない**(s01115_18_p2 は種 200 行の表で，行の検出が 8 本
    しか出ず，格子は 10 行．谷も 10 で「行は落ちていない」と見えた)．

    **列の検出は表の全高に伸びる**ので，そちらで下端を測る
    (18_p2 は col が y 400-10107，row は 1445-2225)．
    下は「1回出現種」の帯の手前まで．
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty:
        return None
    y1, y2 = float(comp['y1'].min()), float(comp['y2'].max())
    if df_det is None:
        return y1, y2
    col = df_det[df_det['obj_name'] == 'col']
    if col.empty:
        return y1, y2
    y2 = max(y2, float(col['y2'].max()))
    # **上端も列の検出まで広げる**(2026-09-04)．下端は列で広げていたのに
    # 上端は格子のままだったので，`row` の最初の検出が表の途中にあると
    # 谷もそこから下しか数えず，表頭の直下の種群が黙って落ちていた
    # (68 表中 28 表・約 371 行．s01115_15_p2 は 65 行，22_p1 は 42 行)．
    # 検査も同じ上端を使っていたので「行が落ちている表 0」と出ていた
    y1 = min(y1, float(col['y1'].min()))
    # **上端は表頭の下**(2026-09-03)．行の検出が表題や表頭にも出ると，
    # 格子が上へ伸びて表頭を組成部として飲み込む(s01115_07_p1 は
    # comp が y 213 から始まり，表頭は y 575-1403 だった．
    # 端の 1 行が 1,246 px の高さになり，行の高さが極端に不揃いになる)
    # **上端にかかる，いちばん上の表頭(とそれに重なる表頭)だけ**を使う．
    # 表の中ほどや下に出た表頭(縦に重なる別の表のもの)で切ると，
    # 表の大半を捨ててしまう(s01115_23_p1_t2 は下段の表の表頭が
    # y 6847-7428 にあり，そこで切ると 6,165 px ぶんの行が消える)
    # 項目名の列(`header_col`)も表頭の目印に数える．`header` が出ないのに
    # `header_col` だけ出る表頭がある(s01115_23_p1_t2 の上段)
    head = df_det[df_det['obj_name'].isin(('header', 'header_col'))]
    if not head.empty:
        near = head[(head['y2'] > y1) & (head['y1'] < y1 + (y2 - y1) * 0.5)]
        if not near.empty:
            top = near.sort_values('y1').iloc[0]
            grp = near[near['y1'] < float(top['y2'])]      # 上の表頭に重なる箱
            bottom = float(grp['y2'].max())
            if y1 < bottom < y2:
                y1 = bottom
    once = df_det[df_det['obj_name'] == 'once_species']
    if not once.empty:
        top = float(once['y1'].min())
        if top > y1:
            y2 = min(y2, top)
    return y1, y2


def find_gutter(dark, x_lo, x_hi, y1, y2, min_w=GUTTER_MIN_W, blank=GUTTER_BLANK):
    """組成部の左端 `x_hi` にいちばん近い空白の縦の帯 (x1, x2) を返す(無ければ None)

    `y1`-`y2` は確実に表の行である範囲(組成部の上半分)を渡す．
    """
    sub = dark[int(y1):int(y2), int(x_lo):int(x_hi)]
    if sub.size == 0:
        return None
    prof = sub.mean(axis=0)
    found, start = None, None
    for i, v in enumerate(list(prof) + [1.0]):
        if v < blank:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_w:
                found = (int(x_lo) + start, int(x_lo) + i)
            start = None
    return found


def body_extent_ink(dark, df_loc, df_det=None):
    """`body_extent()` の下端を，流し込みの始まる所でさらに詰める

    種名の列と組成部のあいだの隙間が取れれば，列の境と隙間の両方が
    埋まる所で(`body_bottom(gutter=…)`)，取れなければ列の境だけで見る．
    """
    ext = body_extent(df_loc, df_det)
    if ext is None:
        return None
    y1, y2 = ext
    comp = df_loc[df_loc['obj_name'] == 'comp']
    edges = sorted(set(comp['x1']) | set(comp['x2']))[1:-1]
    pitch = float((comp['y2'] - comp['y1']).median())
    x_hi = float(comp['x1'].min())
    names = df_loc[df_loc['obj_name'].isin(('sname', 'species_col'))]
    x_lo = float(names['x1'].min()) if len(names) else 0.0
    gutter = find_gutter(dark, x_lo, x_hi, y1, y1 + (y2 - y1) * 0.5)
    xs = sorted(set(comp['x1']) | set(comp['x2']))
    cell_w = float(np.median(np.diff(xs))) if len(xs) > 2 else float(comp['x2'].max()
                                                                    - comp['x1'].min())
    # 字と字の間隔から見た流し込みの先頭(境の埋まりでは足りない表がある)
    grow = running_text_top(dark, xs[0], xs[-1], y1, y2, pitch, cell_w,
                            gutter=gutter)
    if len(edges) < 3 and gutter is None:
        # 境も隙間も測れない(1 地点の表で種名の箱が組成部に接している)．
        # 流し込みの始まりを確かめられないので，下端は検出の格子のまま
        # (kinki_024 は列まで広げると流し込みが 1 行入った．2026-09-04)
        return y1, min(y2, float(comp['y2'].max()), grow)
    return y1, min(body_bottom(dark, edges, y1, y2, pitch, gutter=gutter), grow)


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
            '関門1で表の下の方を目で確かめる．'
            '--conf を下げると取れることがある']


PITCH_SEARCH = (0.5, 1.6)   # 行の高さを探す範囲(事前値に対する倍率)
PITCH_MIN = 6               # 行の高さの下限(px)
LATTICE_SNAP = 0.3          # 境を黒画素の最小へ寄せる範囲(行の高さに対する比)
LATTICE_PULL = 0.5          # 寄せるとき，等間隔の位置から離れることへの罰(黒画素の代表値に対する比)


PITCH_HARMONIC = 0.8        # 自己相関の最大のこの倍以上なら，短い周期を優先する
PITCH_PROM_MIN = 0.15       # 自己相関の頂の突出度(中央値との差)がこれ未満の候補は使わない
PITCH_PROM_TOL = 0.1        # 最大の突出度からこの差以内の候補のうち，いちばん短い高さを採る
PITCH_MIN_ROWS = 12         # 組成部の高さが事前値のこの倍未満なら，自己相関を使わず事前値を採る


def row_pitch(prof, prior, search=PITCH_SEARCH, min_pitch=PITCH_MIN,
              harmonic=PITCH_HARMONIC, candidates=0):
    """組成部の黒画素の並びの**自己相関**から行の高さを求める

    行の高さは 1 つの表の中でほぼ一定なので(2026-09-04 ユーザ指摘)，
    黒画素の並びは行の高さの周期をもつ．事前値 `prior`(検出の行の高さ)の
    `search` 倍の範囲でいちばん相関の高いずれ幅を採る．
    谷の間隔から見積もってはいけない: 「・」だけの行は行ごと谷になり，
    汚れや破線で 1 つの行間が 3 つの谷に割れる(08_p1・kinki_034)．
    """
    x = np.asarray(prof, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = float(np.dot(x, x)) + 1e-9
    lo = max(min_pitch, int(prior * search[0]))
    hi = min(n // 3, int(prior * search[1]))
    lags = list(range(lo, hi + 1))
    if not lags:
        return [] if candidates else float(prior)
    rs = np.array([float(np.dot(x[:-lag], x[lag:])) / denom for lag in lags])
    if len(rs) == 0 or not np.isfinite(rs).any() or rs.max() <= 0:
        return [] if candidates else float(prior)    # 周期が取れない(字が無いなど)
    # 周期の 2 倍・3 倍でも相関は高い(08_p1 は 61 px，kinki_034 は 88 px を
    # 拾って行数が半分になった)．最大の `harmonic` 倍以上の**山の頂**のうち，
    # いちばん短いずれ幅を採る．裾を含めると山の手前の裾 (kinki_019 は 55 px
    # の山の裾 42 px) を拾い，行が割れすぎる(2026-09-04)
    peak = np.ones(len(rs), dtype=bool)
    peak[1:] &= rs[1:] >= rs[:-1]
    peak[:-1] &= rs[:-1] >= rs[1:]
    ok = np.flatnonzero(peak & (rs >= rs.max() * harmonic))
    if len(ok) == 0:
        ok = np.flatnonzero(rs >= rs.max() * harmonic)
    if candidates:
        base = float(np.median(rs))
        return [(float(lags[int(i)]), float(rs[int(i)]) - base) for i in ok[:candidates]]
    return float(lags[int(ok[0])])


def lattice_rows(prof, y1, y2, pitch, snap=LATTICE_SNAP, pull=LATTICE_PULL):
    """`y1` から `y2` まで，行の高さ `pitch` で進みながら境を並べる

    各境は，等間隔の位置から `snap` 倍の範囲で**黒画素が最小になる所**へ
    寄せる(位相のずれの蓄積を防ぐ)．黒画素が平らな所では等間隔の位置に
    近い方を採る(`pull`)．`prof` は `y1` を 0 とする行ごとの黒画素．

    Returns:
        境の並び(画像の y．両端を含む)
    """
    n = len(prof)
    if n < 2 or pitch <= 0:
        return [float(y1), float(y2)]
    k = 3
    sm = np.convolve(np.asarray(prof, dtype=float), np.ones(k) / k, mode='same')
    scale = float(np.median(sm[sm > 0])) if (sm > 0).any() else 1.0
    reach = max(1, int(pitch * snap))
    edges = [0.0]
    y = 0.0
    while True:
        target = y + pitch
        if target >= n - pitch * 0.5:
            break
        lo, hi = max(0, int(target) - reach), min(n, int(target) + reach + 1)
        if hi <= lo:
            break
        idx = np.arange(lo, hi)
        cost = sm[lo:hi] + pull * scale * np.abs(idx - target) / pitch
        y = float(lo + int(np.argmin(cost)))
        edges.append(y)
    edges.append(float(n))
    return [float(y1) + e for e in edges]


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
            '飲み込んでいることが多い．関門1で表の上端を目で確かめる']


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
            '関門1で表頭と組成部の境を目で確かめる']


ROW_EDGE_GAIN = 0.7     # 行の境の線上の黒画素がこの倍以下なら，決め直した方を採る
ROW_KEEP_MIN = 0.8      # ただし，決め直した行がいまの行のこの割合を下回るなら採らない


def _row_edge_ink(dark, edges, x1, x2):
    """行の境の線上の黒画素(組成部の中央値との比)．小さいほど字を割っていない"""
    if len(edges) < 3 or x2 - x1 < 10:
        return None
    prof = dark[:, int(x1):int(x2)].sum(axis=1).astype(float)
    inner = [int(e) for e in edges[1:-1] if 0 <= int(e) < len(prof)]
    if len(inner) < 2:
        return None
    med = float(np.median(prof[prof > 0])) if (prof > 0).any() else 0.0
    return (float(np.mean(prof[inner])) / med) if med else None


def expected_row_edges(dark, df_loc, df_det, ext):
    """組成部の縦の範囲 `ext` を，行の高さの格子で刻んだ境と，その高さを返す

    周期は**組成部と種名の列を別々に測り，短い方を採る**．どちらも
    2 倍に化けることがある: 組成部は「・」ばかりの表で周期が弱い
    (08_p1 59 px 対 種名 29 px，kinki_034 86 対 55)．種名は 1 種が階層で
    2 行に分かれる表で 1 行おきにしか字が無い(10_p1 59 対 組成部 28，
    12_p1 56 対 27)．短い方は 14 表すべてで目視の高さと ±3 px．
    事前値は `row` の検出の高さ(無ければ格子の行の高さ)．
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    by1, by2 = ext
    bx1, bx2 = float(comp['x1'].min()), float(comp['x2'].max())
    names = df_loc[df_loc['obj_name'].isin(('sname', 'species_col'))]
    nx1 = float(names['x1'].min()) if len(names) else bx1
    sub = dark[int(by1):int(by2), int(bx1):int(bx2)]
    if sub.size == 0:
        # 組成部の範囲が取れない(列を組み直したあとに幅が 0 になるなど)．
        # 行は決め直さない(呼び出し側は境が 2 本なら何もしない)
        return [float(by1), float(by2)], 0.0
    body = sub.sum(axis=1).astype(float)
    # **破線の行は黒画素から除く**(2026-09-04 ユーザ提案)．種群を囲む破線が
    # 「・」の行の中間に来ると，黒画素の並びに半分の周期が生まれ，行の高さを
    # 半分に誤る(傾きを直した 08_p2 は 34 px の表が 19 px・72 行になった)
    body[ink.dashed_rows(sub)] = 0.0
    name = dark[int(by1):int(by2), int(nx1):int(bx1)].sum(axis=1).astype(float)
    det_rows = df_det[df_det['obj_name'] == 'row'] if df_det is not None else []
    heights = (comp.groupby('row').agg(a=('y1', 'min'), b=('y2', 'max'))
               .pipe(lambda g: g['b'] - g['a']))
    prior = (float((det_rows['y2'] - det_rows['y1']).median()) if len(det_rows)
             else float(heights.median()))
    # **候補ごとに格子を作り，境の線上の黒画素がいちばん少ないものを採る**
    # (2026-09-04．列の位相合わせと同じ手)．自己相関には偽の頂があり
    # (kinki_019 は 1 行 52 px なのに 42 px の頂を拾い，5 行ごとに細い帯が
    # 入った)，正しい高さなら境は行間に落ちるが，違えば字を切る
    # **突き出ていない頂は候補にしない**(2026-09-04)．「・」ばかりの組成部は
    # 周期がほとんど無く，自己相関が平ら(傾きを直した 08_p2 は 17-38 px で
    # 0.30-0.35)．雑音の頂(突出度 0.09-0.12)を候補に入れると半分の高さを
    # 採る(34 px の表が 19 px・72 行)．正しい頂の突出度は最小でも 0.17 (08_p5)
    cands = {}
    for p, prom in row_pitch(body, prior, candidates=3):
        cands[p] = max(cands.get(p, -1.0), prom)
    if nx1 < bx1 and name.any():
        for p, prom in row_pitch(name, prior, candidates=3):
            cands[p] = max(cands.get(p, -1.0), prom)
    cands = {p: prom for p, prom in cands.items() if prom >= PITCH_PROM_MIN}
    # **突出度が最大のものから `PITCH_PROM_TOL` 以内で，いちばん短い高さ**を採る．
    # 2 倍の高さも同じくらい突き出るので短い方を，雑音の頂は突出度で落とす．
    # 「境の線上の黒画素が少ない高さ」で選ぶ案は取り下げた: 大きい高さほど
    # 境の置き場に自由があって有利で，08_p1 の正しい 30 px が 40 px に負けた
    # 行が 10 行前後しかない小さな表は自己相関の統計が効かない(kinki_013 は
    # 55 px の表で 42 px を拾い 7 行が 9 行に)．検出の高さをそのまま使う
    if cands and (by2 - by1) >= prior * PITCH_MIN_ROWS:
        top = max(cands.values())
        pitch = min(p for p, prom in cands.items() if prom >= top - PITCH_PROM_TOL)
    else:
        pitch = float(prior)
    # **境を寄せる黒画素は組成部を優先する**(2026-09-04)．種名と組成部を
    # 足した黒画素で寄せると，数字が種名より 10 px ほど上に組まれた表
    # (08_p2)で境が種名に引かれ，`2.2` が上下の行に割れた(28 セルが
    # 読めなかった)．組成部は非出現でも「・」が行の中央にあるので，読む対象
    # そのものに合わせる．組成部に字の無い行(種群の見出し)だけ種名で寄せる
    k = 3
    body_sm = np.convolve(body, np.ones(k) / k, mode='same')
    if name.any():
        scale = float(body.mean()) / (float(name.mean()) + 1e-9)
        snap_prof = np.where(body_sm > 0, body, name * scale)
    else:
        snap_prof = body
    return lattice_rows(snap_prof, by1, by2, pitch), pitch


def rows_from_body(image, df_det, df_loc, min_ratio=ROW_COVER_MIN):
    """組成部の谷から `row` の検出を作り直す

    **行の検出は，地点が多い表や字の細かい表で大きく落ちる**
    (s01115_11 の Tab.74 は種名 110 行の表から 12 行しか取れなかった)．
    組成部は非出現でも「・」があるので**どの行にも字がある**．
    その谷を数えれば，検出に頼らず行の境が決まる．

    `locate.py` は `row` の箱から行の境を作るので，谷から箱を作って
    差し替える．表の幅いっぱいの箱にする(行は幅いっぱいに広がるもの)．

    Returns:
        (作り直した検出, 警告)．作り直す必要が無ければ (None, [])
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty:
        return None, []
    dark = ink.binarize(Image.open(image))
    bx1, bx2 = float(comp['x1'].min()), float(comp['x2'].max())
    # **列の検出が覆う範囲**で数える(`body_extent`)．格子の範囲で数えると，
    # 行の検出が表の一部にしか出ないときに比が 1.0 になって気づけない
    by1, by2 = body_extent_ink(dark, df_loc, df_det)
    # **行の高さは 1 つの表の中でほぼ一定**(2026-09-04 ユーザ指摘)．
    # 谷をそのまま行にすると，「・」だけの行は行ごと谷になって数行が繋がり，
    # 汚れや破線で 1 つの行間が 3 つに割れる(08_p1 は 23 行の表が 38 行，
    # kinki_034 は 21 行の表が 30 行)．行の高さを黒画素の自己相関で求め，
    # その高さで進みながら各境を黒画素の最小に寄せる．
    # 周期は**種名の列も合わせて**測る．組成部だけでは「・」ばかりの表で
    # 周期が弱く，2 倍の高さを拾う(08_p1 59，kinki_034 86)．種名は毎行に
    # 字があるので周期がはっきり出る(14 表で目視の高さと ±3 px)
    edges, pitch = expected_row_edges(dark, df_loc, df_det, (by1, by2))
    have = int(comp['row'].nunique())
    if len(edges) < 3:
        return None, []
    # **格子の上端が表頭の直下から `TOP_GAP_ROWS` 行分より下でも決め直す**
    # (2026-09-04)．行の数が足りていても，表頭直下の見出し行と最初の種
    # (階層で 3 行に分かれることもある)が落ちている表が 6 つ残っていた
    # (07_p4 は 4 行)．`_extend_rows_to_block()` は端に 2 行までしか足さず，
    # 種名の箱は最初の種から始まるので見出し行に届かない．下端は流し込みの
    # 危険があるので条件に入れない
    # **下端も同じように見る**(2026-09-05)．種名の列を補うと行の高さの
    # 見積もりが変わり，「行は足りている」と判定されて決め直しが走らず，
    # 検出のままの格子が組成部の下 12 行を落としていた(s01115_22_p3 は
    # 84 行 → 72 行)．組成部の範囲は流し込みの手前で切ってあるので，
    # そこに届いていなければ行が落ちている
    top_gap = (float(comp['y1'].min()) - by1) / pitch if pitch > 0 else 0.0
    bottom_gap = (by2 - float(comp['y2'].max())) / pitch if pitch > 0 else 0.0
    # **「境が字を割っているか」で決め直しを判断する案は取り下げた**(2026-09-05)．
    # 列の境と同じ手を行にも当てたが，行では**どの範囲で測っても正しく順位が
    # 付かない**(組成部だけでは種名を割っていても分からず，種名も含めると
    # s01115_14_p1(250 行が正)と 13_p2(63 行が正)が同じ向きに出る)．
    #
    # 本体の問題は**行の高さの推定**にある．自己相関でも「種名の字の行の間隔」でも
    # 測れるのは**種名の間隔**で，1 種が階層ごとに複数行に分かれる表では
    # 刻むべき行の高さより大きく出る(14_p1 は 35 対 33，23_p1_t1_s2 は 34 対 20)．
    # 過大な高さで決め直すと**階層の行が落ちる**ので，行数が足りているうちは
    # 検出の格子を信じる(余分な空行は読めば `absent` になり，実害が小さい)
    if (have >= len(edges) * min_ratio
            and top_gap < TOP_GAP_ROWS and bottom_gap < TOP_GAP_ROWS):
        return None, []

    row = df_det[df_det['obj_name'] == 'row']
    if row.empty:
        return None, []
    x1 = float(min(df_det['x1'].min(), bx1))
    like = row.iloc[0].to_dict()
    made = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 4:
            continue
        r = dict(like)
        r.update(dict(obj_name='row', confidence=1.0, x1=int(x1), x2=int(bx2),
                      y1=int(a), y2=int(b)))
        made.append(r)
    if len(made) < 3:
        return None, []
    out = pd.concat([df_det[df_det['obj_name'] != 'row'],
                     pd.DataFrame(made)], ignore_index=True)
    return out, [
        f'**行を組成部の黒画素から決め直した**({have} 行 → {len(made)} 行，'
        f'行の高さ {pitch:.0f} px)．検出では行が大きく落ちていた．'
        '行の高さは表の中でほぼ一定なので，黒画素の周期から高さを求め，'
        'その高さで進みながら境を黒画素の最小に寄せた．関門1で行の対応を目で確かめる']


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
        '短冊の境目で列が重複していないか，関門1で目で確かめる')

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
            '関門1で列の中身を目で確かめる')
    elif lattice:
        warnings.append(
            '列の境を印字の隙間から決めようとしたが，**等間隔に乗らなかった**．'
            '検出から決めた境をそのまま使う．関門1で列を目で確かめる')

    out['source_image'] = str(image).replace('\\', '/')
    return out, warnings
