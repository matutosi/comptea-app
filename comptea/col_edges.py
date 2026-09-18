"""列の境を決める

地点の列は**等間隔に組まれている**．検出した `col` はその写しだが，短冊の
継ぎ目や値の粗密で崩れるので，印字の**隙間**から等間隔の格子を当て，
**字を割らない位置**へ寄せる．表頭の境も，本体の境から作り直す．

効くのは組み合わせたときだけで，隙間を良くしても格子に当てなければ
列の境には反映されない(2026-09-02 に測った)．
"""

import numpy as np
import pandas as pd

from . import ink
from .axes import _edge_ink


GAP_MIN_PX = 20         # 隙間とみなす最小の幅(300 dpi のスキャンで測った)


BLANK_RATIO = 0.98      # 「空白」とみなす，黒でない画素の割合


LATTICE_TOL = 0.25      # 格子と隙間のずれがこの割合(間隔比)を超えたら当てない


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
    from . import locate

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
    cur = _edge_ink(edges, dark, y1, y2)
    new = _edge_ink(lat, dark, y1, y2)
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
        '線上の黒画素が少ない．段階1で列の中身を目で確かめる']


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
    from . import locate

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
    # **項目名の列の右端は，値の左端まで届かせる** (2026-09-10)．左端の外の列を足す
    # 処理が，あとで捨てられる列を一時的に作ることがあり (kinki_003 は 1420-1516 px の
    # 余白)，そのとき項目名の右端がその列の左端で止まったままになる (96 px 短い)．
    # 字は切れないが，区切りが変わったと数えられ，見直しの対象が 36 表増えた
    ja = out['obj_name'] == 'header_item_ja'
    if not ja.any():
        ja = out['obj_name'] == 'header_item'
    if ja.any():
        x2 = float(out.loc[ja, 'x2'].max())
        if x2 < float(edges[0]):
            out.loc[ja & (out['x2'] >= x2 - 0.5), 'x2'] = float(edges[0])
    if 'cell_id' in out.columns:
        out['cell_id'] = range(1, len(out) + 1)
    return out, [f'表頭の列を本体の境から作り直した({n_head} 列 → {n_body} 列)．'
                 '本体で捨てた列が表頭に残ると，表頭の項目が地点とずれる']


# ---- 和名と階層の境を 1 本にする (2026-09-09 ユーザ提案) ---------------------------
#
# 階層は**和名の右**にある (階層のある 81 段すべて)．重なるのは 8 段だけで，
# しかも 3〜21 px．いまは「和名の右端」と「階層の左端」が別々に決まるので，
# 字を割る行が 685 ある．**1 本の境**にして，いまの境の ±1.5 行の窓で
# 「字を割る行がいちばん少ない x」に置くと 33 になる (悪化する段は無い)．
#
# 窓は必須．窓を外すと 10_p2 で境が 223 px 飛び，和名の列ごと外へ出る．
# 階層の無い段では，和名の右端を**字の右端**まで広げる (組成部は越えない)．
# はみ出す行が 477 → 16 になり，組成部に食い込む段は無い．

NL_WINDOW = 1.5         # 境を動かす窓 (行の高さの倍数)
NL_MIN_GAP = 4          # 列の幅がこれ未満になる位置へは動かさない (px)
NL_PAD = 3              # 字の右端に足す余白 (px)
NL_NEAR = 0.15          # 境の左右この幅 (行の高さの倍数) に字があれば「字を割る」とみなす．
                        # 字間 (2-5 px) や記号「S・K」の中 (見本で 14 px．行 56 px) は割るとみなし，
                        # 和名と記号の間 (見本で 40 px．接する段では 11 px ほど) は割らない．
                        # 0.25 では 11 px の隙間まで割るとみなした (test_name_layer_edge)


def _split_rows_near(dark, xs, y_rows, near):
    """境の候補 `xs` ごとに，左右 `near` px 以内に字がある行 (字を割る行) の数

    `_split_rows` は境の隣の 1 px だけを見るので，2 px の字間も「割らない」になる．
    和名の字間に境が落ちて名前を割っていた (見本の「オオツルウメモドキ」．2026-09-18)．
    """
    xs = np.asarray(xs, dtype=int)
    if xs.size == 0:
        return np.zeros(0, int)
    x1 = max(0, int(xs[0]) - near)
    x2 = min(dark.shape[1], int(xs[-1]) + near + 1)
    out = np.zeros(xs.size, int)
    for a, b in y_rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        col = dark[a:b, x1:x2].any(axis=0).astype(int)
        cum = np.concatenate(([0], np.cumsum(col)))
        i = xs - x1
        lo, hi = np.clip(i - near, 0, col.size), np.clip(i + near + 1, 0, col.size)
        mid = np.clip(i, 0, col.size - 1)
        left = cum[np.clip(i + 1, 0, col.size)] - cum[lo] > 0
        right = cum[hi] - cum[mid] > 0
        out += (left & right)
    return out


def _split_rows(dark, x, y_rows):
    """境 `x` の左右にインクが続く行 (字を割っている行) の数"""
    n = 0
    xi = int(x)
    if xi < 1 or xi + 1 >= dark.shape[1]:
        return 10 ** 6
    for a, b in y_rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        if dark[a:b, xi - 1].any() and dark[a:b, xi + 1].any():
            n += 1
    return n


def _ink_right(dark, x1, x2, y_rows):
    """帯の中で，インクのいちばん右の x (無ければ None)"""
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(x2))
    if x2 - x1 < 2:
        return None
    best = None
    for a, b in y_rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        cols = np.flatnonzero(dark[a:b, x1:x2].any(axis=0))
        if cols.size:
            r = int(cols[-1]) + x1
            best = r if best is None else max(best, r)
    return best


NL_BLANK = 0.5          # 和名の列を広げるとき，越えない空白の幅 (行の高さの倍数)


def _first_wide_blank(dark, x1, x2, y_rows, pitch, frac=NL_BLANK):
    """`x1` から右へ見て，**幅 `frac` 行ぶん以上の空白**が始まる x を返す (無ければ x2)

    和名の右にある空白のうち，語間より広いものは「別の列との境」．そこで止めないと，
    検出されなかった階層の記号を和名の列に取り込む．
    """
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(x2))
    if x2 - x1 < 3:
        return float(x2)
    need = max(3, int(frac * pitch))
    blank = np.array([not _has_ink_col(dark, x, y_rows) for x in range(x1, x2)])
    for a, b in _blank_runs(blank):
        if b - a >= need:
            return float(x1 + a)
    return float(x2)


def _blank_runs(blank):
    """空白の列が続く区間を [(始まり, 終わり)] で返す (終わりは含まない)"""
    b = np.asarray(blank)
    if b.size == 0 or not b.any():
        return []
    pad = np.concatenate(([0], b.astype(np.int8), [0]))
    d = np.diff(pad)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _has_ink_col(dark, x, y_rows):
    """境の候補 `x` の列に，どれかの行の字があるか"""
    xi = int(x)
    if xi < 0 or xi >= dark.shape[1]:
        return True
    for a, b in y_rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a >= 2 and dark[a:b, xi].any():
            return True
    return False


def _ink_left(dark, x1, x2, y_rows):
    """帯の中で，インクのいちばん左の x (無ければ None)"""
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(x2))
    if x2 - x1 < 2:
        return None
    best = None
    for a, b in y_rows:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        cols = np.flatnonzero(dark[a:b, x1:x2].any(axis=0))
        if cols.size:
            v = int(cols[0]) + x1
            best = v if best is None else min(best, v)
    return best


NL_RULE_FRAC = 0.9      # この割合以上の行で黒い x は印字の縦罫線とみなす
NL_RULE_SPREAD = 0.2    # 罫線の左に，この割合以上の行で黒い x が続けば罫線の傾きとみなす
NL_LAYER_KEEP = 0.5     # 境の右 (階層の列) に，階層の字のある行をこの割合以上残す
NL_LAYER_MIN = 3        # 階層の字のある行がこれ未満なら，残す条件をかけない
NL_CORE = 0.5           # 階層の記号の列の芯: この割合以上の行に字がある x


def _ink_rows(dark, x1, x2, y_rows):
    """行 × x の「字がある」の表 (縦罫線から右は消す) と，使った行を返す

    **縦罫線から右は字に数えない**．階層の列の右端には組成部との罫線があり，
    数えると「どこに境を置いても階層の字が残る」ことになる．罫線は少し傾いて
    隣の x にも散るので，全行で黒い x から左へ，黒い行の多い x が続く所までを
    罫線とみなす (見本では罫線の左 3 px が 18 行で黒かった)．
    """
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(x2))
    rows = [(max(0, int(a)), min(dark.shape[0], int(b))) for a, b in y_rows]
    rows = [(a, b) for a, b in rows if b - a >= 2]
    if x2 <= x1 or not rows:
        return np.zeros((0, max(0, x2 - x1)), bool), rows
    m = np.array([dark[a:b, x1:x2].any(axis=0) for a, b in rows])
    cover = m.mean(axis=0)
    rule = np.flatnonzero(cover >= NL_RULE_FRAC)
    if rule.size:
        left = int(rule[0])
        while left > 0 and cover[left - 1] >= NL_RULE_SPREAD:
            left -= 1
        m[:, left:] = False
    return m, rows


def _rows_right(dark, x1, x2, y_rows):
    """x ごとに「その x から `x2` までに字のある行の数」を返す (x1..x2-1 の配列)"""
    m, _rows = _ink_rows(dark, x1, x2, y_rows)
    if m.size == 0:
        return np.zeros(m.shape[1], int)
    # 右から累積した OR: その x より右に字があるか
    right = np.logical_or.accumulate(m[:, ::-1], axis=1)[:, ::-1]
    return right.sum(axis=0)


def _symbol_rows(dark, x1, x2, y_rows):
    """階層の記号のある行だけを返す (見つからなければ全行)

    記号の列の芯は，罫線の手前でいちばん右にある「半分以上の行に字がある x」の
    続き．芯に字の無い行 (種群の見出しなど，和名から記号の列まで横切る行) を
    外して境を決める．見出しの行が和名と記号のあいだを横切ると空白の帯が
    できず，記号「S・K」の中の隙間に境が落ちていた (見本．2026-09-18)
    """
    m, rows = _ink_rows(dark, x1, x2, y_rows)
    if m.size == 0:
        return list(y_rows)
    dense = np.flatnonzero(m.mean(axis=0) >= NL_CORE)
    if dense.size == 0:
        return list(y_rows)
    # いちばん右の続き
    end = int(dense[-1])
    start = end
    while start - 1 >= 0 and (start - 1) in set(dense.tolist()):
        start -= 1
    keep = m[:, start:end + 1].any(axis=1)
    picked = [r for r, k in zip(rows, keep) if k]
    return picked if len(picked) >= NL_LAYER_MIN else list(y_rows)


def fix_name_layer_edge(img, df_loc, window=NL_WINDOW):
    """和名の右端と階層の左端を 1 本の境にする (階層が無ければ字の右端まで広げる)

    Returns:
        (直した格子, 警告のリスト)．直す所が無ければ元の格子をそのまま返す
    """
    from . import row_track
    if df_loc is None or len(df_loc) == 0 or 'obj_name' not in df_loc.columns:
        return df_loc, []
    if 'block' not in df_loc.columns or 'row' not in df_loc.columns:
        return df_loc, []
    if not (df_loc['obj_name'] == 'species_col').any():
        return df_loc, []
    dark = None
    out = df_loc
    warnings = []
    changed = False
    for block, g in df_loc.groupby('block', sort=True):
        ja = g[g['obj_name'] == 'species_col']
        lay = g[g['obj_name'] == 'layer']
        comp = g[g['obj_name'] == 'comp']
        if ja.empty or comp.empty or ja['row'].nunique() < 5:
            continue
        pitch = float(np.median(comp['y2'].astype(float) - comp['y1'].astype(float)))
        if not pitch > 0:
            continue
        if dark is None:
            dark = ink.binarize(img)
        y_rows = [(float(a), float(b)) for a, b in
                  ja.drop_duplicates('row')[['y1', 'y2']].values]
        ja_r = float(ja['x2'].max())
        comp_l = float(comp['x1'].min())
        if lay.empty:
            # 階層が無い段: 和名の右端を字の右端まで広げる (組成部は越えない)．
            # ただし**広い空白は越えない**．越えると，検出されなかった階層の記号を
            # 和名の列に飲み込む (07_p3 の記号 1190-1265 が和名の列に入った．2026-09-09)
            stop = _first_wide_blank(dark, ja_r, comp_l, y_rows, pitch)
            right = _ink_right(dark, ja_r, stop, y_rows)
            if right is None or right + NL_PAD <= ja_r + 1:
                continue
            new = min(float(right + NL_PAD), stop, comp_l - NL_MIN_GAP)
            if new - float(ja['x1'].min()) < NL_MIN_GAP or new <= ja_r + 1:
                continue
            # 広げて字を割るようになるなら広げない (組成部の左端まで届くと，
            # 最初の地点の「・」に接することがある)
            n_old = _split_rows(dark, ja_r, y_rows)
            if _split_rows(dark, new, y_rows) > n_old:
                warnings.append(f'段{block}: 和名の列は広げなかった'
                                f'(広げると字を割る行が増える)')
                continue
            mask = (out['block'] == block) & (out['obj_name'] == 'species_col')
            out = out.copy() if out is df_loc else out
            out.loc[mask, 'x2'] = new
            changed = True
            warnings.append(
                f'段{block}: **和名の列の右端を字の右端まで広げた** '
                f'({ja_r:.0f} → {new:.0f} px．組成部は {comp_l:.0f} px)')
            continue
        lay_l = float(lay['x1'].min())
        lay_r = float(lay['x2'].max())
        base = (ja_r + lay_l) / 2.0
        lo = max(float(ja['x1'].min()) + NL_MIN_GAP, base - window * pitch)
        hi = min(lay_r - NL_MIN_GAP, base + window * pitch)
        if hi <= lo:
            continue
        xs = np.arange(int(lo), int(hi) + 1)
        # **どの行にも字が無い x の帯**があればそこに置く．「字を割る行」の数だけで
        # 選ぶと，語間の空白に落ちて和名の末尾が枠の外に残る (「アラゲミツバ|ツツジ」の
        # 類．この指標には出ない)．字が接している段 (81 段中 32 段) では帯が無いので，
        # そのときだけ「字を割る行が最小」に戻す
        near = max(2, int(NL_NEAR * pitch))
        # 割る行と空白の帯は，階層の記号のある行だけで数える (見出しの行を外す)
        y_sym = _symbol_rows(dark, xs[0], lay_r - NL_MIN_GAP, y_rows)
        blank = np.array([not _has_ink_col(dark, x, y_sym) for x in xs])
        counts = _split_rows_near(dark, xs, y_sym, near)
        # 字間より狭い空白の帯は採らない (字間に境が落ちて名前を割る)
        for a, b in _blank_runs(blank):
            if b - a < 2 * near + 1:
                blank[a:b] = False
        # **境の右に階層の字を残す** (2026-09-18)．階層の記号が和名に接していると
        # 左側に空白の帯が無く，記号と組成部のあいだの帯が「いちばん広い」として
        # 選ばれ，記号がまるごと和名の列に入っていた (見本で階層の読みが 25 → 0)
        kept = np.zeros(len(xs), int)
        tail = _rows_right(dark, xs[0], lay_r - NL_MIN_GAP, y_rows)
        kept[:min(len(xs), tail.size)] = tail[:len(xs)]
        ref = int(kept[min(len(xs) - 1, max(0, int(lay_l) - int(xs[0])))])
        if ref >= NL_LAYER_MIN:
            ok = kept >= NL_LAYER_KEEP * ref
            if not ok.any():
                continue
            blank = blank & ok
            counts = np.where(ok, counts, counts.max() + 1)
        runs = _blank_runs(blank)
        if runs:
            # **いちばん広い空白の帯**を採る．語間も空白になるが，和名と記号のあいだの
            # 空きの方が広い．同じ幅なら，いまの境に近い方
            w = max(b - a for a, b in runs)
            wide = [(a, b) for a, b in runs if b - a == w]
            a, b = min(wide, key=lambda ab: abs((ab[0] + ab[1]) / 2 + xs[0] - base))
            new = float(xs[0] + (a + b - 1) / 2.0)
            best = int(counts[xs == int(new)][0]) if (xs == int(new)).any() else 0
        else:
            best = int(counts.min())
            # 同点が続く区間のうち**いちばん広いもの**の中央に置く (同じ幅なら階層の
            # 左端に近い方)．空白の帯と同じ考え方で，見出しの行が横切って帯が
            # できない段でも，和名と記号のあいだの広い空きを選べる．
            # 以前は同点のうち階層の左端に近い 1 点を採っていて，記号「S・K」の
            # S と ・K のあいだを割っていた (見本．2026-09-18)
            spans = _blank_runs(counts == best)
            a, b = min(spans, key=lambda ab: (-(ab[1] - ab[0]),
                                              abs((ab[0] + ab[1] - 1) / 2 + xs[0] - lay_l)))
            new = float(xs[0] + (a + b - 1) // 2)
        n_old = int(_split_rows_near(dark, [ja_r, lay_l], y_sym, near).min())
        if abs(new - ja_r) < 1 and abs(new - lay_l) < 1:
            continue
        if best > n_old:
            warnings.append(f'段{block}: 和名と階層の境はまとめなかった'
                            f'(字を割る行が {n_old} → {best} に増える)')
            continue
        m_ja = (out['block'] == block) & (out['obj_name'] == 'species_col')
        m_lay = (out['block'] == block) & (out['obj_name'] == 'layer')
        out = out.copy() if out is df_loc else out
        out.loc[m_ja, 'x2'] = new
        out.loc[m_lay, 'x1'] = new
        changed = True
        warnings.append(
            f'段{block}: **和名と階層の境を 1 本にした** '
            f'(和名の右端 {ja_r:.0f} / 階層の左端 {lay_l:.0f} → {new:.0f} px．'
            f'字を割る行 {n_old} → {best})')
    if not changed:
        return df_loc, warnings
    return out, warnings

GAP_SNAP_NEAR = 0.45    # 境をこの割合 (列の幅) より近い隙間へ寄せる


GAP_FAR_MIN = 10        # これ以下のずれは寄せない (`checks.CHECK_NEAR_PX` と同じ
                        # 「隙間に乗っている」距離．寄せると副作用だけが出る)


GAP_BAD_RATIO = 0.5     # 境のこの割合以上が離れている表だけ直す (刻みのずれが
                        # 積もっている表)．数本だけずれている表は，そこが正しい


GAP_FAR_MED = 15        # ずれの中央値がこの px 以上の表だけ直す (kinki_070 は 27 px，
                        # 列を失った 19_p2 は 10 px．割合だけでは分かれなかった)


GAP_KEEP_W = 0.7        # 寄せたあと，隣の列がこの割合 (列の幅) より細くなるなら寄せない


def snap_to_plot_gaps(dark, x_edges, y_range, near=GAP_SNAP_NEAR, bands=None):
    """**列の境を，印字の地点の隙間へ寄せる** (2026-09-11 ユーザ指摘: kinki_070)

    列の刻みは内挿で決まるので，隙間が拾えない所では刻みがずれて積もります
    (kinki_070 は印字が 75 px 刻みなのに格子は 78〜80 px で進み，通し番号 3・4 の
    境が 27〜32 px 右にあって「KF」を割っていた)．**印字の隙間は紙面が示す正しい
    区切り**なので，近ければそこへ寄せます．

    **直すのは，刻みのずれが積もっている表だけ**です (境の半分以上が隙間から
    10 px 以上離れている)．数 px のずれまで寄せると，もともと正しい境が動いて
    列が細くなり，後段の「細い列の併合」で列ごと消えます (07_p3 は 8 → 5 列，
    19_p2 は 16 → 14 列，kinki_086 は階層が消えた．どれもずれの中央値は 4〜5 px)．

    寄せるのは (a) 隙間が 10 px 以上・列の幅の `near` 倍以下に離れており，
    (b) 隣の境を越えず，(c) 隣の列が細くなりすぎず，(d) 寄せて**字を割る回数が
    増えない**とき．「直して悪くならないこと」は他の直しと同じ歯止めです．

    `bands` は行の帯 [(y1, y2), ...]．(d) の「字を割る回数」は行ごとに数える
    (本体の全高を 1 つの帯にすると，ほぼ全部の x が 1 本のかたまりになり，
    回数が 0 か 1 しか取らず歯止めが効かなかった．2026-09-18)．
    渡さなければ `y_range` を 1 つの帯として扱う．

    Returns:
        (境, 寄せた本数)
    """
    xs = [float(v) for v in x_edges]
    if len(xs) < 4:
        return np.asarray(xs, dtype=float), 0
    w = float(np.median(np.diff(xs)))
    if not w > 0:
        return np.asarray(xs, dtype=float), 0
    y1, y2 = int(y_range[0]), int(y_range[1])
    gaps = plot_gaps(dark, (int(xs[0]), y1, int(xs[-1]), y2))
    if not gaps:
        return np.asarray(xs, dtype=float), 0
    gaps = np.asarray(gaps, dtype=float)
    dist = [float(np.min(np.abs(gaps - e))) for e in xs[1:-1]]
    if (np.mean([v > GAP_FAR_MIN for v in dist]) < GAP_BAD_RATIO
            or float(np.median(dist)) < GAP_FAR_MED):
        return np.asarray(xs, dtype=float), 0     # 刻みは積もっていない
    cross = crossing_counts(dark, bands or [(float(y1), float(y2))], dark.shape[1])
    moved = 0
    for i in range(1, len(xs) - 1):
        e = xs[i]
        g = float(gaps[int(np.argmin(np.abs(gaps - e)))])
        if not GAP_FAR_MIN < abs(g - e) <= w * near:
            continue
        if not (xs[i - 1] + 2 < g < xs[i + 1] - 2):
            continue
        if min(g - xs[i - 1], xs[i + 1] - g) < w * GAP_KEEP_W:
            continue
        if int(cross[int(g)]) > int(cross[int(e)]):
            continue
        xs[i] = g
        moved += 1
    return np.asarray(xs, dtype=float), moved

