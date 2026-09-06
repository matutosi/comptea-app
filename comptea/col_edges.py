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
    if 'cell_id' in out.columns:
        out['cell_id'] = range(1, len(out) + 1)
    return out, [f'表頭の列を本体の境から作り直した({n_head} 列 → {n_body} 列)．'
                 '本体で捨てた列が表頭に残ると，表頭の項目が地点とずれる']
