"""組成部の先頭に紛れ込んだ階層の列を，表頭が空であることから見つける

`layer` は学習ラベルが 22 件しかなく，検出がほとんど育っていない．
`locate._guess_layer_column()` は「**種名の列と組成部の隙間**に字があるか」で
補うが，s01115 のように**階層の列が地点の列と同じ幅で組成部の先頭に並ぶ**
資料では効かない．`col` の検出がその列から始まるので，隙間そのものが無い．

見分け方は**表頭**．調査番号・調査年月日・海抜高といった表頭の値は
地点の列にだけ入り，階層の列は空になる．実測(2026-09-02)．

    s01115_01 の Tab.7  先頭の列 0.07 / 他の列の中央値   (37 列)
    s01115_18 の表      先頭の列 0.16 / 他の列の中央値   (27 列)
    地点の列はどれも 0.85 以上

見るのは**先頭の列だけ**にする．階層の列は必ず組成部の左端にあり，
値の少ない地点の列を巻き込んで消す事故を避けられる．

間違って階層としても，読んだ中身が階層として通らなければ段階2に挙がるので，
黙って値が化けることはない(`locate._guess_layer_column()` と同じ考え方)．
"""
import numpy as np
from PIL import Image

from . import ink

Image.MAX_IMAGE_PIXELS = None       # 折り込みは1億画素を超える

# 他の列の中央値に対する比．これ未満なら階層の列とみなす．
# **0.4 → 0.5**(2026-09-03)．階層の列でも表頭の枠線や汚れで 0.44 になる表が
# あった(10_p2)．66 表で 0.4/0.45/0.5/0.55/0.6 を測ると，0.45-0.55 は同じ結果で
# (±1 一致 64/66，ずれの合計 24．0.4 は 63/66・25)，**0.6 で崩れる**
# (06_p1 が 12 → 11，07_p3 が 8 → 5)．平らな範囲の真ん中を採る．
# 既存 88 枚は 0.4 と 0.5 で結果が変わらない
HEADER_INK_MAX = 0.5
MIN_COLS = 3            # 列がこれ未満だと中央値が当てにならない


def find_layer_column(img, df_loc, ink_max=HEADER_INK_MAX, min_cols=MIN_COLS):
    """組成部の先頭の列が階層の列なら，その列番号を返す(違えば None)

    Args:
        img   : 表の画像(PIL)
        df_loc: `locate.number_cells()` を通した格子
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    head = df_loc[df_loc['obj_name'] == 'header_value']
    if comp.empty or head.empty or comp['col'].nunique() < min_cols:
        return None
    hy1, hy2 = float(head['y1'].min()), float(head['y2'].max())
    if hy2 - hy1 < 10:
        return None
    cols = (comp.groupby('col')
            .agg(x1=('x1', 'min'), x2=('x2', 'max')).sort_index())
    dark = ink.binarize(img)
    # **縦の罫線は字と数えない**(表の境の線が列を通ると，字が無くても
    # 「値がある」と見えてしまう．`ink.thick_ratio` は罫線を落とす)
    vals = np.array([ink.text_ratio(dark, hy1, hy2, r.x1, r.x2)
                     for r in cols.itertuples()])
    first, rest = vals[0], np.median(vals[1:])
    if not rest or first / rest >= ink_max:
        return None
    return int(cols.index[0])


def mark_layer_column(img, df_loc, **kw):
    """先頭の列が階層の列なら `obj_name` を `layer` に変える

    セルを組成部から外すだけでよい．地点番号は `comp_table.py` が
    組成部の列を数え直して振る(`rank(method='dense')`)ので，
    階層の列を外せば地点番号は自動で 1 から振り直される．

    Returns:
        (書き換えた格子, 警告のリスト)
    """
    col = find_layer_column(img, df_loc, **kw)
    if col is None:
        return df_loc, []
    hit = (df_loc['obj_name'] == 'comp') & (df_loc['col'] == col)
    if not hit.any():
        return df_loc, []
    out = df_loc.copy()
    out.loc[hit, 'obj_name'] = 'layer'
    return out, [
        f'組成部の先頭の列(列 {col})は**表頭が空**なので，'
        f'地点の列ではなく**階層の列**とみなした({int(hit.sum())} セル)．'
        '地点番号はその次の列から 1 で振り直す．段階1で列の中身を目で確かめる']


NAME_OVERLAP = 0.5      # 種名などの箱と横にこれ以上重なる組成列は，組成部ではない
BODY_INK_MIN = 0.05     # 本体の黒画素が他の列の中央値のこれ未満なら，空の列
MAX_LEADING = 3         # 先頭(と末尾)から見る列の数の上限
WIDE_RATIO = 3.0        # 他の列の幅の中央値のこれ倍より広い先頭の列は，組成部ではない
SLIVER_RATIO = 0.5      # 幅の中央値のこれ倍未満の列は切れ端(隣に併合するか捨てる)


def _col_table(comp):
    return (comp.groupby('col')
            .agg(x1=('x1', 'min'), x2=('x2', 'max'), y1=('y1', 'min'), y2=('y2', 'max'))
            .sort_values('x1'))


def _merge_slivers(out, dark, body_min, sliver=SLIVER_RATIO):
    """幅が中央値の `sliver` 倍未満の列を，端なら捨て，内側なら隣に併合する

    05_p1_t2 は幅 16-28 px の切れ端が 4 本あった(中央値 45 px)．
    端の切れ端は本体が空なら捨てる．内側の切れ端は，併合した幅が中央値に
    近くなる側の隣に併合する(値がその切れ端に掛かっていることがあるので捨てない)．
    """
    comp = out[out['obj_name'] == 'comp']
    cols = _col_table(comp)
    if len(cols) < 3:
        return out, []
    med = float(np.median(cols['x2'] - cols['x1']))
    body = {c: ink.text_ratio(dark, r.y1, r.y2, r.x1, r.x2) for c, r in cols.iterrows()}
    bmed = float(np.median(list(body.values())))
    dropped, merged = [], []
    order = list(cols.index)
    for i, c in enumerate(order):
        r = cols.loc[c]
        w = float(r.x2 - r.x1)
        if w >= med * sliver:
            continue
        hit = (out['obj_name'] == 'comp') & (out['col'] == c)
        at_edge = i == 0 or i == len(order) - 1
        blank = bmed > 0 and body[c] / bmed < body_min
        if at_edge and blank:
            out = out[~hit]
            dropped.append(int(c))
            continue
        # 隣のうち，併合後の幅が中央値に近い方へ
        cands = []
        if i > 0:
            n = order[i - 1]
            cands.append((abs(float(cols.loc[n].x2 - cols.loc[n].x1) + w - med), n, 'x2', r.x2))
        if i < len(order) - 1:
            n = order[i + 1]
            cands.append((abs(float(cols.loc[n].x2 - cols.loc[n].x1) + w - med), n, 'x1', r.x1))
        if not cands:
            continue
        _, n, side, val = min(cands)
        nh = (out['obj_name'] == 'comp') & (out['col'] == n)
        out.loc[nh, side] = val
        out = out[~hit]
        merged.append((int(c), int(n)))
        cols.loc[n, side] = val
    warn = []
    if dropped:
        warn.append(f'幅の細い切れ端の列 {dropped} を捨てた(端にあって本体が空)．')
    if merged:
        warn.append(f'幅の細い切れ端の列を隣に併合した {merged}(切れ端 → 併合先)．'
                    '値がその境に掛かっていることがあるので，段階1で見る')
    return out, warn


def fix_columns(img, df_loc, ink_max=HEADER_INK_MAX, min_cols=MIN_COLS,
                overlap=NAME_OVERLAP, body_min=BODY_INK_MIN):
    """組成部の左端・右端に混ざった，組成部でない列を整理する(2026-09-03)

    折り込み(s01115)の 66 表で，地点数より列が 2 つ多い表が 8 つあった．
    実物を見ると，どれも**組成部の左端**に次のいずれかが付いていた．

    - 種名の領域の上に出た `col`(15_p5 は 学名+和名 の 965 px が 1 列)．
      **丸ごと捨ててはいけない**．その右端 1 列ぶんは地点 1 だった
      (15_p5 で地点 1 が消えた)．右端を他の列の幅だけ残す
    - 種名と組成部のあいだの**空の隙間**(05_p2・21_p4．表頭も本体も空)
    - 階層の列(表頭は空だが本体に S・K の字がある．`find_layer_column()` は
      先頭の 1 列しか見ないので，手前に空の列があると届かなかった)
    - 幅 16-28 px の**細い切れ端**(05_p1_t2)．`_merge_slivers()` で先に片づける

    **右端**には，表頭が空の**群の要約の列**(常在度．10_p2 の「調査区数 12」
    「(30)」「II(+-2)」)が付くことがある．通し番号が無いだけの実在の列だが，
    地点として下流に流すと常在度の文字列が被度になる．`summary` に変えて外す．

    先頭から順に見て，種名・階層の箱と重なる列は右端を残し，
    表頭が空(他の列の `ink_max` 未満)なら，本体も空の列は捨て，
    字のある列は `layer` にする．表頭に字がある列に当たったら止める．
    **階層の列は 1 本だけ**(字がいちばん多い列)．

    Returns:
        (整理した格子, 警告のリスト)
    """
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty or comp['col'].nunique() < min_cols:
        return df_loc, []
    dark = ink.binarize(img)
    out, warnings = _merge_slivers(df_loc.copy(), dark, body_min)

    comp = out[out['obj_name'] == 'comp']
    head = out[out['obj_name'] == 'header_value']
    cols = _col_table(comp)
    names = out[out['obj_name'].isin(('sname', 'species_col', 'header_col', 'layer'))]
    spans = [(float(a), float(b)) for a, b in
             names.groupby('col').agg(x1=('x1', 'min'), x2=('x2', 'max')).itertuples(index=False)] \
        if len(names) else []
    name_right = max([b for _, b in spans], default=0.0)
    hy1, hy2 = (float(head['y1'].min()), float(head['y2'].max())) if len(head) else (0, 0)
    have_head = len(head) > 0 and hy2 - hy1 >= 10
    widths = (cols['x2'] - cols['x1']).to_numpy(dtype=float)
    body = np.array([ink.text_ratio(dark, r.y1, r.y2, r.x1, r.x2) for r in cols.itertuples()])
    hb = (np.array([ink.text_ratio(dark, hy1, hy2, r.x1, r.x2) for r in cols.itertuples()])
          if have_head else None)
    dropped, layered, trimmed = [], [], []
    n = len(cols)
    for k, (c, r) in enumerate(cols.iterrows()):
        if k >= MAX_LEADING:
            break
        w = float(r.x2 - r.x1)
        on_name = any(min(r.x2, b) - max(r.x1, a) >= w * overlap for a, b in spans)
        rest_w = float(np.median(widths[k + 1:])) if k + 1 < n else 0.0
        if rest_w > 0 and w > rest_w * WIDE_RATIO:
            on_name = True
        rest_body = float(np.median(body[k + 1:])) if k + 1 < n else 0.0
        rest_head = float(np.median(hb[k + 1:])) if hb is not None and k + 1 < n else 0.0
        head_blank = hb is not None and rest_head > 0 and hb[k] / rest_head < ink_max
        body_blank = rest_body > 0 and body[k] / rest_body < body_min
        hit = (out['obj_name'] == 'comp') & (out['col'] == c)
        if on_name:
            # 右端 1 列ぶんが種名の箱の外にあれば，そこだけ地点の列として残す
            keep_x1 = float(r.x2) - rest_w
            if rest_w > 0 and keep_x1 >= name_right and float(r.x2) - keep_x1 >= rest_w * 0.5:
                out.loc[hit, 'x1'] = keep_x1
                trimmed.append(int(c))
                break
            out = out[~hit]
            dropped.append(int(c))
            continue
        if head_blank and body_blank:
            out = out[~hit]
            dropped.append(int(c))
            continue
        if head_blank:
            layered.append((int(c), float(body[k])))
            continue
        break
    if layered:
        # **階層の列は 1 本だけ**．表頭が空で本体に字のある列が 2 本以上あるとき，
        # 字がいちばん多い 1 本が階層の列で，残りは字のこぼれた隙間
        # (21_p4 で隙間・階層・隙間の 3 列が階層になり，`comp_table.py` が
        #  「左のもの」を採って階層が空になるところだった)
        keep = max(layered, key=lambda t: t[1])[0]
        for c, _ in layered:
            hit = (out['obj_name'] == 'comp') & (out['col'] == c)
            if c == keep:
                out.loc[hit, 'obj_name'] = 'layer'
            else:
                out = out[~hit]
                dropped.append(c)
        layered = [keep]

    # 右端: 表頭が空の列は群の要約の列
    summary = []
    if hb is not None:
        for k in range(n - 1, max(n - 1 - MAX_LEADING, 0), -1):
            c = cols.index[k]
            if int(c) in dropped or int(c) in layered:
                break
            rest_head = float(np.median(hb[:k])) if k > 0 else 0.0
            if rest_head <= 0 or hb[k] / rest_head >= ink_max:
                break
            hit = (out['obj_name'] == 'comp') & (out['col'] == c)
            if not hit.any():
                break
            out.loc[hit, 'obj_name'] = 'summary'
            summary.append(int(c))

    if trimmed:
        warnings.append(
            f'組成部の左端の列 {trimmed} は種名の領域にかかっていたので，右端 1 列ぶんだけを'
            '地点の列として残した(15_p5 で地点 1 がここにあった)．段階1で見る')
    if dropped:
        warnings.append(
            f'組成部の左端の列 {sorted(dropped)} を組成部から外した'
            '(種名の箱の上か，表頭も本体も空の隙間)．地点番号は詰めて振り直す')
    if layered:
        warnings.append(
            f'組成部の左端の列 {layered} を階層の列とみなした'
            '(表頭が空で，本体に字がある)．地点番号は詰めて振り直す')
    if summary:
        warnings.append(
            f'**組成部の右端の列 {sorted(summary)} は表頭が空**なので，群の要約の列'
            '(常在度など)とみなして地点から外した(obj_name は summary)．'
            '地点なら --conf を疑う．段階1で列の中身を目で確かめる')
    return out, warnings


# ---- 階層の列の幅を黒画素で決め直す (対策 E．2026-09-09) ----------------------------
#
# 階層の列は「表頭が空」であることから組成部の先頭の列を階層とみなして作るので，
# **幅は地点の列のまま**になる．記号 (B1・B2・S・K) の実際の幅とは合わず，81 段中
# 66 段で枠の右端が縦罫線の上に乗り，13.5% の行で記号が枠をはみ出していた．
#
# 幅は**票** (その x に黒画素がある行の数．対策 B の `header_cols` と同じ考え) の
# かたまりから決める．黒画素の和ではいけない (1 行の長い塊が谷を埋める)．
# 全段に当てると良い段と悪い段が釣り合うので (24 対 24)，**いまはみ出している段だけ**
# 直す．広げると種名を巻き込む紙面があるため，上限と「太いかたまりを越えない」枷を置く．

LAYER_LO_F = 0.05       # 票がこの割合 (行数比) を超える x を，かたまりとみなす
LAYER_GAP_P = 0.8       # 行の高さのこの倍までの隙間は，同じかたまりとしてつなぐ
LAYER_MIN_W = 0.25      # かたまりの幅の下限 (行の高さの倍数．5 px 未満は縦罫線の spike)
LAYER_MAX_W_P = 2.0     # 幅の上限 (行の高さの倍数)
LAYER_MAX_W_C = 1.5     # 幅の上限 (地点の列の幅の倍数．大きいほうを使う)


LAYER_COMP_MIN_W = 0.4  # 境を寄せたあと，組成の 1 列目に残す幅 (列の幅の割合)
LAYER_THICK_P = 1.5     # 幅がこの倍 (行の高さ) を超えるかたまりは種名とみなし，越えない
LAYER_CLIP_MIN = 0.05   # はみ出す行がこの割合を超える段だけ直す
LAYER_PAD = 2           # 決めた範囲の外側に足す余白 (px)


def _layer_votes(dark, x1, x2, rows, pitch):
    """x ごとに「黒画素のある行の数」(票) を返す

    罫線は**段の全高で 1 回**消す．1 行ぶんの帯ごとに消すと，表をまたぐ縦罫線が
    その帯の中では短く見えて残り，枠の右端に乗った罫線を記号と数えてしまう
    (81 段中 66 段で枠の右端が縦罫線の上にある．2026-09-09)．
    """
    from . import row_track
    x1, x2 = int(x1), int(x2)
    if x2 - x1 < 5 or not rows:
        return np.zeros(0)
    top = max(0, int(min(a for a, _b in rows)))
    bot = min(dark.shape[0], int(max(b for _a, b in rows)))
    if bot - top < 2:
        return np.zeros(0)
    clean = row_track.clean_rules(dark, x1, x2, top, bot, pitch)
    votes = np.zeros(x2 - x1, dtype=int)
    for a, b in rows:
        a, b = max(top, int(a)) - top, min(bot, int(b)) - top
        if b - a < 2:
            continue
        votes += (clean[a:b].sum(axis=0) > 0).astype(int)
    return votes


def _blobs(votes, need, gap):
    """票が `need` を超える x を，`gap` px までの隙間でつないだかたまり"""
    on = np.flatnonzero(votes > need)
    if on.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(on) > gap)
    starts = np.r_[on[0], on[breaks + 1]]
    ends = np.r_[on[breaks], on[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _clip_rows(dark, x1, x2, rows, pitch):
    """記号が枠の左右をはみ出している行の数 (罫線は段の全高で消してから見る)"""
    from . import row_track
    if not rows:
        return 0
    xa, xb = max(0, int(x1) - 1), min(dark.shape[1], int(x2) + 1)
    top = max(0, int(min(a for a, _b in rows)))
    bot = min(dark.shape[0], int(max(b for _a, b in rows)))
    if xb - xa < 3 or bot - top < 2:
        return 0
    clean = row_track.clean_rules(dark, xa, xb, top, bot, pitch)
    n = 0
    for a, b in rows:
        a, b = max(top, int(a)) - top, min(bot, int(b)) - top
        if b - a < 2:
            continue
        cols = clean[a:b].sum(axis=0) > 0
        if cols[0] or cols[-1]:
            n += 1
    return n


def refit_layer_width(img, df_loc):
    """階層の列の x を，記号の黒画素に合わせて決め直す

    直すのは**いま記号がはみ出している段だけ** (`LAYER_CLIP_MIN` を超える段)．
    行番号・行の境・他の列は変えない．

    Returns:
        (直した格子, 警告のリスト)．直す所が無ければ元の格子をそのまま返す
    """
    from . import row_track
    if df_loc is None or len(df_loc) == 0 or 'obj_name' not in df_loc.columns:
        return df_loc, []
    if 'block' not in df_loc.columns or 'row' not in df_loc.columns:
        return df_loc, []
    if not (df_loc['obj_name'] == 'layer').any():
        return df_loc, []
    dark = None
    out = df_loc
    warnings = []
    changed = False
    for block, g in df_loc.groupby('block', sort=True):
        lay = g[g['obj_name'] == 'layer']
        comp = g[g['obj_name'] == 'comp']
        names = g[g['obj_name'].isin(('sname', 'species_col'))]
        if lay.empty or comp.empty or lay['row'].nunique() < 5:
            continue
        pitch = float(np.median(comp['y2'].astype(float) - comp['y1'].astype(float)))
        if not pitch > 0:
            continue
        if dark is None:
            dark = ink.binarize(img)
        rows = [(float(a), float(b)) for a, b in
                lay.drop_duplicates('row')[['y1', 'y2']].values]
        x1, x2 = float(lay['x1'].min()), float(lay['x2'].max())
        n_clip = _clip_rows(dark, x1, x2, rows, pitch)
        if n_clip <= LAYER_CLIP_MIN * len(rows):
            continue
        # 帯は「種名の右端」から「組成部の左端」まで．種名が無ければ枠の 2 倍幅
        left = float(names['x2'].max()) if not names.empty else x1 - (x2 - x1)
        # 右は組成部の左端から**半行ぶん先まで**見る．記号が枠の右へはみ出している
        # (その先は縦罫線か余白であることが多い) 場合に，はみ出した分を拾うため
        right = float(comp['x1'].min()) + 0.5 * pitch
        if right - left < 10:
            continue
        votes = _layer_votes(dark, left, right, rows, pitch)
        if votes.size == 0:
            continue
        blobs = _blobs(votes, LAYER_LO_F * len(rows),
                       max(1, int(LAYER_GAP_P * pitch)))
        min_w = max(5, int(LAYER_MIN_W * pitch))
        blobs = [(a, b) for a, b in blobs if b - a >= min_w]
        if not blobs:
            continue
        # いまの枠と重なるかたまりのうち，票の山がいちばん高いものを芯にする
        cx1, cx2 = x1 - left, x2 - left
        over = [(a, b) for a, b in blobs if a < cx2 and b > cx1]
        if not over:
            continue
        core = max(over, key=lambda ab: int(votes[ab[0]:ab[1]].max()))
        # 太いかたまり (種名) は越えない
        thick = [(a, b) for a, b in blobs if b - a > LAYER_THICK_P * pitch]
        lo_stop = max([b for a, b in thick if b <= core[0]] + [0])
        hi_stop = min([a for a, b in thick if a >= core[1]] + [len(votes)])
        a, b = core
        cap = max(LAYER_MAX_W_P * pitch, LAYER_MAX_W_C * float(
            np.median(comp.groupby('col')['x2'].max() - comp.groupby('col')['x1'].min())))
        na = max(lo_stop, a - LAYER_PAD)
        nb = min(hi_stop, b + LAYER_PAD)
        if nb - na > cap:                       # 上限を超えたら芯の中心から切る
            mid = (a + b) / 2.0
            na = max(lo_stop, int(mid - cap / 2))
            nb = min(hi_stop, int(mid + cap / 2))
        nx1, nx2 = left + na, left + nb
        # **和名と共有している左端は動かさない** (2026-09-10 ユーザ指摘 45: kinki_045-1 は
        # 「B, S」の B が票の山 (S・K の列) の外にあり，左端を 1663 → 1758 px に寄せて
        # B を落とした)．「和名の右端と階層の左端は 1 本」の規則を破らない．直すのは右端
        if not names.empty and abs(x1 - left) <= 2:
            nx1 = x1
        if nx2 - nx1 < min_w or (abs(nx1 - x1) < 2 and abs(nx2 - x2) < 2):
            continue
        after = _clip_rows(dark, nx1, nx2, rows, pitch)
        if after > n_clip:
            warnings.append(f'段{block}: 階層の列の幅は直さなかった'
                            f'(直すと記号のはみ出しが {n_clip} → {after} 行に増える)')
            continue
        mask = (out['block'] == block) & (out['obj_name'] == 'layer')
        out = out.copy() if out is df_loc else out
        out.loc[mask, 'x1'] = float(nx1)
        out.loc[mask, 'x2'] = float(nx2)
        # **階層の右端と組成部の左端は 1 本にする** (2026-09-10．和名と階層の境と
        # 同じ規則)．記号が組成部の左端より右へ出ている紙面があり (09_p5 は
        # 「B1」「B2」の末尾が 20 px 食い込む)，どちらかが必ず字を割る．
        # 組成の 1 列目の左端をこちらへ寄せる (列の数は変えない)
        c_left = float(comp['x1'].min())
        c_w = float(np.median(comp.groupby('col')['x2'].max()
                              - comp.groupby('col')['x1'].min()))
        # 1 列目が細くなりすぎるなら動かさない (値が入らなくなる)
        if nx2 > c_left and (c_w <= 0 or c_left + c_w - nx2 >= c_w * LAYER_COMP_MIN_W):
            first = ((out['block'] == block) & (out['obj_name'] == 'comp')
                     & (out['x1'].astype(float) <= c_left + 0.5))
            out.loc[first, 'x1'] = float(nx2)
            warnings.append(
                f'段{block}: 階層の記号が組成部の左端より右へ出ていたので，'
                f'**組成の 1 列目の左端を階層の右端に合わせた** '
                f'({c_left:.0f} → {nx2:.0f} px)．階層と組成の境は 1 本')
        changed = True
        warnings.append(
            f'段{block}: **階層の列の幅を黒画素で決め直した** '
            f'({x1:.0f}-{x2:.0f} → {nx1:.0f}-{nx2:.0f} px．'
            f'記号のはみ出し {n_clip} → {after} / {len(rows)} 行)．'
            '段階1で階層の記号が入っているかを目で確かめる')
    if not changed:
        return df_loc, warnings
    return out, warnings
