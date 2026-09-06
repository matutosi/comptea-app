"""組成部の縦の範囲と，行の境を決める

**行と列はアルゴリズムで決める**(2026-09-02 の方針)．行の高さは表の中で
ほぼ一定なので，黒画素の自己相関から刻みを求め，その刻みで進みながら境を
黒画素の谷へ寄せる．谷だけを行の切れ目にしてはいけない(「・」だけの行は
行ごと谷になり，汚れで 1 つの行間が 3 つに割れる)．

縦の範囲は，上は**上端をまたぐ表頭の下**，下は**1 回出現種の流し込みが
始まる所**．流し込みは，字と字の間隔(太らせたときの伸び)と，種名の列と
組成部のあいだの隙間の埋まりで見分ける．
"""

import numpy as np
import pandas as pd
from PIL import Image

from . import ink

Image.MAX_IMAGE_PIXELS = None


ROW_COVER_MIN = 0.8     # 格子の行 / 組成部の字の行．これを下回ると知らせる


TOP_GAP_ROWS = 1.5      # 格子の上端が表頭の下端からこの行数より下なら，行を決め直す


ROW_VALLEY_K = 0.2      # 行の谷とみなす黒画素の量．組成部の中央値に対する比


ROW_VALLEY_MIN = 2      # 谷とみなす最小の幅(px)


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
        'その高さで進みながら境を黒画素の最小に寄せた．段階1で行の対応を目で確かめる']
