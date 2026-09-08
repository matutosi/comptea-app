"""組成部の行の高さを，その表の行の高さにそろえる(格子ができたあとの後処理)

**組成部の行の高さは 1 つの表の中で一定**(2026-09-08 ユーザ指摘．極端に高い行も
低い行も無い)．ただし**厳密な等倍にはしない**．紙面の伸び縮みや字の高さで
±2 割ほどは揺れるので，そのぶんの幅をもたせる(同日ユーザ指示)．

格子の行は `row` の検出と内挿から作るので，検出が 1 本欠けると内挿が
**細い行を挟み**(s01115_01_p3 の行 117 は 10 px)，以後の行が 1 行ずつずれる．
行ごとの高さが ±2 割に収まっていても，36→40→54→46 のように**ずれが累積**して
字がセルを上下にまたぐ(07_p1)．

格子ごと黒画素から決め直す手(`body_rows.rows_from_body`)は，行の高さの推定が
階層別の副行で 2 倍に出て階層の行を落とすため，行数が足りているときは使わない
(2026-09-05)．ここでは**大半の行は正しい**ことを前提に，行の高さは検出された行の
**中央値**から取る(自己相関ではないので 2 倍に化けない)．その刻みで格子を
並べ直し(`lattice_rows`．各境は黒画素の谷へ寄せる)，全体の位相を合わせる．

- 刻みが印字の行の**半分**(「・」や下線が半周期を作る)なら，先に 2 倍にする
- 種名・和名・階層の列は，組成部と**別の上下のずれ**をもつ(タイプ打ちでは文字と
  「・」の高さが 10 px ほど違う)．行の対応は添字で保ったまま，列ごとにずらす

行の境は同じ段の全クラス(組成・学名・和名・階層・要約)で共有しているので，
段ごとに境を直して全クラスのセルを組み直す．表頭のセルは触らない．
"""

import numpy as np
import pandas as pd

from . import ink
from .body_rows import lattice_rows


ROW_TOL = 0.25          # 行の高さの許容(中央値に対する比)．これを外れた行があれば並べ直す


MIN_ROWS = 5            # これより行の少ない段は触らない(中央値が当てにならない)


CV_MAX = 1.0            # 変動係数がこれを超える段は触らない(表題を飲み込んだ格子)


PHASE_REACH = 0.4       # 位相を探すずれ幅(行の高さに対する比)


PHASE_GAIN = 0.7        # 境の上の黒画素がこの倍以下に減るときだけ位相を動かす


FINE_SNAP = 0.12        # 位相を合わせたあと，境ごとに谷へ寄せる幅(行の高さに対する比)


CLASS_GAIN = 0.95       # 種名側のずれは，境の上の黒画素がこの倍以下に減れば動かす


HALF_AC_MAX = 0.12      # 格子の高さでの自己相関がこれ未満なら，その高さは周期でない


HALF_AC2_MIN = 0.25     # 2 倍の高さでの自己相関がこれ以上なら，そちらが本当の周期


HALF_MIN_ROWS = 30      # 半分の刻みの判定に要る最小の行数(少ないと自己相関が当てにならない)


HALF_RUNS_MAX = 0.6     # 字の区間の数が行数のこの倍以下なら半分の刻み(正しい表は 0.75 以上)


RUN_THR = 0.3           # 字の区間とみなす黒画素(正の値の中央値に対する比)


BODY_CLASSES = ('comp', 'sname', 'species_col', 'layer', 'summary')


def _profile(dark, x1, x2):
    """組成部の横幅ぶんの，行ごとの黒画素(3 px で平滑化)"""
    prof = dark[:, int(x1):int(x2)].sum(axis=1).astype(float)
    return np.convolve(prof, np.ones(3) / 3, mode='same')


def autocorr(prof, lag, slack=2):
    """黒画素の並びの自己相関(ずれ幅 `lag` の前後 `slack` px で最大のもの)"""
    x = np.asarray(prof, dtype=float)
    x = x - x.mean()
    d = float(np.dot(x, x)) + 1e-9
    best = -1.0
    for l in range(max(2, int(lag) - slack), int(lag) + slack + 1):
        if l >= len(x):
            break
        best = max(best, float(np.dot(x[:-l], x[l:])) / d)
    return best


def text_runs(seg, thr=RUN_THR):
    """黒画素の並びの中の，字のある区間の数(印字の行の数の目安)"""
    seg = np.asarray(seg, dtype=float)
    pos = seg[seg > 0]
    if len(pos) == 0:
        return 0
    on = seg > float(np.median(pos)) * thr
    return int(np.sum(on[1:] & ~on[:-1])) + int(on[0])


def is_half_pitch(prof, y1, y2, med, ac_max=HALF_AC_MAX, ac2_min=HALF_AC2_MIN,
                  min_rows=HALF_MIN_ROWS, runs_max=HALF_RUNS_MAX):
    """格子の行の高さが，印字の行の**半分**になっていないかを見る

    「・」や値の下線が行の中間に来ると黒画素に半分の周期が生まれ，行の高さを
    半分に誤ることがある(s01115_23_p1_t1_s2 は 38 px の紙面が 20 px・231 行)．
    正しい格子なら，組成部の黒画素は格子の高さで強く相関する(9 表で 0.20-0.74)．
    半分の刻みでは相関が消え(0.04)，2 倍の高さで戻る(0.41)．

    **自己相関だけでは足りない**(2026-09-08)．行の少ない表では標本が足りず，
    正しい 56 px の格子でも 0.07 対 0.53 のように出て，行が半分に減った
    (kinki_017・021・016)．行の対応が失われる最悪の誤りなので，2 つの歯止めを置く．
    (1) 30 行未満では判定しない．(2) 組成部の**字の区間**を数え，格子の行数の
    `runs_max` 倍以下のときだけ半分とみなす(23_p1 は 0.53．正しい表は 0.75-1.5)．
    """
    seg = np.asarray(prof[int(y1):int(y2)], dtype=float)
    if med <= 0 or len(seg) < med * min_rows:
        return False
    if autocorr(seg, med) >= ac_max or autocorr(seg, 2 * med) < ac2_min:
        return False
    n_rows = len(seg) / med
    return text_runs(seg) <= n_rows * runs_max


def lattice_edges(edges, prof, pitch):
    """境の両端のあいだを，行の高さ `pitch` の格子で並べ直す(各境は谷へ寄せる)"""
    y1, y2 = float(edges[0]), float(edges[-1])
    seg = np.asarray(prof[int(y1):int(y2)], dtype=float)
    out = lattice_rows(seg, y1, y2, pitch)
    return [float(e) for e in out]


def best_shift(inner, prof, med, reach=PHASE_REACH, gain=PHASE_GAIN):
    """境の並び `inner` をまとめて動かすとき，線上の黒画素の和が最小になるずれ幅(px)

    元の位置から `gain` 倍以下に減らないときは 0(動かさない)．
    """
    inner = np.asarray(inner, dtype=float)
    n = len(prof)
    if len(inner) < 3 or med <= 0 or n == 0:
        return 0

    def cost(d):
        ys = np.clip((inner + d).astype(int), 0, n - 1)
        return float(prof[ys].sum())

    span = max(1, int(med * reach))
    ds = list(range(-span, span + 1))
    costs = [cost(d) for d in ds]
    cmin = min(costs)
    # 谷の底が平らなら，その真ん中を採る(端に寄せると次の行の字に近づく)
    good = [d for d, c in zip(ds, costs) if c <= cmin * 1.02 + 1e-9]
    d = good[len(good) // 2]
    base = cost(0)
    if abs(d) < 2 or base <= 0 or cmin > base * gain:
        return 0
    return int(d)


def class_offsets(edges, dark, body, med, classes=('sname', 'species_col', 'layer')):
    """種名・和名・階層の列に，組成部とは別の上下のずれを与える

    タイプ打ちの紙面では「・」や数字と文字の上下位置が同じ行でも 10 px ほど
    ずれて打たれている(s01115_14_p1 は組成部に合わせた境が種名の 209 行中
    157 行を割った．08_p2 は逆向き)．行の境を全クラスで共有すると両方には
    合わないので，これらの列だけ境をまとめて動かす．行の対応は添字で保つ．

    Returns:
        {obj_name: ずれ幅 px}(0 は含めない)
    """
    inner = np.array(edges[1:-1], dtype=float)
    out = {}
    for cls in classes:
        g = body[body['obj_name'] == cls]
        if g.empty:
            continue
        prof = _profile(dark, g['x1'].min(), g['x2'].max())
        if len(prof) == 0 or prof.max() <= 0:
            continue
        # 行の対応は変えないので害が無い．少しでも減るなら動かす
        d = best_shift(inner, prof, med, gain=CLASS_GAIN)
        if d:
            out[cls] = d
    return out


def rephase_edges(edges, prof, med, reach=PHASE_REACH, gain=PHASE_GAIN,
                  fine=FINE_SNAP):
    """行の境の並びを**まとめて**上下に動かし，字の上を通らない位相にする

    行の高さが一定なら，境が字を割っているときは全部が同じ向きに同じだけ
    ずれている(s01115_14_p1 は全行が 1/3 行ほど上)．境ごとに谷へ寄せる手
    (`_snap_axis`)は，タイプ打ちの狭い行間では谷が浅く動かないことがある．
    まず全体のずれ幅を「境の上の黒画素の和が最小」で決め，そのあと境ごとに
    少しだけ谷へ寄せる．

    Returns:
        (動かした境, ずれ幅 px)．動かさなければ ずれ幅 0
    """
    edges = [float(e) for e in edges]
    inner = np.array(edges[1:-1])
    if len(inner) < 3 or med <= 0:
        return edges, 0
    n = len(prof)
    d = best_shift(inner, prof, med, reach=reach, gain=gain)
    if d == 0:
        # 動かさない表の境は触らない(良い表を数 px 揺らして ±25% を外す行を作らない)
        return edges, 0
    moved = inner + d
    # 谷が平らなら元の位置にいちばん近い所を採る(平らな所で端へ寄らないように)
    r = max(1, int(med * fine))
    out = [edges[0]]
    for y in moved:
        lo, hi = max(int(out[-1]) + 1, int(y) - r), min(n, int(y) + r + 1)
        if hi > lo:
            seg = prof[lo:hi]
            ties = np.flatnonzero(seg <= seg.min() + 1e-9)
            k = ties[int(np.argmin(np.abs(lo + ties - y)))]
            y = float(lo + k)
        out.append(float(y))
    out.append(edges[-1])
    out = list(np.maximum.accumulate(out))
    return out, int(d)


def _renumber_rows(df):
    """段ごとに y1 の順で行番号を振り直す(`locate.number_cells` と同じ決まり)"""
    rows = df[['block', 'y1']].drop_duplicates().sort_values(['block', 'y1'])
    rows['row'] = range(1, len(rows) + 1)
    return df.drop(columns=['row']).merge(rows, on=['block', 'y1'], how='left')


def _rebuild(g, body, edges):
    """段の全クラスのセルを，新しい行の境で組み直す(表頭はそのまま)"""
    head = g[~g['obj_name'].isin(BODY_CLASSES)]
    cols = (body[['obj_name', 'x1', 'x2']].drop_duplicates()
            .sort_values(['obj_name', 'x1']))
    y1 = np.array(edges[:-1])
    y2 = np.array(edges[1:])
    made = []
    for c in cols.itertuples(index=False):
        like = body[(body['obj_name'] == c.obj_name) & (body['x1'] == c.x1)]
        base = like.iloc[0].to_dict()
        # 列ぶんの気になる点(内挿など)は残す．行ぶんは付け直す
        x_note = ';'.join(sorted(set().union(
            *[set(str(v).split(';')) for v in like['note'].fillna('')])
            - {'', 'nan', 'interpolated', 'snapped', 'unresolved'}))
        for r in range(len(y1)):
            d = dict(base)
            d.update(dict(y1=float(y1[r]), y2=float(y2[r]),
                          note=';'.join(p for p in (x_note, 'row_fixed') if p)))
            made.append(d)
    nb = pd.DataFrame(made, columns=list(g.columns))
    return pd.concat([head, nb], ignore_index=True)


def fix_row_heights(img, df_loc, tol=ROW_TOL):
    """格子の行の高さを段ごとにそろえる

    Returns:
        (直した格子, 警告のリスト)．直す所が無ければ元の格子をそのまま返す
    """
    if df_loc is None or len(df_loc) == 0 or 'block' not in df_loc.columns:
        return df_loc, []
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty:
        return df_loc, []
    dark = None
    warnings = []
    pieces = []
    shifts = {}
    changed = False
    for block, g in df_loc.groupby('block', sort=True):
        body = g[g['obj_name'].isin(BODY_CLASSES)]
        cb = body[body['obj_name'] == 'comp']
        if cb.empty or cb['row'].nunique() < MIN_ROWS:
            pieces.append(g)
            continue
        ys = np.sort(cb['y1'].unique())
        edges = [float(y) for y in ys] + [float(cb['y2'].max())]
        h = np.diff(edges)
        med = float(np.median(h))
        if med <= 0 or float(h.std() / med) > CV_MAX:
            pieces.append(g)
            continue
        bad = int(((h < med * (1 - tol)) | (h > med * (1 + tol))).sum())
        if dark is None:
            dark = ink.binarize(img)
        prof = _profile(dark, cb['x1'].min(), cb['x2'].max())
        if len(prof) == 0 or not np.isfinite(prof).all() or prof.max() <= 0:
            pieces.append(g)
            continue
        relaid = False
        new = edges
        if is_half_pitch(prof, edges[0], edges[-1], med):
            # 刻みが印字の行の半分．2 倍の高さで組み直してから，あとの段階に通す
            med = 2 * med
            new = lattice_edges(edges, prof, med)
            relaid = True
            warnings.append(
                f'段{block}: **格子の行の高さ({med / 2:.0f} px)は印字の行の半分**だった'
                f'(組成部の黒画素が {med / 2:.0f} px では相関せず {med:.0f} px で相関する)．'
                f'{len(edges) - 1} 行 → {len(new) - 1} 行に組み直した．「・」や下線が行の中間に'
                '来ると半分の周期が出る．段階1で行の対応を目で確かめる')
        elif bad:
            # 行の高さが一定なら，外れた行は検出漏れか誤検出の内挿．
            # 行ごとに直すと ±2 割の中でずれが累積して字を割る(07_p1)ので，
            # 中央値の刻みで並べ直す
            new = lattice_edges(edges, prof, med)
            relaid = True
            warnings.append(
                f'段{block}: **行を中央値の高さ({med:.0f} px)で並べ直した**'
                f'(±{tol:.0%} を外れる行が {bad} 行あった．{len(edges) - 1} 行 → '
                f'{len(new) - 1} 行)．行の高さは表の中で一定なので，外れた行は'
                '検出漏れか誤検出の内挿．段階1で行の対応を目で確かめる')
        new, shift = rephase_edges(new, prof, med)
        if shift:
            warnings.append(
                f'段{block}: **行の境をまとめて {shift:+d} px 動かした**'
                f'(行の高さ {med:.0f} px)．境が字の上を通っていた．'
                '行の高さが一定なので，ずれは全行で同じ向き．段階1で目で確かめる')
        offs = class_offsets(new, dark, body, med)
        if offs:
            warnings.append(
                f'段{block}: 種名側の行の境を組成部とは別に動かした('
                + '，'.join(f'{k} {v:+d} px' for k, v in offs.items())
                + ')．文字と「・」の上下位置が同じ行でずれて打たれている紙面．'
                '行の対応は変えていない')
        if not relaid and not shift and not offs:
            pieces.append(g)
            continue
        pieces.append(_rebuild(g, body, new))
        if offs:
            shifts[block] = offs
        changed = True
    if not changed:
        return df_loc, warnings
    out = pd.concat(pieces, ignore_index=True)
    # 行番号は**ずらす前**の y1 で振る(種名側をずらしたあとに振ると，
    # 同じ行の学名と組成が別の番号になる)
    out = _renumber_rows(out)
    for block, offs in shifts.items():
        for cls, off in offs.items():
            hit = (out['block'] == block) & (out['obj_name'] == cls)
            out.loc[hit, ['y1', 'y2']] = out.loc[hit, ['y1', 'y2']] + float(off)
    return out, warnings
