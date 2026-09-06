"""種名の列が検出されないとき，組成部の左の黒画素の帯から補う(2026-09-04)

折り込み(s01115)の 68 表のうち 3 表(08_p1・08_p2・18_p2)は `sname` も
`species_col` も 0 件で，格子が組成部しか作れなかった(実物には学名・和名が
ある)．`col` の検出は表の全高に伸びるので，その左側で x ごとの黒画素を
取れば，学名の帯と和名の帯が空白で分かれて見える．

**左が学名，右が和名**．ラベル済み 33 枚のすべてで学名が左にあり，幅も
学名 491-918 px に対し和名 265-407 px と学名の方が広い．
検出が 1 件でもあれば何もしない(検出の方が確か)．
"""
import numpy as np
import pandas as pd

import ink

NAME_CLASSES = ('sname', 'species_col')
SMOOTH = 25           # x ごとの黒画素をならす窓(px)
EDGE_INK = 0.02       # 帯の端とみなす黒画素の割合(字の芯は 0.12-0.25)
PEAK_MIN = 0.05       # 山とみなす黒画素の割合
PEAK_SEP = 150        # 学名の山と和名の山の最小の間隔(px)
VALLEY_RATIO = 0.5    # 谷が低い方の山のこの倍より浅ければ，同じ帯とみなす
WIDE_RATIO = 3.0      # `col` の幅の中央値の何倍までを組成部の列とみなすか
ISOLATED_GAP = 3.0    # 次の列との間が列の幅のこの倍を超えたら，離れた誤検出とみなす


def body_left(df_det, wide=WIDE_RATIO, gap=ISOLATED_GAP):
    """組成部の左端の x を，`col` の検出から返す(無ければ None)

    種名の領域の上に出た幅の広い `col` は数えない．
    **左に離れて出た 1 本も数えない**(2026-09-05)．種名の列の上に出た
    `col` が 1 本あるだけで左端がそこになり，種名の帯が探索の外に落ちる
    (s01115_13_p1 は x90-139 の 1 本で左端が 90 px になり，本体の x1191 から
     始まる学名・和名を補えなかった)．
    """
    col = df_det[df_det['obj_name'] == 'col']
    if col.empty:
        return None
    w = (col['x2'] - col['x1']).to_numpy(dtype=float)
    ok = col[w <= float(np.median(w)) * wide]
    if ok.empty:
        ok = col
    xs = np.sort(ok['x1'].to_numpy(dtype=float))
    if len(xs) < 3:
        return float(xs[0])
    # 左から見て，次の列との間が列の幅の `gap` 倍を超えるなら，そこまでは捨てる
    pitch = float(np.median(np.diff(xs))) if len(xs) > 2 else float(np.median(w))
    for i in range(len(xs) - 1):
        if xs[i + 1] - xs[i] <= max(pitch, float(np.median(w))) * gap:
            return float(xs[i])
    return float(xs[-1])


def top_below_header(df_det, y1, y2):
    """上端 `y1` にかかる表頭(いちばん上のものと，それに重なるもの)の下端"""
    head = df_det[df_det['obj_name'].isin(('header', 'header_col'))]
    if head.empty:
        return y1
    near = head[(head['y2'] > y1) & (head['y1'] < y1 + (y2 - y1) * 0.5)]
    if near.empty:
        return y1
    top = near.sort_values('y1').iloc[0]
    grp = near[near['y1'] < float(top['y2'])]
    bottom = float(grp['y2'].max())
    return bottom if y1 < bottom < y2 else y1


def ink_bands(dark, x1, x2, y1, y2, smooth=SMOOTH, edge=EDGE_INK,
              peak_min=PEAK_MIN, sep=PEAK_SEP, valley=VALLEY_RATIO):
    """x の範囲 [x1, x2) の字の帯を，**山と山のあいだの谷**で分けて返す

    空白の閾値では分けられない(2026-09-04 に測った)．学名と和名の隙間は
    長い学名や見出しがまたぐので，x ごとの黒画素は 0.002-0.008 と
    ゼロにならない．一方，字の芯は 0.12-0.25 で，隙間は必ず谷になる．
    ならした黒画素の山を左から拾い，隣り合う 2 つの山のあいだで
    いちばん低い x を帯の境にする．

    Returns:
        [(x1, x2), ...]．字が無ければ []
    """
    sub = dark[int(y1):int(y2), int(x1):int(x2)]
    if sub.size == 0:
        return []
    prof = sub.mean(axis=0).astype(float)
    k = max(1, int(smooth))
    sm = np.convolve(prof, np.ones(k) / k, mode='same')
    on = np.flatnonzero(sm > edge)
    if len(on) == 0:
        return []
    lo, hi = int(on[0]), int(on[-1]) + 1
    # 山: 前後 `sep`/2 の窓の中で最大になり，`peak_min` を超える点
    half = max(1, int(sep) // 2)
    peaks = []
    for i in range(lo, hi):
        if sm[i] <= peak_min:
            continue
        a, b = max(lo, i - half), min(hi, i + half + 1)
        if sm[i] < sm[a:b].max():
            continue
        if peaks and i - peaks[-1] < sep:
            continue                      # 平らな山の頂は最初の 1 点だけ
        peaks.append(i)
    if not peaks:
        return []
    # 同じ帯の中の小さな凸凹は 1 つの山にまとめる．隣り合う山のあいだの谷が，
    # 低い方の山の `valley` 倍より浅ければ同じ帯(学名の帯は 0.11-0.13 の
    # 平らな山で，そのまま拾うと列の内側で割れた．2026-09-04)
    merged = [peaks[0]]
    for p in peaks[1:]:
        q = merged[-1]
        if sm[q:p].min() > min(sm[q], sm[p]) * valley:
            if sm[p] > sm[q]:
                merged[-1] = p
            continue
        merged.append(p)
    peaks = merged
    cuts = [lo]
    for a, b in zip(peaks[:-1], peaks[1:]):
        cuts.append(a + int(np.argmin(sm[a:b])))
    cuts.append(hi)
    return [(int(x1) + a, int(x1) + b) for a, b in zip(cuts[:-1], cuts[1:])
            if b - a >= k]


NAME_OVERLAP = 0.5      # 既にある箱とこの割合以上重なる帯は，その列とみなす


def name_columns_from_ink(img, df_det):
    """`sname`・`species_col` の足りない方を，黒画素の帯から作って足す

    **片方だけ検出される表が多い**(折り込みの 68 表中 28 表．2026-09-05)．
    s01115_22_p2 は学名の列(x 28-612)だけが出ており，その右の和名の列
    (x 613-941)が立たないので**和名が 1 つも取れていなかった**．
    帯が 2 本あるのに片方しか検出されていないときは，足りない方を補う．

    Returns:
        (検出, 警告のリスト)．足すものが無ければそのまま返す
    """
    have = {k: df_det[df_det['obj_name'] == k] for k in NAME_CLASSES}
    if all(len(v) for v in have.values()):
        return df_det, []               # 両方ある
    left = body_left(df_det)
    col = df_det[df_det['obj_name'] == 'col']
    if left is None or col.empty:
        return df_det, []
    y1, y2 = float(col['y1'].min()), float(col['y2'].max())
    y1 = top_below_header(df_det, y1, y2)
    dark = ink.binarize(img)
    bands = ink_bands(dark, 0, left, y1, y2)
    if not bands:
        return df_det, ["種名の列が検出されず，組成部の左にも字の帯が無い．"
                        "この表に種名の列は無いとみなした"]
    like = col.iloc[0].to_dict()
    made, warn = [], []
    if len(bands) >= 2:
        pairs = [(bands[0], 'sname'), (bands[1], 'species_col')]
        if len(bands) > 2:
            warn.append(f'組成部の左に字の帯が {len(bands)} 本ある．'
                        '左の 2 本を学名・和名としたが，残りは使っていない．'
                        '段階1で列の対応を目で確かめる')
    else:
        pairs = [(bands[0], 'species_col')]
        warn.append('組成部の左の字の帯が 1 本だけなので，和名の列とみなした．'
                    '学名の列だったなら，読んだ中身で分かる')
    # **既にある列は作り直さない**．帯がその列と重なるなら飛ばす
    keep = []
    for (a, b), name in pairs:
        if len(have[name]):
            continue
        other = pd.concat([v for k, v in have.items() if k != name and len(v)]) \
            if any(len(v) for k, v in have.items() if k != name) else None
        if other is not None:
            ox1, ox2 = float(other['x1'].min()), float(other['x2'].max())
            # **重なりの割合では判定できない**(2026-09-05)．学名の箱が和名の列まで
            # 覆っていることがあり(s01115_22_p1 は sname が x16-639 で，和名の帯
            # 520-653 をほぼ含む)，割合で見ると「同じ列」とみなして補えなかった．
            # **その箱の中心が，この帯の中にあるか**で見る
            oc = (ox1 + ox2) / 2
            if a <= oc <= b:
                continue               # 検出済みの列と同じ帯
        keep.append(((a, b), name))
    pairs = keep
    if not pairs:
        return df_det, []
    for (a, b), name in pairs:
        r = dict(like)
        r.update(dict(obj_name=name, confidence=1.0,
                      x1=int(a), x2=int(b), y1=int(y1), y2=int(y2)))
        made.append(r)
    out = pd.concat([df_det, pd.DataFrame(made)], ignore_index=True)
    warn.insert(0, f"**種名の列を，組成部の左の黒画素の帯から補った**"
                   f"({', '.join(n for _, n in pairs)}．"
                   f"x {', '.join(f'{a}-{b}' for (a, b), _ in pairs)})．"
                   '段階1で列の位置を目で確かめる')
    return out, warn
