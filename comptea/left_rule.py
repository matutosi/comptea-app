"""組成部の左端の列だけを，左の縦罫線より右へ押し出す (段階 2 の切り出しだけ)

2026-09-16 に足した．19_p2 は縦罫線の x が本体の上端から下端で 55 px 右へ
ずれるのに，格子の列の境は全行で同じ x だった．下の方ほど組成の左端が
罫線の左に入り込み，**階層の記号 `K` と罫線が地点 1 のセルに入る**
(要確認 70 件のうち 65 件が地点 1)．

**先に試した 2 案は取り下げた** (`docs/lessons.md`)．

- 画像を回す: 検出器の成績が落ち，既存の歯止めが拒否する．
  発動させると 21_p4 が列の半分，18_p2 が 48 行を失う．
- 罫線の傾きで**全列を一律に**ずらす: 紙面が剛体のように回っていないので，
  **箱の境にインクが乗る割合**が 21_p4 の下半分で 7% → 44% に悪化した．

そこで**罫線に接する左端の列の，左の境だけ**を動かす．動かすのは
**罫線より右へ押し出す向きだけ** (左へは動かさない — 記号を引き込むため)．
他の列と，左端の列の右の境には触らない．格子のファイルも変えない
(x1 で列を束ねる段が格子の側に 3 つある)．
"""
import numpy as np

RULE_BANDS = 12          # 本体を縦にこの数の帯に分けて，帯ごとに罫線を探す
RULE_SEARCH = 70         # 組成の左端から左右にこの px の範囲で罫線を探す
RULE_MIN_RUN = 0.15      # 帯の高さのこの割合以上**続けて**黒い x を罫線とみなす
RULE_MIN_BANDS = 6       # 罫線が見つかった帯がこれ未満なら測らない
RULE_DOUBLE = 20         # 芯の右この px 以内に並ぶ縦線も罫線の一部 (二重罫線 `‖`)
FIT_MAX_RESID = 8.0      # 直線からの外れがこの px を超えたら測り損ね
RULE_MARGIN = 10         # 罫線の右にこの px 空けて左の境を置く．**8〜12 が平らに最良**
                         # (左端の列の左の境にインクが乗るセル: 19_p2 は元 58・4 px で 91・
                         #  8 で 43・12 で 42・20 で 51．04_p1 は元 15・4 で 39・8〜16 で 0)．
                         # 4 px では罫線の太さと直線の外れ (最大 4 px) で縁に乗る
KEEP_W = 0.5             # 左端の列に残す幅の下限 (元の幅の割合)


def _rule_x(dark, x_from, x_to, ya, yb, min_run, double=RULE_DOUBLE):
    """帯 [ya, yb) の中の縦罫線の**右の縁**の x．足りなければ None

    いちばん長く続けて黒い x を芯にして，そこから `double` px 以内に並ぶ
    縦線 (続けて黒い長さが同じく足りるもの) の**いちばん右**を返す．
    **二重罫線** (`‖`) で左の線だけを拾うと，押し出した境が 2 本のあいだに
    落ちて右の線がセルに残り，`1` と読まれる (01_p2 は `2・1` が `1;2` になった．
    2026-09-16 に目視で見つけた)．
    """
    x_from = max(0, int(x_from))
    band = dark[ya:yb, x_from:max(x_from, int(x_to))]
    if band.size == 0:
        return None
    need = max(3, int((yb - ya) * min_run))
    runs = np.zeros(band.shape[1], dtype=int)
    for k in range(band.shape[1]):
        col = band[:, k]
        if not col.any():
            continue
        idx = np.flatnonzero(np.diff(
            np.concatenate(([False], col, [False])).astype(np.int8)))
        runs[k] = int((idx[1::2] - idx[0::2]).max())
    if not runs.size or int(runs.max()) < need:
        return None
    core = int(np.argmax(runs))
    right = core
    for k in range(core + 1, min(len(runs), core + double + 1)):
        if runs[k] >= need:
            right = k
    return x_from + right


def fit_rule(dark, comp, bands=RULE_BANDS, search=RULE_SEARCH,
             min_run=RULE_MIN_RUN, min_bands=RULE_MIN_BANDS,
             max_resid=FIT_MAX_RESID):
    """組成部の左の縦罫線に直線 x = a*y + b を当てる．当たらなければ None

    Returns:
        (a, b)
    """
    if comp is None or len(comp) == 0:
        return None
    x0 = float(comp['x1'].min())
    y1, y2 = int(float(comp['y1'].min())), int(float(comp['y2'].max()))
    if y2 - y1 < bands * 10:
        return None
    ys, xs = [], []
    for k in range(bands):
        a = y1 + (y2 - y1) * k // bands
        b = y1 + (y2 - y1) * (k + 1) // bands
        x = _rule_x(dark, x0 - search, x0 + search, a, b, min_run)
        if x is not None:
            ys.append((a + b) / 2.0)
            xs.append(float(x))
    if len(ys) < min_bands:
        return None
    slope, icpt = np.polyfit(ys, xs, 1)
    resid = np.abs(np.array(xs) - (slope * np.array(ys) + icpt))
    if float(resid.max()) > max_resid:
        return None
    return float(slope), float(icpt)


def edge_ink(dark, df, pad=1):
    """左端の列のセルのうち，**左の境 (±`pad` px) に黒画素が乗る**数"""
    comp = df[df['obj_name'] == 'comp']
    if comp.empty or 'col' not in comp.columns:
        return 0
    first = comp[comp['col'] == comp['col'].min()]
    h, w = dark.shape
    n = 0
    for _i, r in first.iterrows():
        y1, y2 = int(max(0, r['y1'])), int(min(h, r['y2']))
        a, b = int(max(0, r['x1'] - pad)), int(min(w, r['x1'] + pad + 1))
        if y2 > y1 and b > a and dark[y1:y2, a:b].any():
            n += 1
    return n


def push_first_column(df, rule, margin=RULE_MARGIN, keep_w=KEEP_W, dark=None):
    """左端の列の左の境を，罫線より右へ押し出した写しと，動かしたセルの数を返す

    押し出すのは**罫線が左の境を越えている行だけ**．左へは動かさない．

    **`dark` を渡すと，左の境に黒画素が乗るセルが減るときだけ採る**
    (2026-09-16)．全 147 表で測ると 56 表で発動し，減ったのは 25 表だけだった．
    変わらない 24 表には，**列の内側をまっすぐ走る線**を罫線と誤認して
    値の左を切りうるもの (kinki_004-1 は全セルを 53 px 押し出していた)，
    増えた 7 表 (06_p2 は 5 → 14) がある．「直して悪くならないこと」は
    他の直しと同じ歯止め．
    """
    out = df.copy()
    comp = out[out['obj_name'] == 'comp']
    if comp.empty or rule is None:
        return out, 0
    a, b = rule
    first = comp['col'].min() if 'col' in comp.columns else None
    mask = (out['obj_name'] == 'comp') & (out['col'] == first)
    idx = out.index[mask]
    if not len(idx):
        return out, 0
    x1 = out.loc[idx, 'x1'].astype(float)
    x2 = out.loc[idx, 'x2'].astype(float)
    ym = (out.loc[idx, 'y1'].astype(float) + out.loc[idx, 'y2'].astype(float)) / 2.0
    want = a * ym + b + margin
    cap = x2 - (x2 - x1) * keep_w                  # 幅を半分より細くしない
    new = np.minimum(np.maximum(x1, want), cap)
    moved = int((new > x1 + 0.5).sum())
    if not moved:
        return out, 0
    out.loc[idx, 'x1'] = new
    if dark is not None and edge_ink(dark, out) >= edge_ink(dark, df):
        return df.copy(), 0                    # 減らないなら採らない
    return out, moved
