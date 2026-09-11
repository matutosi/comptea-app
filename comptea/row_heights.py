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
- 上下端は，種名の箱まで広げて流し込みの手前で詰めた範囲まで行の高さで埋める．
  下端は組成部に字のある行だけ足し，流し込み(列の境が埋まる行)には入らない
- **行の境は全クラスで共有し，位相は組成部の谷に置く**(2026-09-08 ユーザ指示で共有に
  決めた．複数階層の種で和名の行が空くのは紙面どおりで，ずれではない)．
  タイプ打ちでは文字と「・」の高さが同じ行でも 10 px ほど違い，共有する限り
  どちらかが少し切れる．読む対象の値(組成部)を優先し，種名は上端が数 px 切れるのを許す．
  **測って取り下げた折衷**(146 表の「境が字を割る行」の割合で比べた．基準は
  学名 32%・和名 32%・階層 28%・組成 3%):
  (a) 列ごとに「割る行の数」を数えて和が最小の位相 → 学名 23%・和名 21%・階層 21%・
  組成 13%．数は減るが，学名の真ん中を境が通る表が出た(14_p1)．
  (b) 種名側と組成部の黒画素の和の外縁を上端・下端にして行数で等分 → 行の高さは
  完全にそろうが，紙面の伸縮で端の境が字に乗り，組成 46%・学名 37%．
  (c) 種名側の列だけ別のずれ → 学名の枠が組成より 1 行近くずれ，ユーザ指示で取り下げ

行の境は同じ段の全クラス(組成・学名・和名・階層・要約)で共有しているので，
段ごとに境を直して全クラスのセルを組み直す．表頭のセルは触らない．
"""

import numpy as np
import pandas as pd

from . import ink
from .body_rows import RULE_RUN, _strip_long_runs, body_extent_ink, lattice_rows


ROW_TOL = 0.25          # 行の高さの許容(中央値に対する比)．これを外れた行があれば並べ直す


MIN_ROWS = 5            # これより行の少ない段は触らない(中央値が当てにならない)


CV_MAX = 1.0            # 変動係数がこれを超える段は触らない(表題を飲み込んだ格子)


PHASE_REACH = 0.4       # 位相を探すずれ幅(行の高さに対する比)


PHASE_GAIN = 0.7        # 境の上の黒画素がこの倍以下に減るときだけ位相を動かす


FINE_SNAP = 0.12        # 位相を合わせたあと，境ごとに谷へ寄せる幅(行の高さに対する比)


HALF_AC_MAX = 0.12      # 格子の高さでの自己相関がこれ未満なら，その高さは周期でない


HALF_AC2_MIN = 0.25     # 2 倍の高さでの自己相関がこれ以上なら，そちらが本当の周期


HALF_MIN_ROWS = 30      # 半分の刻みの判定に要る最小の行数(少ないと自己相関が当てにならない)


HALF_RUNS_MAX = 0.6     # 字の区間の数が行数のこの倍以下なら半分の刻み(正しい表は 0.75 以上)
HALF_RUNS2_LO = 0.75    # 相関が曖昧なとき: 2 倍の刻みで 字の区間/行 がこの範囲なら半分 (23_p2 は 1.07)
HALF_RUNS2_HI = 1.5
HALF_AC_RATIO = 0.7     # 同上: 2 倍の刻みの自己相関が元の刻みのこの倍を下回れば触らない
HALF_AC_SOFT = 0.35     # 同上: 元の刻みの自己相関がこれ以上なら本物の刻み (17_p1 の副行は 0.566)


# **刻みの選び直し** (2026-09-10)．行の高さは検出した行の中央値から取るが，
# 誤検出の細い行が混じると中央値が低く出る (13_p2 は 29 px．印字は 35 px)．
# 低い刻みで並べ直すとずれが積もり，上の 18 行で字を割り，値が 1 行下へ落ちた．
# 黒画素の自己相関が**はっきり**強い刻みがあれば，そちらを採る．範囲を狭く
# 絞るのは，1.6〜1.8 倍に弱い山が立つ表が 9 つあるため (相関 0.08〜0.33)
PITCH_LO, PITCH_HI = 0.85, 1.35   # 探す刻みの範囲 (中央値の倍数)
PITCH_MIN_R = 0.35                # これ未満の相関では動かさない
PITCH_GAIN = 0.10                 # 中央値の刻みよりこれだけ強くないと動かさない


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
    n_rows = len(seg) / med
    runs = text_runs(seg)
    if runs > n_rows * runs_max:
        return False
    ac1, ac2 = autocorr(seg, med), autocorr(seg, 2 * med)
    if ac1 < ac_max and ac2 >= ac2_min:
        return True
    # **相関が曖昧でも，字の区間で決められる** (2026-09-10)．切り出しの範囲が変わった
    # 23_p2 は 20 px で 0.205・40 px で 0.191 と相関が両方とも中途半端で，上の判定を
    # 通らなかった (前の切り出しでは 0.04 対 0.41)．字の区間は 2 倍の刻みで 1 行 1 区間
    # (1.07) なので，そちらで半分と分かる．相関が 2 倍の刻みで極端に落ちるときは触らない
    # 元の刻みで強く相関する表は本物 (17_p1 は階層の副行が 22 px で 0.566)．触らない
    return (ac1 < HALF_AC_SOFT
            and HALF_RUNS2_LO <= runs / (n_rows / 2) <= HALF_RUNS2_HI
            and ac2 >= ac1 * HALF_AC_RATIO)


def refine_pitch(prof, y1, y2, med, lo=PITCH_LO, hi=PITCH_HI,
                 min_r=PITCH_MIN_R, gain=PITCH_GAIN):
    """行の高さを，黒画素の自己相関がはっきり強い刻みへ選び直す

    検出した行の中央値は，誤検出の細い行が混じると低く出ます．低い刻みで
    並べ直すと，`lattice_rows` が谷へ寄せながら進むあいだにずれが積もり，
    表の上の方で字を割ります (13_p2 は 29 px で組み，印字の 35 px に対して
    18 行で 2 行ぶんずれた)．

    動かすのは，(1) 範囲が中央値の 0.85〜1.35 倍，(2) 相関が `min_r` 以上，
    (3) 中央値の刻みより `gain` 以上強い，の 3 つがそろうときだけです．
    `is_half_pitch` (2 倍の刻み) とは別の話なので，そちらを先に見ます．
    """
    seg = np.asarray(prof[int(y1):int(y2)], dtype=float)
    if len(seg) < 10 or med <= 0:
        return float(med)
    lags = range(max(3, int(med * lo)), int(med * hi) + 1)
    rs = {L: autocorr(seg, L, slack=0) for L in lags}
    if not rs:
        return float(med)
    best = max(rs, key=rs.get)
    r0 = autocorr(seg, int(round(med)), slack=0)
    if rs[best] < min_r or rs[best] < r0 + gain or abs(best - med) < 2:
        return float(med)
    return float(best)


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
    # **両端の境も一緒に動かす**．内側だけ動かすと，端の行の高さが
    # 「行の高さ ± ずれ幅」になり，細い帯が残る(04_p2 の行 138．2026-09-08)
    # 谷が平らなら元の位置にいちばん近い所を採る(平らな所で端へ寄らないように)
    r = max(1, int(med * fine))
    out = [max(0.0, edges[0] + d)]
    for y in moved:
        lo, hi = max(int(out[-1]) + 1, int(y) - r), min(n, int(y) + r + 1)
        if hi > lo:
            seg = prof[lo:hi]
            ties = np.flatnonzero(seg <= seg.min() + 1e-9)
            k = ties[int(np.argmin(np.abs(lo + ties - y)))]
            y = float(lo + k)
        out.append(float(y))
    out.append(min(float(n), edges[-1] + d))
    out = list(np.maximum.accumulate(out))
    return out, int(d)


END_MIN_GAP = 0.75      # 範囲の端との差が行の高さのこの倍を超えたら，行を足しにいく(端を越えるのは 0.25 行まで)


END_MIN_INK = 0.25      # 足す行の黒画素 / 行の中央値．これ未満なら足さない(空の行は足さない)


END_TRIM_INK = 0.05     # 末尾の行の組成の黒画素 / 行の中央値．これ未満なら落とす(余分な行)


END_INNER = 0.25        # 下端の判定で除く，帯の上下それぞれの割合(上の行の下線と，直下の流し込みの字の上端が食い込む)


END_SHORT = 0.75        # 末尾の行がこの倍(中央値比)に満たなければ余りとみなして落とす


def _row_ink(prof, a, b):
    a, b = max(0, int(a)), min(len(prof), int(b))
    return float(prof[a:b].sum()) if b > a else 0.0


TEXT_BLANK_MIN = 0.7    # 帯の中で空いている列の境の割合がこれ未満なら，流し込みの行とみなす(表の行はほぼ全部空く．流し込みの 1 行目は語間が境に当たって 0.5 を超えた: 04_p2)


def text_like(dark, xs, a, b, rule):
    """帯 `a`-`b` が流し込み(文章)の行かどうか

    表の行は地点と地点のあいだ(列の境)が空いているが，流し込みは幅いっぱいに
    字が流れるので境が埋まる(`body_rows.body_bottom` と同じ見分け方)．
    組成部に字があるかだけでは流し込みも「字がある」ので止められず，
    04_p2 は流し込みの 10 行が表の行として足された(2026-09-08)．
    罫線は全部の境を一度に埋めるので，横に長い黒画素を消してから見る．
    """
    if not xs:
        return False
    band = dark[int(a):int(b)]
    if band.size == 0:
        return False
    band = _strip_long_runs(band, int(rule))
    blank = 0
    for x in xs:
        x = int(x)
        lo, hi = max(0, x - 2), min(band.shape[1], x + 3)
        if hi > lo and not band[:, lo:hi].any():
            blank += 1
    return blank / len(xs) < TEXT_BLANK_MIN


JUDGE_PAD = 0.3         # 帯を読むときに上下へ広げる割合 (行の高さの倍数)


ONCE_TALL = 1.25        # 帯の高さ / 行の高さ．これを超える帯は見出しでも落とさない
                        # (本物の行と混ざっている)



def extend_edges(edges, prof_comp, prof_all, med, ext, is_text=None, prof_names=None,
                 judge=None):
    """格子の上下端を，組成部の本当の範囲 `ext` まで行の高さで埋める

    格子の縦の範囲は `row` の検出で決まり，`_extend_rows_to_block()` は片側 2 行
    までしか足さず，端も列の箱の中央値で決める．`col` の箱が途中で止まると
    その下の行がまるごと落ちる(s01115_07_p2 は下 4 行．2026-09-08)．
    `ext` は種名の箱まで広げてから流し込みの手前で詰めた範囲
    (`body_extent_ink(names=True)`)なので，そこまで行の高さで足す．

    **下端の行は「組成部に字がある」か「学名と和名の両方に字がある」とき表の行**
    とみなす．表の下の「出現1回の種」の見出しは種名の側の 1 列にしか字が無く，
    全幅で見ると行として足されていた(02_p1_s2・04_p2・06_p1・06_p2 の余分な 1 行)．
    組成部は非出現でも「・」があるが，タイプの薄い「・」は二値化で消えることがある
    (14_p1 の最終行は 400 px 幅で黒画素 16)ので，組成部だけで決めると種の行を落とす．
    学名と和名の両方に字があれば種の行(見出しは片方だけ)．
    流し込み(列の境が埋まる行)は `is_text` で止める．
    **上端は全幅で判断する**．最初の行は種群の見出し(組成部は空)のことが多い．

    **行としても文章としても取れるときは，読んだ内容で決めます** (2026-09-11
    ユーザ指示)．表の最終行の真下に流し込みの 1 行目があると，帯の下端に文章の字が
    食い込んで境が埋まり，`is_text` が立ちます (kinki_079-1 の「ヤマルリソウ」は
    組成の黒画素 695 でしきい値 252 を超えているのに，文章として止まっていた)．
    `judge(a, b)` が 'species' を返せば表の行とみなします．

    Args:
        prof_names: {obj_name: 黒画素の並び}(sname・species_col・layer のうち有るもの)
        judge: 帯 (a, b) を読んで役割を返す関数．`is_row` と `is_text` が両方
            立ったときだけ呼ぶ (読むのは表の端の 1〜2 帯だけ)

    Returns:
        (境, 下に足した行数, 上に足した行数, 下から落とした行数)
    """
    out = [float(e) for e in edges]
    if len(out) < 2 or med <= 0 or ext is None:
        return out, 0, 0, 0
    lo, hi = float(ext[0]), float(ext[1])
    prof_names = prof_names or {}

    # 判定は**帯の中央 1/2 だけ**で測る．表の直下に流し込みがあると，
    # その 1 行目の字の上端が末尾の帯の下端に食い込み(14_p1 の行 217)，
    # 上の行の値の**下線**は帯の上端に食い込む(04_p2 の行 138)．どちらも空の行が
    # 「字がある」と判定されて落ちなかった(2026-09-08)．「・」も値も字も行の中央にある
    def mid(prof, a, b):
        h = (b - a) * END_INNER
        return _row_ink(prof, a + h, b - h)

    def base_of(prof):
        vals = [mid(prof, a, b) for a, b in zip(out[:-1], out[1:])]
        pos = [v for v in vals if v > 0]
        return float(np.median(pos)) if pos else 0.0

    base = base_of(prof_comp)
    if base <= 0:
        return out, 0, 0, 0
    name_base = {k: base_of(p) for k, p in prof_names.items()}
    name_base = {k: v for k, v in name_base.items() if v > 0}
    need = 2 if len(name_base) >= 2 else len(name_base)

    def is_row(a, b):
        if mid(prof_comp, a, b) >= base * END_MIN_INK:
            return True
        if not need:
            return False
        hits = sum(1 for k, v in name_base.items()
                   if mid(prof_names[k], a, b) >= v * END_MIN_INK)
        return hits >= need

    def is_flow(a, b):
        """文章の行か (**読みを正とする**)

        **読んだ内容を先に見る** (2026-09-11 ユーザ指示)．「出現 1 回の種」の見出しが
        読めたら，形の判定によらずそこから下は流し込みです (17_p1 は形の判定だけでは
        見出しの下に 2 行入っていた)．見出しの中には種名も並ぶので，見出しを先に見ます．
        読めて種名なら表の行です (kinki_079-1 の最終行)．
        """
        if judge is not None:
            kind = judge(a, b)
            # **落とすのは見出しと読めた帯だけ** (2026-09-11 ユーザ指示:
            # 欠落は絶対に避ける)．「1 行に何種も並ぶ」で落とすと，和名が
            # OCR の濁点で 2 つに割れた行 (22_p3)，帯に 2 行ぶんが入った行
            # (20_p1)，記号がカナに読まれた行 (08_p5・10_p1) が落ちた．
            # 取りすぎた流し込みは段階 3 が `note` を付ける
            # **落とすのは帯の高さが 1 行ぶんのときだけ**．帯に本物の最終行と
            # 見出しが両方入ることがあり (kinki_079-1 の「ヤマルリソウ」，
            # 15_p2 の「オクノカンスゲ」)，そのまま落とすと欠落する．
            # 実測: 純粋な見出しの帯は 1.14 行，混ざった帯は 1.47〜1.66 行
            if kind == 'once' and (b - a) <= med * ONCE_TALL:
                return True
            # **読めたら，形では止めない** (2026-09-11 ユーザ指示: 学名・和名・
            # 組成の欠落は絶対に避ける．多めに取ってから OCR で除外する)．
            # 列の境の空きは，地点の列が少ない表では表の行でも 0.38〜0.50 に
            # なり，流し込み (0.33〜1.00) と重なる (09_p5 は本物の 4 行が
            # 落ちていた)．読んで違うと分かったものだけ落とす
            return False
        return is_text is not None and is_text(a, b)

    trimmed = 0
    # 行の高さは一定なので，**中央値の 3/4 に満たない末尾の行は無条件に落とす**．
    # 並べ直し(`lattice_rows`)は最後に 0.5〜1.5 行の余りを残し，20 px の余りが
    # 直下の流し込みの字に触れて「字がある」と残った(14_p1 の行 219．2026-09-08)
    while len(out) > 2 and (out[-1] - out[-2]) < med * END_SHORT:
        out.pop()
        trimmed += 1
    while len(out) > 2 and (not is_row(out[-2], out[-1])
                            or is_flow(out[-2], out[-1])):
        out.pop()
        trimmed += 1
    # 足す行は**行の高さのまま**足す(範囲の端で切り詰めると短い行ができ，
    # 行の高さがそろわない)．範囲の端は箱の端なので数 px 越えてよい
    below = 0
    while hi - out[-1] > med * END_MIN_GAP:
        new = out[-1] + med
        if is_flow(out[-1], new):
            break                                   # 流し込みに入った
        if not is_row(out[-1], new):
            break
        out.append(new)
        below += 1
    above = 0
    base_all = float(np.median([_row_ink(prof_all, a, b)
                                for a, b in zip(out[:-1], out[1:])]))
    while out[0] - lo > med * END_MIN_GAP:
        new = max(out[0] - med, 0.0)
        if _row_ink(prof_all, new, out[0]) < base_all * END_MIN_INK:
            break
        out.insert(0, new)
        above += 1
    return out, below, above, trimmed


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


def fix_row_heights(img, df_loc, df_det=None, tol=ROW_TOL):
    """格子の行の高さを段ごとにそろえ，上下端を組成部の本当の範囲まで埋める

    `df_det` を渡すと，種名の箱まで広げて流し込みの手前で詰めた範囲
    (`body_extent_ink(names=True)`)を上下端の目安にする(無ければ端は触らない)．

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
        # **行の境は組成部と種名・和名・階層で共有し，位相は組成部の谷に置く**
        # (2026-09-08 ユーザ指示で共有に決めた．複数階層の種で和名の行が空くのは
        # 紙面どおりで，ずれではない)．タイプ打ちでは文字と「・」の高さが 10 px ほど
        # 違い，共有する限りどちらかが少し切れる．読む対象の値(組成部)を優先する．
        # 種名側との折衷(列ごとの「割る行の数」の和・上端下端からの等分)は測って
        # 取り下げた(モジュールの docstring)
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
            # 中央値の刻みで並べ直す．**刻み自体が低く出ていることがある**ので，
            # 黒画素の自己相関がはっきり強い刻みがあればそちらへ選び直す
            fine = refine_pitch(prof, edges[0], edges[-1], med)
            if fine != med:
                warnings.append(
                    f'段{block}: **行の高さを {med:.0f} px → {fine:.0f} px に選び直した**'
                    '(検出した行の中央値より，組成部の黒画素の自己相関がはっきり強い'
                    '刻み)．細い誤検出が混じると中央値が低く出て，並べ直すときに'
                    'ずれが積もる．段階1で行の対応を目で確かめる')
                med = fine
            new = lattice_edges(edges, prof, med)
            relaid = True
            warnings.append(
                f'段{block}: **行を中央値の高さ({med:.0f} px)で並べ直した**'
                f'(±{tol:.0%} を外れる行が {bad} 行あった．{len(edges) - 1} 行 → '
                f'{len(new) - 1} 行)．行の高さは表の中で一定なので，外れた行は'
                '検出漏れか誤検出の内挿．段階1で行の対応を目で確かめる')
        # 上下端: 種名の箱まで広げて流し込みの手前で詰めた範囲まで，行の高さで埋める
        grew = False
        if df_det is not None:
            ext = body_extent_ink(dark, g, df_det, names=True)
            prof_all = _profile(dark, body['x1'].min(), body['x2'].max())
            # 流し込みの見分けに使う x: 地点の列の内側の境と，種名と組成部のあいだの隙間
            xs = sorted(set(cb['x1']) | set(cb['x2']))[1:-1]
            names = body[body['obj_name'].isin(('sname', 'species_col', 'layer'))]
            if len(names):
                gap_a, gap_b = float(names['x2'].max()), float(cb['x1'].min())
                if gap_b - gap_a >= 10:
                    xs.append((gap_a + gap_b) / 2)
            rule = med * RULE_RUN

            def is_text(a, b):
                return text_like(dark, xs, a, b, rule)

            prof_names = {}
            for cls in ('sname', 'species_col', 'layer'):
                nc = body[body['obj_name'] == cls]
                if len(nc):
                    prof_names[cls] = _profile(dark, nc['x1'].min(), nc['x2'].max())
            # **行としても文章としても取れる帯は，種名の領域を読んで決める**
            # (2026-09-11 ユーザ指示)．範囲は学名の左端から組成部の手前まで．
            # 「出現 1 回の種」の見出しはこの幅に書かれており，流し込みの中にも
            # 種名が並ぶので，**見出しを先に見る**必要がある．読むのは表の端の
            # 1〜2 帯だけなので速度には響かない
            nm = body[body['obj_name'].isin(('sname', 'species_col'))]
            name_box = ((float(nm['x1'].min()), float(cb['x1'].min()))
                        if len(nm) else None)

            def judge(a, b, box=name_box, pad=med * JUDGE_PAD):
                if box is None:
                    return 'other'
                from . import row_kinds
                # 帯を広げて読む (端の帯は境が字の下端を切る)．採るのは
                # **字の中心が帯の中にある読みだけ** (`read_kind` の中で絞る)
                return row_kinds.read_kind(img, (box[0], a, box[1], b), pad=pad)

            new, n_below, n_above, n_trim = extend_edges(new, prof, prof_all, med, ext,
                                                         is_text=is_text,
                                                         prof_names=prof_names,
                                                         judge=judge)
            if n_below or n_above or n_trim:
                grew = True
                warnings.append(
                    f'段{block}: **格子の上下端を直した**(下に {n_below} 行・上に {n_above} 行'
                    f'足し，組成の字が無い末尾の {n_trim} 行を落とした)．'
                    '種名の箱まで広げて流し込みの手前で詰めた範囲を目安にし，'
                    '下端は組成部に字がある行だけ足す．段階1で表の端を目で確かめる')
        new, shift = rephase_edges(new, prof, med)
        if shift:
            warnings.append(
                f'段{block}: **行の境をまとめて {shift:+d} px 動かした**'
                f'(行の高さ {med:.0f} px)．境が字の上を通っていた．'
                '行の高さが一定なので，ずれは全行で同じ向き．段階1で目で確かめる')
        if not relaid and not grew and not shift:
            pieces.append(g)
            continue
        pieces.append(_rebuild(g, body, new))
        changed = True
    if not changed:
        return df_loc, warnings
    out = pd.concat(pieces, ignore_index=True)
    return _renumber_rows(out), warnings
