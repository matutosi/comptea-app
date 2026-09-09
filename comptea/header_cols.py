"""表頭の**独文と和文を分ける縦の境**を黒画素から決める (対策 B．2026-09-09)

表頭は左から「項目名 (独文)」「項目名 (和文)」「地点ごとの値」が並ぶ．いまは縦の
区切りが `header_col` の検出枠の右端 1 本しかなく，その置き場所が定まらない．
枠が和文にかかると**和名が 2 つに割れ** (kinki_004-1・027・032・036・071・085・088)，
枠が和文の右に出ると**独文と和文が 1 列にまとまる** (kinki_025・038・069・081-1，
s01115_01_p3・02_p1_s1)．146 表のうち 9 表は枠が値の列まで伸びて和文の列が作られない．
崩れる 11 表は `header_col` が 2 つ検出され，その和が境になっていた．

**「いちばん広い谷を採る」だけでは解けない** (2026-09-09 に実データで確かめた)．
20 表中 9 表では「和文と値のあいだ」の方が広く，そこを選ぶと独文と和文が 1 列になる．
また黒画素の**和**で谷を測ると，表題や学名の 1 行が和文の上を横切るだけで谷が埋まる
(kinki_004-1 は谷が 87 px → 15 px)．

そこで 2 つ変える．
- **x ごとに「字のある項目行の数」を数える** (票)．1 行が横切っても谷は埋まらない．
- **谷の右端の相対位置**で選ぶ．独文と和文のあいだの谷の右端は 20 表とも 0.53〜0.68
  (146 表でも 0.44〜0.77)，和文と値のあいだは 0.98 以上で，重なりがない．

146 表に当てると，境が字を横切る表は 43/112 → 9/112 になり (残る 9 表は本体の見出し行
の巻き添えで項目名は割れない)，和文側の黒画素の割合は 109/112 表で増え，和文の列が
作られる表は 103 → 112 になった．
"""

import numpy as np

from .body_rows import _strip_long_runs


MIN_BANDS = 3           # 項目行がこれより少なければ決めない


RULE_FRAC = 0.5         # 領域の幅のこの倍より長く続く黒は罫線として消す


MIN_W = 20              # 谷の幅の下限 (px)．語間 (10〜40 px) を落とす


MIN_W_FRAC = 0.05       # 同上 (領域の幅に対する比．大きいほう)


RIGHT_LO = 0.35         # 谷の右端の相対位置の下限


RIGHT_HI = 0.80         # 同上限 (和文と値のあいだの谷は 0.98 以上なので切れる)


THR_FRAC = 0.4          # 票の閾値の上限 (項目行の数に対する比)


BACK_OFF = 12           # 境を谷の右端からこれだけ左へ戻す (和文の 1 画目に触れないため．112 表で 5→178 本，8→163，12→152，16→194，20→229 と 12 が最少)


def vote_profile(dark, box, bands, rule_frac=RULE_FRAC):
    """x ごとに「黒画素のある項目行の数」を返す (票)

    黒画素の**和**ではいけない: 表題や学名の 1 行が和文の上を横切るだけで谷が埋まる．
    帯ごとに横に長い黒 (罫線) を消してから数える．
    """
    x1, x2 = int(box[0]), int(box[1])
    if x2 - x1 < 10:
        return np.zeros(0)
    votes = np.zeros(x2 - x1, dtype=int)
    min_len = max(10, int((x2 - x1) * rule_frac))
    for a, b in bands:
        a, b = max(0, int(a)), min(dark.shape[0], int(b))
        if b - a < 2:
            continue
        band = _strip_long_runs(dark[a:b, x1:x2], min_len)
        votes += band.any(axis=0).astype(int)
    return votes


def valleys(votes, thr):
    """票が `thr` 以下の連続する区間を [(始まり, 終わり)] で返す (終わりは含まない)"""
    v = np.asarray(votes)
    if v.size == 0:
        return []
    flat = (v <= thr).astype(np.int8)
    pad = np.concatenate(([0], flat, [0]))
    d = np.diff(pad)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def choose(votes, min_w=MIN_W, min_w_frac=MIN_W_FRAC,
           lo=RIGHT_LO, hi=RIGHT_HI, thr_max=0):
    """票の並びから，独文と和文のあいだの谷を選んで境の x (領域の中の位置) を返す

    票の閾値は 1 から上げ，条件に合う谷が出た所で止める (146 表中 103 表は 1 で決まる．
    上下の欄外の字が入る表だけ 2〜4 が要る)．条件は「幅が下限以上」かつ
    「右端の相対位置が `lo`〜`hi`」．複数あればいちばん広い谷を採る．
    """
    v = np.asarray(votes)
    n = len(v)
    if n < 10:
        return None
    need = max(min_w, int(n * min_w_frac))
    for thr in range(0, max(1, int(thr_max)) + 1):
        best = None
        for a, b in valleys(v, thr):
            if b - a < need:
                continue
            if not lo <= b / n <= hi:
                continue
            if best is None or (b - a) > (best[1] - best[0]):
                best = (a, b)
        if best is None:
            continue
        a, b = best
        zeros = np.flatnonzero(v[a:b] == 0)
        edge = (a + int(zeros[-1])) if zeros.size else (b - 1)
        return float(max(a, edge - BACK_OFF))
    return None


def split_x(dark, box, bands, **kw):
    """表頭の項目名の領域 `box` = (左, 値の左端) から，独文と和文の境の x を返す

    決められなければ None (呼び出し側は `header_col` の右端をそのまま使う)．
    `bands` は項目行の (上, 下) の並び (対策 A の OCR の箱で作った行)．
    """
    if bands is None or len(bands) < MIN_BANDS:
        return None
    votes = vote_profile(dark, box, bands)
    if votes.size == 0:
        return None
    x = choose(votes, thr_max=int(len(bands) * THR_FRAC), **kw)
    return None if x is None else float(box[0]) + x
