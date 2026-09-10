"""行の種類 (見出し・学名だけの行・凡例) を見分けて note に付ける (案 e の段階 3)

組成表の本体には，種の行のほかに次の 3 つが混ざる．

- **見出し行**: 「Kenn- u. Trennarten d. Ass.:」「群集標徴種および区分種」「Begleiter:」．
  種名の側にだけ字があり，下に長い下線が引かれる (活字では 13/14 に下線があり，
  長さは 136〜560 px)．
- **学名だけの行**: 学名が長くて 2 行に組まれ，和名と値は次の行にあるもの
  (kinki_047 に 9 行)．種名の側にだけ字があるので**見出しと同じ見え方**をするが，
  下線が無く (17〜23 px しかない)，次の行に和名と値がある．
- **凡例・表題**: 「a: Typische Subass. … 典型亜群集」のように列をまたいで文が流れる．
  種の行も 47% は学名が和名の列へ食い込むので「またぐ」だけでは見分けられないが，
  格子の外の帯は 1 つの塊の幅が中央値 15〜19 行ぶんで，中は 8.5 行ぶんと差が出る．

**行は落とさない**．落とすと真値との一致 (行の recall 1.000) が下がる．印だけを
`note` に付け，段階 2 以降 (読み取り・突き合わせ) が使えるようにする．

組成側に字があるかを見るときは，**枠線を消してから**数える．活字では区分種を囲む
枠線が見出しの帯に入り，「組成に字がある」と誤判定した (010・066 の各 2 見出し)．
"""

import re

import numpy as np
import pandas as pd

from . import ink
from .row_track import _pitch_of, clean_rules, unit_spans


NAME_CLASSES = ('sname', 'species_col')


BODY_CLASSES = ('comp', 'layer')


MIN_ROWS = 5            # 段に要る最小の行数


MIN_AREA = 6            # これより黒画素の少ない連なりはごみ


MIN_INK_P = 0.4         # 字があるとみなす黒画素の下限 (行の高さの倍数．大きいほう)．
                        # 枠線を消すと角に数 px の切れ端が残る (横を先に消すので縦が
                        # そこで切れ，短い方が残る)．薄い「・」でも 20〜35 px はある


RULE_FRAC = 0.5         # 帯の幅のこの倍より長く続く黒は罫線・枠線として消す


UNDERLINE = 3.0         # 行の高さのこの倍より長い横の黒は「下線」(見出しの印)


UNDERLINE_MIN = 100     # 同上 (px．学名だけの行の下線は 17〜23 px)


LEGEND_W = 12.0         # 1 つの塊の幅が行の高さのこの倍を超えたら，列をまたぐ文 (凡例)


BODY_FRAC = 0.35        # 本体の黒画素が，種の行の中央値のこの倍に満たなければ「本体は空」


class _Region:
    """段の 1 つの帯 (x の範囲) を，罫線・枠線を消した写しで持つ

    消すのは**段の全高で 1 回**．1 行ぶんの帯だけを見て消すと，表をまたぐ縦の枠線が
    その帯の中では短く見えて残り，「組成に字がある」と誤判定する (活字で区分種を囲む
    枠線が見出しの帯に入った．010・066 の各 2 見出し．2026-09-09)．
    """

    def __init__(self, dark, x1, x2, y1, y2, pitch):
        self.x1, self.x2 = int(x1), int(x2)
        self.y1, self.y2 = max(0, int(y1)), min(dark.shape[0], int(y2))
        self.ok = self.x2 - self.x1 >= 2 and self.y2 - self.y1 >= 2
        self.need = max(MIN_AREA, int(MIN_INK_P * pitch))
        self.clean = (clean_rules(dark, self.x1, self.x2, self.y1, self.y2, pitch)
                      if self.ok else None)

    def ink(self, a, b):
        """`a`〜`b` (画像の y) の黒画素の数"""
        if not self.ok:
            return 0
        a, b = max(self.y1, int(a)) - self.y1, min(self.y2, int(b)) - self.y1
        if b - a < 1:
            return 0
        return int(self.clean[a:b].sum())

    def has_ink(self, a, b):
        """`a`〜`b` (画像の y) に字があるか"""
        return self.ink(a, b) >= self.need


KANA = set('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモ'
           'ヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ'
           'ァィゥェォッャュョーヴヵヶ')

MIN_KANA = 3            # カナがこれだけ続けば和名がある


# **「出現 1 回の種」の見出し** (2026-09-11 ユーザ指示: そのまま書かれている．
# ただし 1 は漢数字とアラビア数字の両方がある)．独文は「Außerdem je einmal in
# Lfd. Nr.」で，OCR は ß を B・s・β と読むことがあるので緩く当てる
ONCE_JA = re.compile(r'出現\s*[1１一壱]\s*回\s*の?\s*種')
ONCE_DE = re.compile(r'(?:Au[\u00dfBbs\u03b2]+erdem|je\s*einmal)', re.I)


def kind_of_text(text):
    """1 行の読みから，その行の役割を決める

    Returns:
        'once' (出現 1 回の種の見出し) / 'item' (表頭の項目名) /
        'species' (種名) / 'other'
    """
    from .correct_text import known_names
    from .plot_table import match_item
    t = (text or '').strip()
    if not t:
        return 'other'
    packed0 = t.replace(' ', '')
    if ONCE_JA.search(t) or ONCE_JA.search(packed0) or ONCE_DE.search(t):
        return 'once'
    if match_item(t) is not None:
        return 'item'
    # 和名は空白で区切られていることがある (「ア カ マ ツ」)．詰めて引く
    packed = t.replace(' ', '')
    for cand in (t, packed):
        if known_names([cand])[0]:
            return 'species'
    # 種の行は「学名 和名 階層 値」が 1 行に並ぶので，カナの**割合**では見ない．
    # カナが 3 字以上**続く**なら和名がある (辞書に無い古い表記・誤読も含む)．
    # 表頭の項目名の行は先に `match_item` で拾っているので，ここへは来ない
    if re.search('[' + ''.join(sorted(KANA)) + ']{%d,}' % MIN_KANA, packed):
        return 'species'
    return 'other'


def read_kind(img, box, reader=None):
    """帯 `box` (x1, y1, x2, y2) を読んで，その行の役割を返す

    **行としても文章としても取れるときの決め手**に使います (2026-09-11 ユーザ指示)．
    黒画素の形だけでは，表の最終行と直下の流し込みの 1 行目が重なると分けられません
    (kinki_079-1 の「ヤマルリソウ」)．読んで，種名の辞書に当たれば表の行です．
    """
    from . import ocr
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 'other'
    reader = ocr.READER if reader is None else reader
    try:
        cell = np.asarray(img.crop((x1, y1, x2, y2)).convert('RGB'))
        text = ' '.join(reader.readtext(cell, detail=0, paragraph=True))
    except Exception:                           # noqa: BLE001  読めなくても進む
        return 'other'
    return kind_of_text(text)


def _runs(row, gap=2):
    """1 行の黒画素の連なりを [(始まり, 終わり)] で返す"""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def has_underline(dark, x1, x2, y1, y2, pitch):
    """帯の下寄りに，長い横の黒 (下線) があるか"""
    x1, x2 = int(x1), int(x2)
    y1, y2 = max(0, int(y1)), min(dark.shape[0], int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    need = max(UNDERLINE_MIN, int(UNDERLINE * pitch))
    for y in range(y1, y2):
        for a, b in _runs(dark[y, x1:x2]):
            if b - a >= need:
                return True
    return False


def widest_run(dark, x1, x2, y1, y2):
    """帯の中で，横につながった黒のいちばん長い幅 (px)"""
    x1, x2 = int(x1), int(x2)
    y1, y2 = max(0, int(y1)), min(dark.shape[0], int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0
    best = 0
    for a, b in _runs(dark[y1:y2, x1:x2].any(axis=0)):
        best = max(best, b - a)
    return int(best)


def crossing_run(dark, x1, x2, y1, y2, into):
    """種名の側から本体の側 ( より右) へまたぐ 1 つの塊の幅 (px)

    「列をまたぐ」だけでは凡例を見分けられない: 種の行も 47% は学名が和名の列へ
    食い込む．タイプ打ちでは学名と和名がつながって 1 つの長い塊になり，幅だけで見ると
    種の行が凡例と判定される (14_p1 で 14 行．2026-09-09)．凡例は種名の側から
    **本体の側まで** 1 つながりで流れるので，またぐ位置で見分ける．
    """
    x1, x2 = int(x1), int(x2)
    y1, y2 = max(0, int(y1)), min(dark.shape[0], int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0
    best = 0
    for a, b in _runs(dark[y1:y2, x1:x2].any(axis=0)):
        if a + x1 <= into <= b + x1:
            best = max(best, b - a)
    return int(best)


def _band_of(g, classes):
    cells = g[g['obj_name'].isin(classes)]
    if cells.empty:
        return None
    return float(cells['x1'].min()), float(cells['x2'].max())


def mark_rows(img, df_loc):
    """行の種類を見分け，`note` に `heading` / `name_only` / `legend` を足す

    Returns:
        (印を付けた格子, 警告のリスト)．付ける所が無ければ元の格子をそのまま返す
    """
    if df_loc is None or len(df_loc) == 0 or 'block' not in df_loc.columns:
        return df_loc, []
    if 'row' not in df_loc.columns or 'obj_name' not in df_loc.columns:
        return df_loc, []
    dark = None
    out = df_loc
    warnings = []
    marks = {}
    for block, g in df_loc.groupby('block', sort=True):
        comp = g[g['obj_name'] == 'comp']
        if comp.empty or comp['row'].nunique() < MIN_ROWS:
            continue
        pitch = _pitch_of(comp)
        if not pitch > 0:
            continue
        if dark is None:
            dark = ink.binarize(img)
        name_x = _band_of(g, NAME_CLASSES)
        body_x = _band_of(g, BODY_CLASSES)
        sname_x = _band_of(g, ('sname',))
        ja_x = _band_of(g, ('species_col',))
        if name_x is None or body_x is None:
            continue
        rows = sorted(g['row'].unique())
        span = {}
        for r in rows:
            cells = g[g['row'] == r]
            span[r] = (float(cells['y1'].min()), float(cells['y2'].max()))
        top = min(v[0] for v in span.values())
        bot = max(v[1] for v in span.values())
        reg_name = _Region(dark, name_x[0], name_x[1], top, bot, pitch)
        reg_body = _Region(dark, body_x[0], body_x[1], top, bot, pitch)
        reg_ja = _Region(dark, ja_x[0], ja_x[1], top, bot, pitch) if ja_x else None
        # 本体が空かどうかは**種の行との比**で見る．見出しの帯には，隣の行を囲む枠や
        # その値の一部が入り込むので，絶対値では「字がある」になってしまう
        # (kinki_010-1 の見出し 3 行は本体の黒が 269〜397．種の行は 1000〜6900)
        body_ink = {r: reg_body.ink(*span[r]) for r in rows}
        pos = sorted(v for v in body_ink.values() if v > 0)
        med = float(pos[len(pos) // 2]) if pos else 0.0
        need_body = max(reg_body.need, BODY_FRAC * med)
        state = {}
        for r in rows:
            y1, y2 = span[r]
            state[r] = (reg_name.has_ink(y1, y2),
                        reg_ja.has_ink(y1, y2) if reg_ja else False,
                        body_ink[r] >= need_body)
        for i, r in enumerate(rows):
            y1, y2 = span[r]
            has_name, _has_ja, has_body = state[r]
            # 凡例は**生の黒画素**で見る (列をまたぐ 1 つの塊は，罫線として消される
            # ことがあるため)．種の行も学名が和名の列へ食い込むので「またぐ」だけでは
            # 見分けられない．幅が行の高さの 12 倍を超えるものだけを凡例とする
            wide = crossing_run(dark, name_x[0], body_x[1], y1, y2,
                                into=body_x[0] + 0.5 * pitch)
            if wide > LEGEND_W * pitch:
                marks[(block, r)] = 'legend'
                continue
            if not has_name or has_body:
                continue
            if sname_x and has_underline(dark, sname_x[0], sname_x[1], y1, y2, pitch):
                marks[(block, r)] = 'heading'
                continue
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            if nxt is not None and state[nxt][1] and state[nxt][2]:
                marks[(block, r)] = 'name_only'      # 次の行に和名と値がある
            else:
                marks[(block, r)] = 'heading'
        n_head = sum(1 for (b, _r), v in marks.items() if b == block and v == 'heading')
        n_only = sum(1 for (b, _r), v in marks.items() if b == block and v == 'name_only')
        n_leg = sum(1 for (b, _r), v in marks.items() if b == block and v == 'legend')
        if n_head or n_only or n_leg:
            warnings.append(
                f'段{block}: 行の種類を見分けた (見出し {n_head}，学名だけの行 {n_only}，'
                f'凡例 {n_leg})．**行は落としていない**．段階2で使う')
    if not marks:
        return df_loc, warnings
    out = df_loc.copy()
    note = out['note'].fillna('').astype(str)
    key = list(zip(out['block'], out['row']))
    add = pd.Series([marks.get(k, '') for k in key], index=out.index)
    out['note'] = [';'.join(p for p in (a, b) if p) for a, b in zip(note, add)]
    return out, warnings
