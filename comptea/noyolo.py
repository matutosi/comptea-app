"""物体検出を使わずに，紙面の各部分を推定する (別案．2026-09-10 ユーザ提案)

**いまの工程では使っていません**．検出器 (YOLO) が落ちる紙面のための控えとして
置いてあります．必要になったら `guess_parts()` を呼びます．

## 背景

いまの工程は YOLO の検出 (`sname`・`species_col`・`layer`・`col`・`row`・`header`)
から領域を決めます．学習データ 121 件がすべて活字で，折込は 0 件なので，折込では
検出が落ちる紙面があります (17_p1 は組成の左 7 列，20_p3 は階層)．

ユーザの案は 3 段です．

1. 列と行を粗く区切る
2. OCR にかける (**精度は問わない**．格子を区分したあとに OCR はやり直す)
3. OCR の文字列から，どの部分に当たるのかを推測する

## 分かっていること (2026-09-10 に 14_p1 などで試した)

- **幾何だけで部分は決まりません**．「1 行にまとめた黒画素の連なりの中央値」は
  組成部 (8〜9 px = 「・」1 つ) と種名 (13〜30 px) を分けますが，それで組成部の
  左端を推定すると代表 8 表で差の中央値 154〜293 px．組成の 3〜5 列ぶんずれます．
- **OCR の文字列で決めると動きます**．ラテン文字の列 → 学名，カタカナの列 → 和名，
  1〜3 文字の数字・記号 → 組成，と判定できました．
- **「・」だけの列は OCR が空を返しますが，空であること自体が組成部の印**です
  (種名なら必ず字がある)．
- サンプルの行は「その列に字がある行」から選びます．空行を読んでも何も分かりません．
- 表題や見出しの行が混じると誤判定します (組成部の列が「ズイナ群集」を拾って和名と
  出た)．`row_kinds.mark_rows` の印で見出し行を避けるのが next step です．

## 要る精度 (2026-09-10 ユーザ指示)

| 段 | 何に使うか | 要る精度 |
|:--|:--|:--|
| 1 回目の行列 | OCR にかける矩形を切る | 数十 px の誤差でよい |
| 部分の推定 | 表頭・組成・学名・和名・階層の範囲 | 1 行・1 列の単位 |
| 2 回目の行列 | 値を読む格子 | 文字を割らない |

1 回目は「どこに何があるか」が分かれば十分です．部分が決まったあと，いまの
`row_heights`・`col_edges` を当て直します．

## 辞書で確かめる (2026-09-10 に足した．案 10・11・12)

文字の型 (`text_kind`) は「それらしさ」しか見ません．既にある辞書と補正の規則を
当てると，「その列・その行が何か」を直接確かめられます．

- **表頭の語彙** (`plot_table.match_item`): 「通し番号」「調査年月日」に当たる行が
  縦に並ぶ範囲 = 表頭．`item_score` と `row_kind`
- **被度の値の型** (`correct_text.correct_comp`): `1・2`・`+`・`r` の形になる
  セルの割合 = 組成部らしさ．`comp_score`．**「・」だけのセルも `status=None`
  (非出現) として通る**ので，空の列も組成として数えられる
- **種名の辞書** (`correct_text.known_names`): 和名の辞書にそのまま載る読みが
  ある行 = 種名．`name_score`

行の役割は `classify_rows` → `split_header` で決めます．表頭の項目名は紙面の
**左の文字列の列の上のほう**に，種名は同じ列の下のほうにあるので，列ごとの
標本では混ざります．行ごとに読んで，項目名の行と種名の行を分けます．

## 実画像で確かめた結果 (2026-09-10．`guess_layout`．検出器の格子との差)

| 表 | 組成の左端 | 表頭の項目行 | 本体の上端 (黒画素) |
|:--|:--|:--|:--|
| 20_p3 | +72 px (1 列以内) | 2 / 19 行 | 1225 (検出器 1186．1 行) |
| 17_p1 | −120 px (1 列以内) | 0 / 12 行 | 1518 (検出器 1506．1 行) |
| 14_p1 | +1393 px | 7 / 13 行 | 1295 (検出器 1268．1 行) |

- **本体の上端は黒画素だけで 3 表とも 1 行以内**に決まる (`body_top_from_ink`)．
  短冊ごとの黒画素の中央値で行の密度を取ると，「・」の行と数字の行が 5〜10 倍
  離れる．OCR に頼らないので，項目名が崩れる紙面 (17_p1) でも効く．
- **表頭の項目行は OCR の質に左右される**．20_p3 は 19 項目のうち 3 つしか
  `match_item` に当たらない (「調査年月日」が「調衣罪月日」)．1 行ずつ切るより
  領域をまとめて読むほうが良いが，それでも足りない．表頭の範囲は本体の上端
  (黒画素) と項目名の最初の行の組で決めるのが現実的．
- **14_p1 の組成の左端**は，段 1 の列の切り方の問題．種名の領域と組成の左の
  列が 1 つの列 (28〜2111 px) にまとまり，その中は読んでいない．

## 残る課題

- **段 1 の列の切り方**．`col_edges.plot_gaps` (95% 白の帯が 20 px 以上) は組成部の
  中は切れますが，種名の領域が切れません (学名と和名が 1 列 2083 px にまとまる)．
  閾値を緩めるか，投影の谷で切る必要があります．
"""
import re

import numpy as np

from . import ink
from .col_edges import plot_gaps

# 行の高さの探索範囲 (300 dpi のスキャン)
PITCH_LO, PITCH_HI = 12, 80

# 列ごとに読むセルの数 (精度は問わないので少なくてよい)
N_SAMPLE = 6

# 型を決める閾値
SHORT_TEXT = 3          # これ以下の字数なら「短い値」(組成・階層)
MIN_LATIN = 3           # ラテン文字がこれ以上なら学名の候補
MIN_KANA = 3            # カタカナがこれ以上なら和名の候補
LAYER_WORDS = ('B1', 'B2', 'B', 'S', 'K', 'K1', 'K2')

KANA = set('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモ'
           'ヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ'
           'ァィゥェォッャュョーヴヵヶ')


def guess_pitch(dark, lo=PITCH_LO, hi=PITCH_HI):
    """紙面全体の黒画素の自己相関から，行の高さを推定する"""
    prof = dark.sum(axis=1).astype(float)
    s = prof - prof.mean()
    a = np.correlate(s, s, mode='full')[len(s) - 1:]
    if not a[0]:
        return lo
    a = a / a[0]
    hi = min(hi, len(a))
    if hi <= lo:
        return lo
    return lo + int(np.argmax(a[lo:hi]))


def rough_rows(dark, pitch, min_ink=0.005):
    """行の高さで刻み，字のある帯の上端を返す (粗い行)"""
    h = dark.shape[0]
    return [y for y in range(0, h - pitch, pitch)
            if dark[y:y + pitch].mean() > min_ink]


def rough_cols(dark, rows, pitch):
    """印字の隙間で列を切る (粗い列)

    **種名の領域は切れません** (字が詰まっているため)．そこは 1 つの広い列として
    返るので，段 3 で「ラテン文字とカタカナが混じる列」として扱います．
    """
    h, w = dark.shape
    y1 = min(rows) if rows else 0
    y2 = (max(rows) + pitch) if rows else h
    gaps = plot_gaps(dark, (0, y1, w, y2))
    edges = [0] + list(gaps) + [w]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > pitch // 2]


def text_kind(texts):
    """読みの並びから，その列の役割を決める

    Returns:
        'sname' / 'species_col' / 'layer' / 'comp' / 'empty' / None
    """
    kept = [t.strip() for t in texts if t and t.strip()]
    if not kept:
        return 'empty'          # 読めない列．組成部の「・」の列であることが多い
    joined = ' '.join(kept)
    n_lat = sum(c.isascii() and c.isalpha() for c in joined)
    n_kana = sum(c in KANA for c in joined)
    med = float(np.median([len(t) for t in kept]))
    # **階層はラテン文字より先に見る** (`B1`・`S`・`K` はラテン文字なので，
    # 先に学名を見ると取られる．2026-09-10)
    if med <= SHORT_TEXT and sum(t in LAYER_WORDS for t in kept) >= max(2, len(kept) // 2):
        return 'layer'
    if n_lat >= max(MIN_LATIN, n_kana * 2):
        return 'sname'
    if n_kana >= MIN_KANA:
        return 'species_col'
    if med <= SHORT_TEXT:
        return 'comp'
    return None


def read_column(img, dark, box, rows, pitch, reader, n=N_SAMPLE, min_ink=0.01):
    """列から数セルを読む (**精度は問わない**)．その列に字がある行から選ぶ"""
    x1, x2 = int(box[0]), int(box[1])
    have = [y for y in rows if dark[y:y + pitch, x1:x2].mean() > min_ink]
    if len(have) < 2:
        have = rows
    if not have:
        return []
    step = max(1, len(have) // n)
    out = []
    for y in have[::step][:n]:
        cell = np.asarray(img.crop((x1, y, x2, y + pitch)).convert('RGB'))
        try:
            out.append(' '.join(reader.readtext(cell, detail=0, paragraph=True)))
        except Exception:                       # noqa: BLE001  読めなくても進む
            out.append('')
    return out


def guess_parts(img, reader=None):
    """紙面の列ごとに役割を推定する

    Args:
        img: PIL の画像
        reader: easyocr の Reader (省略すると `ocr.READER`)
    Returns:
        [{'x1', 'x2', 'kind', 'texts'}, ...] を左から．`pitch` と `rows` も付ける
    """
    if reader is None:
        from . import ocr
        reader = ocr.READER
    dark = ink.binarize(img)
    pitch = guess_pitch(dark)
    rows = rough_rows(dark, pitch)
    cols = rough_cols(dark, rows, pitch)
    out = []
    for x1, x2 in cols:
        if dark[:, x1:x2].mean() < 0.003:
            continue
        texts = read_column(img, dark, (x1, x2), rows, pitch, reader)
        out.append({'x1': x1, 'x2': x2, 'kind': text_kind(texts), 'texts': texts})
    # **読めない列が横に続く範囲は組成部**．「・」だけの列は OCR が空を返すが，
    # 種名なら必ず字がある (2026-09-10 に分かったこと)
    for i, c in enumerate(out):
        if c['kind'] != 'empty':
            continue
        near = [d['kind'] for d in out[max(0, i - 2):i + 3] if d is not c]
        if 'comp' in near or near.count('empty') >= 2:
            c['kind'] = 'comp'
    return {'pitch': pitch, 'rows': rows, 'columns': out}


# ---------------------------------------------------------------------------
# 辞書で確かめる (案 10・11・12)
# ---------------------------------------------------------------------------

ITEM_MIN = 0.5          # 表頭の項目名に当たる読みがこの割合以上なら項目名の列
COMP_MIN = 0.6          # 被度の形になるセルがこの割合以上なら組成の列
HEADER_MAX_FRAC = 0.5   # 表頭を探すのは紙面の上からこの割合まで


def item_score(texts):
    """読みのうち，表頭の項目名の辞書 (`plot_table.match_item`) に当たる割合"""
    from .plot_table import match_item
    kept = [t.strip() for t in texts if t and t.strip()]
    if not kept:
        return 0.0
    return sum(match_item(t) is not None for t in kept) / len(kept)


def comp_score(texts):
    """読みのうち，被度・群度の形 (`correct_text.correct_comp`) になる割合

    空の読み (「・」= 非出現) も通します．種名や項目名は通りません．
    """
    from .correct_text import correct_comp
    kept = [t.strip() if t else '' for t in texts]
    if not kept:
        return 0.0
    ok = 0
    for t in kept:
        try:
            ok += correct_comp(t)['status'] in ('OK', None)
        except Exception:                       # noqa: BLE001  読めない形は数えない
            pass
    return ok / len(kept)


def name_score(texts, dict_path='j_name.txt'):
    """読みのうち，種名の辞書にそのまま載っている割合 (和名が既定)"""
    from .correct_text import known_names
    kept = [t.strip() for t in texts if t and t.strip()]
    if not kept:
        return 0.0
    return sum(known_names(kept, dict_path)) / len(kept)


def row_kind(text):
    """1 行の読みから，その行の役割を決める

    Returns:
        'item' (表頭の項目名) / 'species' (種名) / 'other'
    """
    from .correct_text import known_names
    from .plot_table import match_item
    t = (text or '').strip()
    if not t:
        return 'other'
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


def classify_rows(img, dark, rows, pitch, box, reader, max_frac=HEADER_MAX_FRAC,
                  min_ink=0.005):
    """列 `box` の上のほう (紙面の `max_frac` まで) を**1 回で**読み，行ごとに役割を付ける

    1 行ずつ切って読むと誤読が増えます (20_p3 の「調査年月日」が「調衣罪月日」．
    19 項目のうち 3 つしか当たらなかった)．領域をまとめて読み，OCR が返す箱の
    y の中心で行に割り当てます．

    Returns:
        [(y, kind, text), ...] を y の順に．字の無い行は入らない
    """
    x1, x2 = int(box[0]), int(box[1])
    y_top = int(min(rows)) if rows else 0
    y_lim = int(dark.shape[0] * max_frac)
    if x2 - x1 < 4 or y_lim - y_top < pitch:
        return []
    region = np.asarray(img.crop((x1, y_top, x2, y_lim)).convert('RGB'))
    try:
        found = reader.readtext(region, detail=1, paragraph=False)
    except Exception:                           # noqa: BLE001
        return []
    by_row = {}
    for pts, text, _conf in found:
        ys = [pt[1] for pt in pts]
        yc = y_top + (min(ys) + max(ys)) / 2.0
        y = y_top + int((yc - y_top) // pitch) * pitch
        if dark[y:y + pitch, x1:x2].mean() <= min_ink:
            continue
        by_row.setdefault(y, []).append(((min(pt[0] for pt in pts)), text))
    out = []
    for y in sorted(by_row):
        text = ' '.join(t for _x, t in sorted(by_row[y]))
        out.append((y, row_kind(text), text))
    return out


def split_header(kinds, pitch):
    """行の役割の並びから，表頭の行と本体の上端を決める

    本体の上端 = **最初の項目名の行より後**にある最初の種名の行．表題や群落名の
    行はカタカナが多く「種名」と読めてしまうので，項目名より前の種名は数えない
    (20_p3 の「ネザサーススキ群集」)．種名の行が無ければ，最後の項目名の行の
    次を本体の上端とする．

    Args:
        kinds: `classify_rows` の返り値
    Returns:
        (表頭の行の y のリスト, 本体の上端の y)．表頭が無ければ ([], None)
    """
    first_item = next((y for y, k, _ in kinds if k == 'item'), None)
    if first_item is None:
        return [], None
    first_sp = next((y for y, k, _ in kinds if k == 'species' and y > first_item), None)
    items = [y for y, k, _ in kinds
             if k == 'item' and (first_sp is None or y < first_sp)]
    body_top = first_sp if first_sp is not None else items[-1] + pitch
    return items, body_top


def column_fill(dark, box, rows, pitch, min_ink=0.001):
    """列 `box` で字のある行の割合 (組成の列は「・」が全行にあるので高い)

    閾値は低く取る．「・」1 つは 9 px² ほどで，90 x 35 のセルの 0.3% しかない
    (0.01 にすると組成の列が全部「空」になり，14_p1 の組成の左端が 4,000 px ずれた)．
    """
    x1, x2 = int(box[0]), int(box[1])
    if not rows:
        return 0.0
    return float(np.mean([dark[y:y + pitch, x1:x2].mean() > min_ink for y in rows]))


COMP_FILL_MIN = 0.5     # 組成の列は行のこの割合以上に字 (「・」) がある
WIDE_COL = 3.0          # 列の幅の中央値のこの倍以上なら「つながった列」
COMP_RUN_MISS = 2       # 右から組成の列を辿るとき，組成でない列をこの数まで飛ばす


def _n_name_like(texts):
    """読みのうち，名前らしい (カナかラテン文字が 3 つ以上) ものの数"""
    n = 0
    for t in texts:
        t = (t or '').strip()
        n_lat = sum(c.isascii() and c.isalpha() for c in t)
        n_kana = sum(c in KANA for c in t)
        n += (n_lat >= MIN_LATIN or n_kana >= MIN_KANA)
    return n


BODY_INK_DROP = 0.35    # 表頭の値の行に対し，本体の行の黒画素はこの割合未満
BODY_WIN = 5            # 本体と判定するのに要る，続けて薄い行の数


def body_top_from_ink(dark, comp_box, rows, pitch, max_frac=HEADER_MAX_FRAC,
                      drop=BODY_INK_DROP, win=BODY_WIN):
    """組成部の帯の黒画素の密度から，本体の上端を決める (OCR を使わない)

    表頭の値の行は数字が全列に並ぶので黒画素が多く，本体の行は「・」が主なので
    少ない．上から見て，`win` 行続けて表頭の水準の `drop` 倍を下回った最初の行を
    本体の上端とする．項目名の OCR が崩れる紙面 (17_p1) の控え．

    Returns:
        本体の上端の y．決められなければ None
    """
    x1, x2 = int(comp_box[0]), int(comp_box[1])
    ys = [y for y in rows if y <= dark.shape[0] * max_frac]
    if len(ys) < win * 2 or x2 - x1 < 4:
        return None
    # 帯を列の幅ほどの短冊に切り，**短冊ごとの黒画素の中央値**を行の密度にする．
    # 平均だと，列の多い表 (17_p1 は 128 列) では本体の行も濃く見える．中央値なら
    # 「ほとんどの列が「・」」の行は「・」1 つぶんの薄さになり，表頭の値の行
    # (どの列にも数字) と 10 倍ほど離れる
    step = max(4, int(pitch * 2))
    xs = list(range(x1, x2 - step + 1, step)) or [x1]
    dens = np.array([np.median([dark[y:y + pitch, a:a + step].mean() for a in xs])
                     for y in ys], dtype=float)
    # 表頭の水準 = 3 行の移動中央値の最大 (分位で取ると，本体の行が多い紙面では
    # 本体の水準になってしまい，境が見つからない．17_p1)
    roll = np.array([np.median(dens[max(0, i - 1):i + 2]) for i in range(len(dens))])
    top = float(roll.max())
    if top <= 0:
        return None
    peak = int(np.argmax(roll >= top * 0.8))    # 表頭の値の行が始まる所
    for i in range(peak, len(ys) - win + 1):
        if (dens[i:i + win] < top * drop).all():
            return ys[i]
    return None


def guess_layout(img, reader=None):
    """`guess_parts` に，辞書で確かめた列の役割と，表頭の範囲を足したもの

    Returns:
        `guess_parts` の返り値に `header_rows`・`body_top`・`name_box`・
        `comp_left` を足す．各列には `comp`・`item`・`name`・`fill` の得点を付ける
    """
    if reader is None:
        from . import ocr
        reader = ocr.READER
    parts = guess_parts(img, reader)
    dark = ink.binarize(img)
    pitch, rows = parts['pitch'], parts['rows']
    widths = sorted(c['x2'] - c['x1'] for c in parts['columns'])
    w_med = float(np.median(widths[:-2])) if len(widths) > 4 else float(np.median(widths))
    for c in parts['columns']:
        c['comp'] = comp_score(c['texts'])
        c['item'] = item_score(c['texts'])
        c['name'] = name_score(c['texts'])
        c['fill'] = column_fill(dark, (c['x1'], c['x2']), rows, pitch)
        n_name = _n_name_like(c['texts'])
        wide = (c['x2'] - c['x1']) >= WIDE_COL * w_med
        # **被度の形になる列は組成**．文字の型より辞書を信じる (17_p1 は組成の
        # 左 7 列が検出から落ちるが，値の形は他の列と同じ)．ただし
        # (a) 名前らしい読みがあれば組成ではない．種名の列の右端の切れ端は，
        #     読みの多くが空で被度の形に見えるが，学名の断片が 1 つ混じる (14_p1)．
        #     例外は**幅の広い列**: 組成の列が隙間で切れずに 20 列ぶんつながった
        #     ものは，見出し行 (「ズイナ群集」) の 1 つだけなら許す
        # (b) 字のある行が半分未満なら組成ではない (「・」が無い)
        if (c['comp'] >= COMP_MIN and c['fill'] >= COMP_FILL_MIN
                and (n_name == 0 or (wide and n_name <= 1))):
            c['kind'] = 'comp'
        elif c['kind'] == 'comp' and (c['fill'] < COMP_FILL_MIN or n_name > 0):
            c['kind'] = None
    # 組成部 = 右端から続く組成の列．その左端が組成の左端．
    # 途中の 2 列までは組成でなくても続きとみなす (表頭の値が混じった標本や
    # 細い切れ端で途切れる．20_p3・14_p1)
    comp_left = None
    miss = 0
    for c in reversed(parts['columns']):
        if c['kind'] == 'comp':
            comp_left = c['x1']
            miss = 0
        elif comp_left is not None:
            miss += 1
            if miss > COMP_RUN_MISS:
                break
    parts['comp_left'] = comp_left
    # 表頭は，**組成の左にある文字の領域**をまとめて読んで決める (1 列に絞ると
    # 細い切れ端を選ぶことがある)
    texts_cols = [c for c in parts['columns']
                  if c['kind'] in ('sname', 'species_col')
                  and (comp_left is None or c['x2'] <= comp_left)]
    parts['header_rows'], parts['body_top'], parts['name_box'] = [], None, None
    if texts_cols:
        box = (min(c['x1'] for c in texts_cols),
               comp_left if comp_left is not None else max(c['x2'] for c in texts_cols))
        kinds = classify_rows(img, dark, rows, pitch, box, reader)
        parts['header_rows'], parts['body_top'] = split_header(kinds, pitch)
        parts['name_box'] = box
        parts['row_kinds'] = kinds
    # OCR で決まらなければ，組成部の黒画素の密度で本体の上端を決める
    parts['body_top_ink'] = None
    if comp_left is not None:
        right = max(c['x2'] for c in parts['columns'] if c['kind'] == 'comp')
        parts['body_top_ink'] = body_top_from_ink(dark, (comp_left, right), rows, pitch)
        if parts['body_top'] is None:
            parts['body_top'] = parts['body_top_ink']
    return parts
