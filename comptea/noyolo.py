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

## 残る課題

- **段 1 の列の切り方**．`col_edges.plot_gaps` (95% 白の帯が 20 px 以上) は組成部の
  中は切れますが，種名の領域が切れません (学名と和名が 1 列 2083 px にまとまる)．
  閾値を緩めるか，投影の谷で切る必要があります．
"""
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
