"""帯の中の黒画素を測る．`eval_grid.py` と `label_gaps.py` の両方から使う

正解ラベルに無い帯(余分な帯)が何なのかを，画像から機械的に振り分けるための道具．
    value  : 組成部に値がある → 本当は行．正解ラベルの付け漏れ
    header : 種名の側にだけ字がある → 見出しの行か，折り返した学名の行
    blank  : どこにも字がない → 検出の取りすぎ

**振り分けは目安にすぎない**．見出しの日本語が組成部まで伸びている行は
`value` に出る(2026-09-01 に 127 本を目視して分かった)．
足すかどうかを決めるときは，必ず切り出した画像を見る．
"""
import numpy as np
from PIL import Image

INK_MIN = 0.15      # 同じ段の正解の行の中央値に対する比．これ未満は字が無いとみなす
LINE_RATIO = 0.60   # 横罫線とみなす1画素行(黒画素がこの割合を超える)は外形から外す


def binarize(image):
    """濃淡から白黒を作る(大津の方法．しきい値は画像ごとに決める)

    しきい値は**画像ぜんたいで1回**決める．領域ごとに大津の方法をかけると，
    白紙の領域でも雑音が半分に割れて「インクがある」ことになってしまう．
    引数はファイルの場所でも，開いた画像(PIL)でもよい．
    """
    im = image if isinstance(image, Image.Image) else Image.open(image)
    g = np.asarray(im.convert('L'), dtype=np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(float)
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * np.arange(256))
    with np.errstate(invalid='ignore', divide='ignore'):
        mu0 = m0 / w0
        mu1 = (m0[-1] - m0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    # `w0` は「濃さが t 以下」の数なので，暗い側の組は `g <= t`．
    # `<` にすると1階調ずれる．**白黒2値の画像では致命的**で，
    # 0 と 255 しか無いと t=0 になり，`g < 0` が1つも当たらず
    # 「黒画素なし」になる(2026-09-02 に折り込みのスキャンで分かった)．
    return g <= int(np.nanargmax(var))        # True が黒


def ratio(dark, y1, y2, x1, x2):
    """帯の中の黒画素の割合"""
    y1, y2 = max(0, int(y1)), min(dark.shape[0], int(round(y2)))
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(round(x2)))
    if y2 <= y1 or x2 <= x1:
        return 0.0
    return float(dark[y1:y2, x1:x2].mean())


def thick_ratio(dark, y1, y2, x1, x2, min_run=4):
    """**細い縦線(罫線)を除いた**黒画素の割合

    1画素の行ごとに黒の連なりを見て，幅が `min_run` 未満のものは数えない．
    字の画は太いので残り，罫線は落ちる．
    **傾いた罫線**でも行ごとに見れば細いままなので落とせる
    (列ごとの割合で見ると，傾いた線は 15 画素ほどに散って見分けられない．
     2026-09-01 に kinki_010 で分かった)．
    """
    y1, y2 = max(0, int(y1)), min(dark.shape[0], int(round(y2)))
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(round(x2)))
    sub = dark[y1:y2, x1:x2]
    if sub.size == 0:
        return 0.0
    total = 0
    for row in sub:
        if not row.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(
            ([0], row.view(np.int8), [0]))))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a >= min_run:
                total += b - a
    return total / sub.size


RULE_FILL = 0.8

DASH_RUNS = 6         # 破線の行とみなす，短い連なりの最小本数
DASH_RUN_LEN = (4, 60)  # 短い連なりの長さの範囲(px)
DASH_GAP_MAX = 40     # 連なりどうしの間隔の上限(px)
DASH_COVER = 0.25     # 連なりの並びが幅のこの割合以上に広がっていること


def dashed_rows(dark, min_runs=DASH_RUNS, run_len=DASH_RUN_LEN, gap_max=DASH_GAP_MAX,
                cover=DASH_COVER):
    """破線(種群を囲む点線の枠)の行を見つけて，行ごとの真偽で返す(2026-09-04)

    破線は**短い黒の連なりが一定の間隔で横に並ぶ**．`run_len` の長さの連なりが
    `gap_max` 以下の間隔で `min_runs` 本以上つづき，その並びが幅の `cover` 以上に
    広がっている行を破線とみなす．字の行は連なりの長さも間隔もばらばらで，
    「・」の行は連なりが短すぎる．
    ユーザ提案(2026-09-04)．傾きを直した 08_p2 では破線が「・」の行のちょうど
    中間に来て，黒画素の並びに半分の周期が生まれ，行の高さを 19 px と誤った．
    """
    out = np.zeros(dark.shape[0], dtype=bool)
    w = dark.shape[1]
    lo, hi = run_len
    for i in range(dark.shape[0]):
        row = dark[i]
        if row.sum() < w * 0.03:
            continue
        pad = np.concatenate(([False], row, [False]))
        d = np.diff(pad.astype(np.int8))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        L = e - s
        ok = (L >= lo) & (L <= hi)
        if ok.sum() < min_runs:
            continue
        ss, ee = s[ok], e[ok]
        gaps = ss[1:] - ee[:-1]
        chain = best = 1
        for g in gaps:
            chain = chain + 1 if g <= gap_max else 1
            best = max(best, chain)
        if best >= min_runs and (ee[-1] - ss[0]) >= w * cover:
            out[i] = True
    return out


BOX_LINE_FRAC = 0.6   # セルの幅(高さ)のこの割合以上に続く黒の連なりは枠線とみなす
BOX_EDGE_FRAC = 0.0   # 縦線を消すのは，セルの左右の端からこの割合の範囲だけ(0 なら消さない)


def erase_box_lines(img, frac=BOX_LINE_FRAC):
    """セルの画像から，セルを横切る枠線(横線・縦線)だけを消した写しを返す(2026-09-04)

    種群を囲む枠線がセルに入ると，EasyOCR は `+` を `4`，`・` を `4` と読む
    (example.jpg の 2 セル．ユーザ指摘で枠線の影響を調べた)．
    消すのは**セルの幅(高さ)の `frac` 以上に一続きで走る**黒だけ．
    短い連なり(数字の横棒，`1` の縦棒，破線の断片)は触らない
    (細くて長いかたまりを全部消す案は，単独の `+`・`1` まで消して改悪だった)．
    """
    g = np.asarray(img.convert('L')).copy()
    dark = g < 128
    h, w = dark.shape
    if h == 0 or w == 0:
        return img

    def long_runs(row, min_len):
        pad = np.concatenate(([False], row, [False]))
        d = np.diff(pad.astype(np.int8))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return [(a, b) for a, b in zip(s, e) if b - a >= min_len]

    for y in range(h):
        for a, b in long_runs(dark[y], int(w * frac)):
            g[y, a:b] = 255
    # 縦線は**セルの左右の端 `edge` の範囲にあるものだけ**消す．タイプ打ちの `1` は
    # まっすぐな縦棒で高さの 6 割を超えるが，セルの中ほどにある(08_p2 で
    # `1;2` が `2` になった 7 セル)．枠の縦線は境の近くにしか来ない
    lo, hi = int(w * BOX_EDGE_FRAC), int(w * (1 - BOX_EDGE_FRAC))
    for x in list(range(0, lo)) + list(range(hi, w)):
        for a, b in long_runs(dark[:, x], int(h * frac)):
            g[a:b, x] = 255
    return Image.fromarray(g)         # 縦罫線とみなす，その x の黒画素の割合


def text_ratio(dark, y1, y2, x1, x2, min_run=4, fill=RULE_FILL):
    """**縦罫線も除いた**黒画素の割合

    `thick_ratio()` は横方向の連なりだけを見るので，**縦の罫線**が
    字として残る(幅 4-5 px の縦線は横に 4-5 px 連なるため min_run を通る)．
    表の左端の罫線だけの列が「本体に字がある」と見え，階層の列と
    判定されていた(s01115_14_p5 は階層の無い表なのに 29 セルの階層の列が
    でき，読み取りで全滅した．2026-09-03)．

    高さの `fill` 以上が黒い x は罫線とみなして外してから測る．
    字は縦に連なっても行の間で切れるので，この割合には届かない．

    **横罫線は外さない**．幅の狭い列では，記号の行がそのまま「幅の 8 割が黒」に
    なるので，字ごと消えてしまう (階層の列)．横罫線を除きたい呼び出し側は，
    `row_track.clean_rules()` で消した写しを渡す．
    """
    y1, y2 = max(0, int(y1)), min(dark.shape[0], int(round(y2)))
    x1, x2 = max(0, int(x1)), min(dark.shape[1], int(round(x2)))
    sub = dark[y1:y2, x1:x2]
    if sub.size == 0:
        return 0.0
    keep = sub.mean(axis=0) < fill
    if not keep.any():
        return 0.0
    sub = sub[:, keep]
    total = 0
    for row in sub:
        if not row.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(
            ([0], row.view(np.int8), [0]))))
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a >= min_run:
                total += b - a
    return total / sub.size


def extent(dark, y1, y2, x1, x2, pad=2):
    """帯の中で，字のある y の範囲(外形)を返す．横罫線は数えない"""
    y1i, y2i = max(0, int(y1)), min(dark.shape[0], int(round(y2)))
    x1i, x2i = max(0, int(x1)), min(dark.shape[1], int(round(x2)))
    if y2i <= y1i or x2i <= x1i:
        return None
    prof = dark[y1i:y2i, x1i:x2i].mean(axis=1)
    prof = np.where(prof > LINE_RATIO, 0.0, prof)   # 罫線を落とす
    if prof.max() <= 0:
        return None
    idx = np.flatnonzero(prof >= max(0.01, prof.max() * 0.10))
    if idx.size == 0:
        return None
    return y1i + int(idx[0]) - pad, y1i + int(idx[-1]) + 1 + pad


def classify(dark, band, comp_x, name_x, base):
    """帯を value / header / blank に振り分け，測った値も返す

    base は同じ段の正解の行の黒画素の中央値(画像ごとの濃さの違いを吸収する)．
    """
    lo, hi = band
    ic = ratio(dark, lo, hi, *comp_x) if comp_x else 0.0
    iname = ratio(dark, lo, hi, *name_x) if name_x else 0.0
    if ic >= base * INK_MIN:
        kind = 'value'
    elif iname >= base * INK_MIN:
        kind = 'header'
    else:
        kind = 'blank'
    return kind, ic, iname

# 常在度のセルの頭(ローマ数字)を，画数から数え直すための決まり(2026-09-06)
HEAD_JOIN = 3          # このすき間までは同じ字とみなす(serif の穴)
HEAD_TALLER = 1.1      # 括弧は頭の字より，これだけ背が高い
HEAD_NARROW = 0.85     # 括弧は頭の字より，これだけ細い
HEAD_MIN_H = 8         # これより背が低いものは汚れ
HEAD_MIN_W = 3         # これより細いものは汚れ
HEAD_MAX = 4           # 頭がこれより多く数えられたら，数え違いとみなす
SAME_W = 1.35          # 頭の字どうしの幅の違いが，これを超えたら数えない
SAME_H = 1.20          # 同じく高さ


def head_strokes(dark, box, join=HEAD_JOIN, taller=HEAD_TALLER,
                 narrow=HEAD_NARROW, min_h=HEAD_MIN_H, min_w=HEAD_MIN_W,
                 max_head=HEAD_MAX):
    """`III(+-1)` のようなセルで，**括弧の左にある字の数**を返す

    OCR はローマ数字の画数を数え違える(`II` を `I` と読む)．
    これは**正しい形の値になってしまう**ので，読みだけでは気づけない．
    画数は黒画素から数えられるので，読みと突き合わせられる．

    **括弧は，頭の字より背が高くて細い**(`I` は高さ 16-20・幅 8-12 に対し，
    `(` は高さ 20-26・幅 5-8)．背の高さだけでは見分けられない．
    セルの端に入り込む**枠線**(高さ 41-52)や，開きより背の高い閉じ括弧を
    「いちばん高いもの」と取り違えるため(s01115_11_p1 で 5 セル誤った)．

    **serif 付きの `I` は，上下の横棒に穴があいて 3 本に割れて見える**ので，
    `join` px までのすき間は同じ字としてつなぐ(字と字の間は 5-7 px)．
    背の低いかたまり(汚れ)は数えない．

    Args:
        dark: `binarize()` の結果
        box: (x1, y1, x2, y2)
    Returns:
        頭の字の数．括弧が見つからなければ None
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    sub = dark[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    prof = sub.any(axis=0)
    idx = np.flatnonzero(np.diff(
        np.concatenate(([False], prof, [False])).astype(np.int8)))
    runs = []
    for a, b in zip(idx[0::2], idx[1::2]):
        ys = np.flatnonzero(sub[:, a:b].any(axis=1))
        if len(ys):
            runs.append((int(a), int(b - a), int(ys[-1] - ys[0] + 1)))
    if not runs:
        return None
    merged = []
    for a, w, h in runs:
        if merged and a - (merged[-1][0] + merged[-1][1]) <= join:
            pa, pw, ph = merged[-1]
            merged[-1] = (pa, a + w - pa, max(ph, h))
        else:
            merged.append((a, w, h))
    # 先頭の汚れを飛ばす
    runs = [r for r in merged if r[2] >= min_h and r[1] >= min_w]
    if not runs:
        return None
    _, w0, h0 = runs[0]
    head = []
    for _, w, h in runs:
        if h >= h0 * taller and w <= w0 * narrow:
            if not (1 <= len(head) <= max_head):
                return None
            # **頭の字は同じ形が並ぶ**(`II` は同じ `I` が 2 つ)．
            # 形がそろっていなければ，括弧や汚れを数に入れている
            # (s01115_11_p1 は `(` が `I` より細くない字体で，3 セル誤った)
            ws = [a for a, _ in head]
            hs = [b for _, b in head]
            if max(ws) > min(ws) * SAME_W or max(hs) > min(hs) * SAME_H:
                return None
            return len(head)
        head.append((w, h))
    return None


LETTER_MIN = 0.4      # いちばん大きいかたまりに対する比．これ未満は区切りや汚れ
LETTER_JOIN = 4       # このすき間までは同じ字とみなす


def letter_blobs(dark, min_ratio=LETTER_MIN, join=LETTER_JOIN):
    """横に並ぶ**字**のかたまりを (x1, x2) で返す(2026-09-07)

    区切りの `・` や端の罫線を数に入れないよう，**いちばん大きいものに対する
    比**で選り分ける(`S・K` は S 21 px・`・` 8 px・K 36 px)．
    読みの字数と突き合わせて，**読み落としを見つける**ために使う．
    """
    prof = dark.any(axis=0)
    idx = np.flatnonzero(np.diff(
        np.concatenate(([False], prof, [False])).astype(np.int8)))
    runs = []
    for a, b in zip(idx[0::2], idx[1::2]):
        if runs and a - runs[-1][1] <= join:
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b)])
    if not runs:
        return []
    wide = max(b - a for a, b in runs)
    return [(a, b) for a, b in runs if (b - a) >= wide * min_ratio]
