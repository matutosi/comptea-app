"""大きな紙面を分割して読み，元の座標へ戻す (tiles.py)

**折込 (A0 級) をそのまま読み手に渡すと，縮小されて読みが崩れます**．
s01115_16_p2 の紙面全体を yomitoku に渡すと「"ry AND 1000」のようになりましたが，
**切り出した注記画像 (3757x1612) は 4.8 秒できれいに読めました** (2026-09-13)．
EasyOCR も長辺を 2560 px に縮めるので，`table_find` は 2400 px のタイルに
分けて読んでいます．

そこで **A3〜A4 ほどに分けて読み，切り出しの原点を足して元の座標へ戻します**
(2026-09-13 ユーザ提案)．**重なった読みは確信度の大きい方**を採ります．

    from comptea import tiles
    got = tiles.read_tiled(im, reader)      # [((x1, y1, x2, y2), 文字列, 確信度)]

**注意**: 同じ「分けて統合する」を，表の**箱を数える**のに使うと悪くなります
(DocLayout-YOLO を 3000 px に分けたら，表の数が合う折込が 10 → 8 枚になった)．
タイルの境で割れた箱のまとめ方が紙面ごとに合わないためです．
**中身を読むのには向きます** — 段落やセルは独立しているので，重なりを除いて
並べるだけで済みます．
"""

SIZE = 3000             # タイルの一辺 (px)．300 dpi の A4 が 2480x3508
OVERLAP = 0.2           # 隣との重なり (一辺に対する割合)．端の語が切れるのを防ぐ
LEAST = 0.3             # 同じ読みとみなす重なり (小さい方の面積に対する割合)


def split(page, size=SIZE, overlap=OVERLAP):
    """紙面 `page` = (幅, 高さ) を，重なり付きのタイルに分ける

    Returns:
        [(x1, y1, x2, y2)]．紙面が小さければ 1 枚のまま
    """
    size_px = int(size)
    w, h = int(page[0]), int(page[1])
    if w <= size_px and h <= size_px:
        return [(0, 0, w, h)]
    step = max(1, int(size_px * (1.0 - overlap)))

    def starts(total):
        if total <= size_px:
            return [0]
        out = list(range(0, total - size_px + 1, step))
        if out[-1] + size_px < total:
            out.append(total - size_px)         # 端は紙面から逆算して収める
        return out

    return [(x, y, min(x + size_px, w), min(y + size_px, h))
            for x in starts(w) for y in starts(h)]


def shift(found, origin):
    """読みを，切り出しの原点ぶんだけずらして元の座標に戻す"""
    ox, oy = float(origin[0]), float(origin[1])
    out = []
    for item in (found or []):
        b, t = item[0], item[1]
        c = item[2] if len(item) > 2 else 1.0
        out.append(((b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy), t, c))
    return out


def _overlap(a, b):
    """2 つの箱の重なり / 小さい方の面積"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def merge(found, least=LEAST):
    """重なった読みを 1 つにまとめる (**確信度の大きい方**を採る)

    タイルの重なりで同じ語が 2 度読まれる．読みが違うこともあるので
    (「アカマツ」と「アカマシ」)，確信度で選ぶ．
    """
    items = sorted((list(x) + [1.0])[:3] for x in (found or []))
    items.sort(key=lambda x: -float(x[2]))
    kept = []
    for b, t, c in items:
        if any(_overlap(b, kb) >= least for kb, _kt, _kc in kept):
            continue
        kept.append((tuple(b), t, float(c)))
    return sorted(kept, key=lambda x: (x[0][1], x[0][0]))


def read_tiled(img, reader, size=SIZE, overlap=OVERLAP, least=LEAST):
    """紙面をタイルに分けて読み，**元の座標**で 1 つにまとめて返す

    Args:
        reader: `read_boxes(img, box)` を持つもの (`comptea.yomi`・`comptea.ndl`)
    """
    out = []
    for box in split(img.size, size=size, overlap=overlap):
        try:
            got = reader.read_boxes(img, box)
        except Exception:                       # noqa: BLE001  読めなくても進む
            continue
        out += shift(got, (box[0], box[1]))
    return merge(out, least=least)
