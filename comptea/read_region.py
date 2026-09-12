"""領域を 1 回読んで，読みをセルに割り当てる (read_region.py)

段階 2 はセルを 1 つずつ読んでいます (`ocr.ocr_images_df`)．**表頭のように
まとまった領域**は，1 回読んで**読みの中心がどのセルに入るか**で割り当てた方が
速く，読みも良くなります．

  - 呼び出し回数が桁で減る (表頭の項目は 10〜20 セルある)
  - 新しい読み手 (ndlocr-lite・yomitoku) は**位置つき**で返すので，そのまま使える
  - 切り出しが狭いと読めない字がある (前後の字が手がかりになる)．
    実測 (2026-09-12): s01115_16_p2 の表頭で，EasyOCR がセルごとに読んで項目名
    5 個に対し，**ndlocr-lite が領域を 1 回読んで 10 個すべて** (1.6 秒)

読み手は `read_boxes(img, box) -> [((x1, y1, x2, y2), 文字列), ...]` を持つもの
(画像の座標で返す)．`comptea.ndl.NdlReader` がその形です．
"""

PAD = 8                 # 領域の箱に足す余白 (px)．端の字が切れるのを防ぐ


def region_box(cells, pad=PAD):
    """セルをすべて囲む箱 (x1, y1, x2, y2)．セルが無ければ None"""
    if cells is None or len(cells) == 0:
        return None
    return (max(0, int(cells['x1'].min()) - pad),
            max(0, int(cells['y1'].min()) - pad),
            int(cells['x2'].max()) + pad, int(cells['y2'].max()) + pad)


def assign(cells, found, split=True):
    """読み `found` をセルへ割り当てる

    Args:
        cells: `cell_id` と `x1,y1,x2,y2` を持つ格子
        found: [((x1, y1, x2, y2), 文字列)]．画像の座標
        split: **1 行の読みを，文字の位置でセルへ分ける**．
            NDLOCR-Lite は行 (テキストライン) 単位で返すので，表頭の値のように
            小さな値が格子状に並ぶ所では 1 行にまとまる．中心だけで割り当てると
            1 セルにしか入らない (実測: 3883 セル中 904)．
            字幅は等しいとみなし，文字の中心が入るセルへ配る．
    Returns:
        {cell_id: 文字列}．読みの無いセルは入らない．
        1 つのセルに複数の読みが入るときは**左から順に空白でつなぐ**
        (「調査地」と「(県名)」が別の読みになる)
    """
    hit = {}
    for idx, (box, text) in enumerate(found or []):
        t = (text or '').strip()
        if not t:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        cy = (y1 + y2) / 2.0
        row = [c for c in cells.itertuples() if c.y1 <= cy <= c.y2]
        if not row:
            continue
        if split and len(t) > 1:
            w = max(1e-6, x2 - x1) / len(t)
            for i, ch in enumerate(t):
                cx = x1 + w * (i + 0.5)
                for c in row:
                    if c.x1 <= cx <= c.x2:
                        hit.setdefault(c.cell_id, []).append((cx, ch, idx))
                        break
        else:
            cx = (x1 + x2) / 2.0
            for c in row:
                if c.x1 <= cx <= c.x2:
                    hit.setdefault(c.cell_id, []).append((cx, t, idx))
                    break
    out = {}
    for k, v in hit.items():
        # **同じ読みから来た文字はつなぎ，別の読みは空白で分ける**
        groups, last = [], None
        for _x, t, idx in sorted(v):
            if idx != last:
                groups.append([])
                last = idx
            groups[-1].append(t)
        joined = ' '.join(''.join(g) for g in groups).strip()
        if joined:
            out[k] = joined
    return out


def read_cells(img, cells, reader, pad=PAD):
    """セルを囲む領域を**1 回だけ**読んで，読みを割り当てる

    読み手が落ちても止めません (空を返す)．読みは補助で，無ければ従来どおり
    セルごとの読みに任せます．
    """
    box = region_box(cells, pad=pad)
    if box is None:
        return {}
    try:
        found = reader.read_boxes(img, box)
    except Exception:                           # noqa: BLE001  読めなくても進む
        return {}
    return assign(cells, found)
