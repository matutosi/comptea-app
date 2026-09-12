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


def union(maps, names=None, with_source=False):
    """複数の読み手の結果を重ねる (**先に来たものを優先**)

    **読み手ごとに落とすセルが違う**ので，重ねると読めるセルが増えます
    (2026-09-12 に真値で確認: kinki_047 の和名は EasyOCR 43・yomitoku 42 に
    対し**合わせて 45**，学名は 3 つ合わせて 32)．

    Args:
        maps: {cell_id: 文字列} の並び．**信頼する順**に渡す
        names: それぞれの読み手の名前 (`with_source` のときに使う)
        with_source: (結果, {cell_id: 読み手の名前}) を返す
    """
    out, src = {}, {}
    for i, m in enumerate(maps or []):
        name = (names[i] if names and i < len(names) else str(i))
        for cid, v in (m or {}).items():
            if not (v or '').strip():
                continue
            if cid not in out:
                out[cid] = v
                src[cid] = name
    return (out, src) if with_source else out


def disagree(maps):
    """**どの読み手も読めたが，中身が食い違う**セル

    目視に回す手がかりになります (既存の `--reader both` と同じ考え方)．
    """
    seen = {}
    for m in (maps or []):
        for cid, v in (m or {}).items():
            t = (v or '').strip()
            if t:
                seen.setdefault(cid, set()).add(t)
    return {cid for cid, vs in seen.items() if len(vs) > 1}


def best(maps, ok=None, names=None, with_source=False):
    """複数の読みから，**質の通るもの**を選ぶ

    `union` は「先の読み手が何か読めていれば採る」ので，**読めてはいるが
    間違っている**セルを後の読み手が直せません (2026-09-12 に真値で確認:
    一致が EasyOCR 単独と同じ 13・11 のまま．yomitoku 単独は 24・31)．

    Args:
        maps: {cell_id: 文字列} の並び．**信頼する順**に渡す
        ok: `ok(text)` が True なら「通る読み」．辞書に当たるか，
            `correct_text` の補正が通るかを渡す．省くと先勝ち (`union` と同じ)
    Returns:
        {cell_id: 文字列}．`with_source` なら (結果, {cell_id: 読み手の名前})

    決め方は 3 段:
      1. **通る読みのうち，いちばん先の読み手**のもの
      2. 通る読みが無ければ，**読めたもののうち先の読み手**のもの
      3. どれも読めなければ入れない
    """
    cells = set()
    for m in (maps or []):
        cells |= set(m or {})
    out, src = {}, {}
    for cid in cells:
        first = None
        for i, m in enumerate(maps or []):
            t = ((m or {}).get(cid) or '').strip()
            if not t:
                continue
            name = (names[i] if names and i < len(names) else str(i))
            if first is None:
                first = (t, name)
            if ok is not None and ok(t):
                out[cid], src[cid] = t, name
                break
        else:
            if first is not None:
                out[cid], src[cid] = first
    return (out, src) if with_source else out


# **クラスごとに，信頼する読み手の順**を変えます (2026-09-12 に実データで決めた)．
#
# 真値 (人が書き起こした 2 表) との一致数 (kinki_010-1 / kinki_047):
#   学名 sname        EasyOCR 13 / 11   yomitoku 24 / 31
#   和名 species_col  EasyOCR 23 / 42   yomitoku 27 / 41
#
# 真値は 2 表しかないので，**和名の辞書 (27,127 件) にそのまま当たるか**を
# 物差しに，本のページ 6 表 + 折込 6 表 (1,269 セル) で確かめた．
# 誤読が実在の和名に当たることは稀なので，強い代理になる．
#
#   和名 … EasyOCR 277・yomitoku 374 → **EasyOCR が先 428** / yomi 先 420
#   学名 … EasyOCR 462・yomitoku 397 → easy 先 529 / **yomitoku が先 548**
#
# **折込でも同じ傾向**なので，紙面の種類では分けない．
# どちらの組み合わせも**単独の最良を上回る** (和名 374 → 428，学名 462 → 548)．
ORDER = {
    'sname': ('yomi', 'easy', 'ndl'),
}
DEFAULT_ORDER = ('easy', 'yomi', 'ndl')


def order(cls):
    """そのクラスで，信頼する読み手の順"""
    return ORDER.get(cls, DEFAULT_ORDER)


def pick(maps, cls, ok=None, with_source=False):
    """読み手ごとの結果 `{名前: {cell_id: 文字列}}` から，クラスの順で選ぶ

    Args:
        maps: {読み手の名前: {cell_id: 文字列}}
        cls: `obj_name` (sname・species_col・comp…)
        ok: 「通る読み」の判定 (`best` に渡す)
    """
    names = [n for n in order(cls) if n in (maps or {})]
    return best([maps[n] for n in names], ok=ok, names=names,
                with_source=with_source)
