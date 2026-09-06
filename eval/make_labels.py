"""組み上がった格子から，labelme の教師データを作る

新しい資料(s01115 の折り込み)を学習に加えるには，ラベルが要る．
手で付け直すのは 40 枚ぶんあって現実的でないので，**パイプラインが
組んだ格子をラベルに書き戻す**(自己教示)．

ただし**格子が正しい表だけ**を使う．`run_pipeline.py` の警告のうち
「要確認」が付いた表は採らない．黙って誤りを学習させないための歯止め．

書き出すのは**ページ大の断片**(既定 2237 x 3300)．学習は長辺 3300 px の
画像を `imgsz=1280` で行っており，折り込みの表(最大 5137 x 11025)を
そのまま入れると字が潰れて何も学べない．断片の切れ目は

    横  種名の列 + 地点のかたまり(`strips.plan_strips()` と同じ考え)
    縦  行の境目(行を途中で割らない)

に置く．`comp` は検出クラスではないのでラベルにしない
(`locate.py` が col x row から計算で作る)．
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comptea import strips                               # noqa: E402

Image.MAX_IMAGE_PIXELS = None

TILE_W = 2237           # 断片の幅の目安(学習画像の幅の中央値)
TILE_H = 3300           # 断片の高さの目安(学習画像の長辺の中央値)
MIN_INSIDE = 0.6        # 箱がこの割合より外に出たら，その断片には書かない
MIN_BOX_PX = 6          # これより小さい箱は書かない

# 格子のセルから region の箱を組み直す規則
#   縦に長い列   … その名前のセル全部の外形
#   横に長い帯   … 行(`row`)ごとの外形
COLUMN_CLASSES = {'sname': 'sname', 'species_col': 'species_col',
                  'layer': 'layer', 'header_item': 'header_col'}
BODY_ROW = 'comp'       # 行の帯を作るもと
HEADER_ROW = 'header_value'     # 表頭の項目行のもと


def region_boxes(df_loc, img=None):
    """格子のセルから，検出クラスの箱を組み直す

    `img` を渡すと，**見出しの行と折り返した種名の行に `row` を付けない**．
    元の教師データの決まりに合わせるため(そちらは組成部に値のある行だけを
    行として数えている．row precision が 0.85 に留まるのはこのため)．
    振り分けは `ink.classify()` で，`label_gaps.py` と同じ物差しを使う．

    Returns:
        [(label, x1, y1, x2, y2), ...]
    """
    out = []
    comp = df_loc[df_loc['obj_name'] == BODY_ROW]
    head = df_loc[df_loc['obj_name'] == HEADER_ROW]

    # 縦に長い列(種名・和名・階層・表頭の項目名)
    for name, label in COLUMN_CLASSES.items():
        g = df_loc[df_loc['obj_name'] == name]
        if not g.empty:
            out.append((label, g['x1'].min(), g['y1'].min(),
                        g['x2'].max(), g['y2'].max()))

    # 地点の列
    for _, g in comp.groupby('col'):
        out.append(('col', g['x1'].min(), g['y1'].min(),
                    g['x2'].max(), g['y2'].max()))

    # 組成部の行．種名の列の左端から組成部の右端まで
    if not comp.empty:
        left = min([comp['x1'].min()] +
                   [df_loc.loc[df_loc['obj_name'] == n, 'x1'].min()
                    for n in COLUMN_CLASSES
                    if (df_loc['obj_name'] == n).any()])
        right = comp['x2'].max()
        skip = _heading_rows(comp, df_loc, img)
        for row, g in comp.groupby('row'):
            if row in skip:
                continue
            out.append(('row', left, g['y1'].min(), right, g['y2'].max()))
        out.append(('table', left, comp['y1'].min(), right, comp['y2'].max()))

    # 表頭の項目行と，表頭そのもの
    if not head.empty:
        left = head['x1'].min()
        item = df_loc[df_loc['obj_name'] == 'header_item']
        if not item.empty:
            left = min(left, item['x1'].min())
        right = head['x2'].max()
        for _, g in head.groupby('row'):
            out.append(('plot_row', left, g['y1'].min(), right, g['y2'].max()))
        out.append(('header', left, head['y1'].min(),
                    right, head['y2'].max()))
    return [(lab, float(a), float(b), float(c), float(d))
            for lab, a, b, c, d in out]


def _heading_rows(comp, df_loc, img):
    """組成部に値の無い行(見出し・折り返した種名)の行番号を返す"""
    if img is None or comp.empty:
        return set()
    from comptea import ink
    dark = ink.binarize(img)
    cx = (int(comp['x1'].min()), int(comp['x2'].max()))
    name = df_loc[df_loc['obj_name'].isin(('sname', 'species_col'))]
    nx = ((int(name['x1'].min()), int(name['x2'].max()))
          if not name.empty else None)
    bands = comp.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
    base = float(np.median([ink.ratio(dark, r.y1, r.y2, *cx)
                            for r in bands.itertuples()]))
    if not base:
        return set()
    out = set()
    for row, r in bands.iterrows():
        kind, _, _ = ink.classify(dark, (r['y1'], r['y2']), cx, nx, base)
        if kind != 'value':
            out.add(row)
    return out


def _row_edges(df_loc):
    """行の境目(縦に切ってよい y)"""
    comp = df_loc[df_loc['obj_name'] == BODY_ROW]
    if comp.empty:
        return []
    g = comp.groupby('row').agg(y1=('y1', 'min'), y2=('y2', 'max'))
    g = g.sort_values('y1')
    return [float(v) for v in g['y2'].tolist()]


def plan_tiles(df_loc, image_size, tile_w=TILE_W, tile_h=TILE_H):
    """断片の (x1, y1, x2, y2) を返す

    横は種名の列を頭に付けたかたまり，縦は行の境目で切る．
    """
    w, h = image_size
    comp = df_loc[df_loc['obj_name'] == BODY_ROW]
    if comp.empty:
        return []
    body_x1, body_x2 = float(comp['x1'].min()), float(comp['x2'].max())
    names = [df_loc.loc[df_loc['obj_name'] == n, 'x1'].min()
             for n in COLUMN_CLASSES if (df_loc['obj_name'] == n).any()]
    table_x1 = float(min(names)) if names else body_x1
    if table_x1 >= body_x1:
        # 種名の列が組成部より右にある．格子の取り方が壊れているので，
        # この表からは断片を作らない(学ばせない)
        return []

    # 横: 地点のかたまり(重なりは付けない．学習では重複が偏りになる)
    cols = comp.groupby('col').agg(x1=('x1', 'min'), x2=('x2', 'max'))
    cols = cols.sort_values('x1')
    name_w = body_x1 - table_x1
    room = max(tile_w - name_w, cols['x2'].sub(cols['x1']).median() * 3)
    spans, start = [], body_x1
    for _, r in cols.iterrows():
        if r['x2'] - start > room and r['x1'] > start:
            spans.append((start, r['x1']))
            start = r['x1']
    spans.append((start, body_x2))

    # 縦: 行の境目
    # 格子の箱は画像の外へわずかに出ていることがある．**画像の中に収める**
    # (はみ出したまま切ると，端に黒い帯が付き，注釈も画像の外を指す)
    edges = [min(e, h) for e in _row_edges(df_loc)]
    top = max(0.0, float(df_loc['y1'].min()))
    bottom = min(float(df_loc['y2'].max()), float(h))
    bands, s = [], top
    for e in edges:
        if e - s >= tile_h:
            bands.append((s, e))
            s = e
    bands.append((s, bottom))

    return [(table_x1, y1, x2, y2, x1) for x1, x2 in spans
            for y1, y2 in bands]


def clip_boxes(boxes, tile, name_x2, min_inside=MIN_INSIDE):
    """断片の中に入る箱だけを，断片の座標に直して返す

    `tile` は (table_x1, y1, x2, y2, chunk_x1)．種名の側と地点のかたまりを
    横につないだ画像になるので，x は `strips._back()` の逆をたどる．
    """
    tx1, ty1, cx2, ty2, cx1 = tile
    name_w = name_x2 - tx1
    out = []
    for label, x1, y1, x2, y2 in boxes:
        nx = []
        for v in (x1, x2):
            if v < name_x2:
                nx.append(min(max(v, tx1), name_x2) - tx1)
            else:
                nx.append(name_w + min(max(v, cx1), cx2) - cx1)
        a, b = nx
        c, d = max(y1, ty1) - ty1, min(y2, ty2) - ty1
        if b - a < MIN_BOX_PX or d - c < MIN_BOX_PX:
            continue
        # **伸びている向きは切り詰めてよい**．行は表の幅いっぱい，列は表の
        # 高さいっぱいに広がるので，ページ大の断片に丸ごとは収まらない．
        # 断片の中では切り詰めた姿が正しい(学習画像にも，隣のページから
        # 続く行や列は同じ形で入っている)．
        # 見るのは**伸びていない向き**．そちらがはみ出していたら，
        # その箱は隣の断片のもの．
        if label in strips.FULL_WIDTH:
            if (d - c) / max(1.0, y2 - y1) < min_inside:
                continue
        elif (b - a) / max(1.0, x2 - x1) < min_inside:
            continue
        out.append((label, a, c, b, d))
    return out


def write_tile(im, tile, boxes, dst_png, name_x2, labelme=True):
    """断片の画像を書く．`labelme` が真なら labelme の JSON も並べて書く

    YOLO のデータセットを組むときは JSON を書かない．画像の置き場に
    別の拡張子が混ざると，何枚あるのか数えられなくなる．
    """
    tx1, ty1, cx2, ty2, cx1 = [int(v) for v in tile]
    strip = strips.make_strip(im.crop((0, ty1, im.width, ty2)),
                                  tx1, int(name_x2), cx1, cx2)
    strip.save(dst_png)
    if not labelme:
        return strip.size, len(boxes)
    shapes = [{'label': lab, 'points': [[x1, y1], [x2, y2]], 'group_id': None,
               'description': '', 'shape_type': 'rectangle', 'flags': {},
               'mask': None}
              for lab, x1, y1, x2, y2 in boxes]
    doc = {'version': '5.4.1', 'flags': {}, 'shapes': shapes,
           'imagePath': os.path.basename(dst_png), 'imageData': None,
           'imageHeight': strip.height, 'imageWidth': strip.width}
    with open(os.path.splitext(dst_png)[0] + '.json', 'w',
              encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return strip.size, len(shapes)


def write_yolo(boxes, size, dst_txt, names):
    """YOLO の注釈(クラス 中心x 中心y 幅 高さ．いずれも 0-1)を書く"""
    w, h = size
    idx = {n: i for i, n in names.items()}
    lines = []
    for label, x1, y1, x2, y2 in boxes:
        if label not in idx:
            continue
        # **0-1 に収める**．格子の箱は画像の外へわずかに出ていることがあり，
        # 1.000233 のような値が1つでもあると ultralytics はその画像を
        # まるごと捨てる(130 枚のうち 66 枚が消えていた．2026-09-02)
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        # 6 桁に丸めると **5e-7 だけ 1 を超える**ことがあり，それだけで
        # 画像がまるごと捨てられる．先に幅を 2e-6 詰めておく(1画素の
        # 1000 分の1にもならないので，学習には影響しない)
        bw, bh = max(1e-6, bw - 2e-6), max(1e-6, bh - 2e-6)
        cx = min(max((x1 + x2) / 2 / w, bw / 2), 1 - bw / 2)
        cy = min(max((y1 + y2) / 2 / h, bh / 2), 1 - bh / 2)
        lines.append(f'{idx[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
    with open(dst_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    return len(lines)


def make(workdir, image, outdir, tile_w=TILE_W, tile_h=TILE_H):
    """1つの表から断片の教師データを作る．書いた枚数を返す"""
    df_loc = pd.read_csv(os.path.join(workdir, 'located.csv'))
    im = Image.open(image)
    boxes = region_boxes(df_loc)
    tiles = plan_tiles(df_loc, im.size, tile_w, tile_h)
    comp = df_loc[df_loc['obj_name'] == 'comp']
    name_x2 = float(comp['x1'].min()) if not comp.empty else 0.0
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.basename(os.path.normpath(workdir))
    n = 0
    for i, tile in enumerate(tiles, 1):
        got = clip_boxes(boxes, tile, name_x2)
        if not any(lab == 'row' for lab, *_ in got):
            continue                            # 行の無い断片は学ばせない
        dst = os.path.join(outdir, f'{stem}_x{i:02d}.png')
        size, k = write_tile(im, tile, got, dst, name_x2)
        print(f'  {os.path.basename(dst)}  {size[0]} x {size[1]}  箱 {k}')
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description='格子から labelme の教師データを作る')
    p.add_argument('workdir', help='run_pipeline.py が作った作業ディレクトリ')
    p.add_argument('image', help='もとの表の画像')
    p.add_argument('--outdir', default='labelme_new', help='書き出し先')
    p.add_argument('--tile-w', type=int, default=TILE_W)
    p.add_argument('--tile-h', type=int, default=TILE_H)
    a = p.parse_args()
    n = make(a.workdir, a.image, a.outdir, a.tile_w, a.tile_h)
    print(f'書いた断片: {n} 枚 -> {a.outdir}')


if __name__ == '__main__':
    main()
