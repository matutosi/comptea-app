"""正解ラベルに付け漏れている `row` の候補を，画像の黒画素から作る

    py -3.12 label_gaps.py [--split train val] [--out work/label_gaps]
    py -3.12 label_gaps.py --apply work/label_gaps/candidates.csv

やっていること
    1. eval_grid と同じ手順で格子を作り，正解と突き合わせて「余分な帯」を出す
    2. 帯の中の黒画素を測り，中身で3つに振り分ける
         value  : 組成部に値がある → 本当の付け漏れ．足す
         header : 種名側にだけ字がある(`Begleiter:` など) → 決まりどおり足さない
         blank  : どこにも字がない → 検出の取りすぎ．足さない
    3. `value` の帯について，**黒画素の外形**から箱を作る
       (格子の出力をそのまま正解にすると，物差しが自分を採点する形になる．
        検出結果は「どこを見るか」の手がかりにだけ使う)
       **縦の位置は黒画素から，高さと横幅は同じ段の正解の中央値**にする．
       字の外形そのままだと，正解の行(行ピッチの 0.95 倍)より薄い箱になる．
    4. 重ねて描いた画像を出す(目視の段階)．`--apply` で .json と .txt に書く

出力
    <out>/candidates.csv     候補の一覧(kind 列で振り分け)
    <out>/<image>_gaps.png   正解=緑 / 足す=赤 / 見出し=黄 / まっさら=灰
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_grid as eg
from comptea import ink

PAD = 2             # 外形に足す余白(画素)


def parse_args():
    p = argparse.ArgumentParser(description='付け漏れの row 候補を作る')
    p.add_argument('--split', nargs='+', default=['train', 'val'])
    p.add_argument('--dataset', default=eg.DATASET)
    p.add_argument('--weights', default='weights/comptea.pt')
    p.add_argument('--conf', type=float, default=30)
    p.add_argument('--conf-col', type=float, default=20,
                   help='col だけの閾値(%%)．run_pipeline.py と同じ 20 が既定')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--out', default='work/label_gaps')
    p.add_argument('--apply', default=None, help='この CSV の value 行を .json/.txt に書く')
    p.add_argument('--labelme', default='labelme_data', help='labelme の .json のある場所')
    return p.parse_args()


def block_x(df_loc, block, name):
    g = df_loc[(df_loc['block'] == block) & (df_loc['obj_name'] == name)]
    if g.empty:
        return None
    return float(g['x1'].min()), float(g['x2'].max())


def truth_rows_px(label_path, names, w, h):
    """正解の row を画素の箱で返す((x1, y1, x2, y2) の並び)"""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding='utf-8').splitlines():
        f = line.split()
        if len(f) < 5 or names.get(int(f[0])) != 'row':
            continue
        cx, cy, bw, bh = (float(v) for v in f[1:5])
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def scan(args):
    dataset = Path(args.dataset)
    names = eg.class_names(dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for split in args.split:
        img_dir = dataset / 'images' / split
        lbl_dir = dataset / 'labels' / split
        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        for image in images:
            recs += scan_one(image, lbl_dir, names, split, args, out_dir)
    df = pd.DataFrame(recs)
    csv = out_dir / 'candidates.csv'
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    print('')
    print('=== まとめ ===')
    if len(df):
        print(df['kind'].value_counts().to_string())
    print('書いた: {}'.format(csv))
    return df


def scan_one(image, lbl_dir, names, split, args, out_dir):
    w, h = Image.open(image).size
    items = eg.truth_items(lbl_dir / (image.stem + '.txt'), names, w, h)
    df_loc, _ = eg.run_one(image, args)
    if df_loc is None or not len(df_loc):
        print('=== {}: 検出なし'.format(image.name))
        return []
    ranges, _ = eg.block_ranges(df_loc)
    truth_b = eg.truth_by_block(items['row'], ranges)
    found = eg.located_bands(df_loc)['row']
    truth_px = truth_rows_px(lbl_dir / (image.stem + '.txt'), names, w, h)
    dark = ink.binarize(image)

    # 正解の行を段へ振り分けて，段ごとの箱の横幅と高さ(ラベルの決まり)を作る
    conv = {}
    for x1, y1, x2, y2 in truth_px:
        conv.setdefault(eg.assign_block((x1 + x2) / 2, ranges), []).append((x1, x2, y2 - y1))
    conv = {b: (float(np.median([v[0] for v in g])),
                float(np.median([v[1] for v in g])),
                float(np.median([v[2] for v in g])))
            for b, g in conv.items()}

    recs = []
    for b in sorted(set(truth_b) | set(found), key=lambda v: (v is None, v)):
        t, f = truth_b.get(b, []), found.get(b, [])
        _, _, extra = eg.match(t, f, args.iou)
        if not extra:
            continue
        comp_x = block_x(df_loc, b, 'comp')
        name_x = block_x(df_loc, b, 'species_col') or block_x(df_loc, b, 'sname')
        row_x = conv.get(b) or (name_x or comp_x) + (float('nan'),)
        # 同じ段の正解の行の黒画素を物差しにする(画像ごとの濃さの違いを吸収する)
        base = [ink.ratio(dark, lo, hi, *comp_x) for lo, hi in t] if comp_x else []
        base = float(np.median(base)) if base else 0.02
        for j in extra:
            lo, hi = f[j]
            kind, ic, iname = ink.classify(dark, (lo, hi), comp_x, name_x, base)
            box = None
            if kind == 'value':
                ext = ink.extent(dark, lo, hi, row_x[0], row_x[1], PAD)
                if ext:
                    # 縦の中心は字から，高さは同じ段の正解の中央値に合わせる
                    ctr = (ext[0] + ext[1]) / 2
                    bh = row_x[2] if row_x[2] == row_x[2] else (ext[1] - ext[0])
                    box = (int(round(ctr - bh / 2)), int(round(ctr + bh / 2)))
            recs.append({'image': image.name, 'split': split, 'block': b,
                         'band_y1': round(lo, 1), 'band_y2': round(hi, 1),
                         'kind': kind, 'ink_comp': round(ic, 4),
                         'ink_name': round(iname, 4), 'ink_base': round(base, 4),
                         'x1': round(row_x[0], 1) if box else '',
                         'y1': box[0] if box else '',
                         'x2': round(row_x[1], 1) if box else '',
                         'y2': box[1] if box else ''})
    draw_overlay(image, truth_px, recs, out_dir)
    n = pd.Series([r['kind'] for r in recs]).value_counts().to_dict() if recs else {}
    print('=== {:<16} 正解 {:>3} / 余分 {:>3}  {}'
          .format(image.name, len(truth_px), len(recs), n))
    return recs


COLOR = {'value': (220, 30, 30), 'header': (230, 180, 0), 'blank': (140, 140, 140)}


def draw_overlay(image, truth_px, recs, out_dir):
    im = Image.open(image).convert('RGB')
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in truth_px:
        d.rectangle([x1, y1, x2, y2], outline=(0, 170, 0), width=3)
    for r in recs:
        c = COLOR[r['kind']]
        if r['kind'] == 'value' and r['y1'] != '':
            d.rectangle([r['x1'], r['y1'], r['x2'], r['y2']], outline=c, width=5)
        else:
            d.rectangle([0, r['band_y1'], im.width - 1, r['band_y2']], outline=c, width=3)
    p = Path(out_dir) / (Path(image).stem + '_gaps.png')
    im.save(p)


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    return '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
        cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def apply(args):
    """CSV の value 行を，labelme の .json と YOLO の .txt へ書き足す"""
    df = pd.read_csv(args.apply)
    df = df[df['kind'] == 'value']
    dataset = Path(args.dataset)
    names = eg.class_names(dataset)
    cls = [k for k, v in names.items() if v == 'row'][0]
    for (image, split), g in df.groupby(['image', 'split']):
        stem = Path(image).stem
        jf = Path(args.labelme) / (stem + '.json')
        obj = json.loads(jf.read_text(encoding='utf-8'))
        w, h = obj['imageWidth'], obj['imageHeight']
        add = [(float(r.x1), float(r.y1), float(r.x2), float(r.y2))
               for r in g.itertuples()]
        for x1, y1, x2, y2 in add:
            obj['shapes'].append({'label': 'row',
                                  'points': [[x1, y1], [x2, y2]],
                                  'group_id': None, 'description': '',
                                  'shape_type': 'rectangle', 'flags': {}, 'mask': None})
        jf.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        tf = dataset / 'labels' / split / (stem + '.txt')
        lines = tf.read_text(encoding='utf-8').rstrip('\n').splitlines()
        lines += [yolo_line(cls, b, w, h) for b in add]
        tf.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('足した: {:<16} {:>3} 本  ({} / {})'.format(image, len(add), jf.name, tf.name))


def main():
    args = parse_args()
    if args.apply:
        return apply(args)
    scan(args)


if __name__ == '__main__':
    sys.exit(main())
