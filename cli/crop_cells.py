"""段階2: 目視で読むセルを切り出す

    python crop_cells.py WORKDIR [--what review|regions|all] [--ids 12,34]
                                 [--class comp] [--by-class]

出力(work/crops/)
    region_<cell_id>_<クラス>.png   文章として読む領域(表頭・1回出現種)は原寸で1枚ずつ
    sheet_NN.png                    小さいセルは cell_id を焼き込んで1枚にまとめる
    sheet_NN_<クラス>.png           `--by-class` のとき．1枚が1クラスだけになる
    index.tsv                       どの cell_id がどの画像に入っているか

小さいセルを1枚ずつ渡すと読む回数が増えるので，**まとめて1枚**にする．
番号が焼いてあるので，読んだ結果は cell_id で指して apply_text.py に渡せる．

`--by-class` は，読む側が**字種を決め打ちできる**ようにするためのもの．
組成部なら来る字は `5 4 3 2 1 + r ・` だけ，階層なら `B T S K H M` と数字だけ．
字種を絞ると読みが良くなるのは `ocr.retry_empty_comp()` で実証済み
(kinki_047 の組成部 0.446 → 1.000)．
"""
import argparse
import sys

from comptea.pipeline import common as _common

TILE_H = 110       # まとめる画像の1タイルの高さ(拡大して読みやすくする)
TILE_W_MAX = 420   # 横に長いセルはここで頭打ちにする
COLS = 4           # 1枚に並べる列数
PER_SHEET = 24
REGIONS = ('header', 'once_species')


def parse_args():
    p = argparse.ArgumentParser(description='目視で読むセルを切り出す')
    p.add_argument('workdir')
    p.add_argument('--what', default='review', choices=['review', 'regions', 'all'],
                   help='review=review.tsv のセル / regions=文章の領域だけ / all=全部')
    p.add_argument('--ids', default=None, help='cell_id を直に指定(カンマ区切り)')
    p.add_argument('--class', dest='cls', default=None, help='クラスで絞る(カンマ区切り)')
    p.add_argument('--pad', type=int, default=6, help='切り出しの余白(px)')
    p.add_argument('--by-class', action='store_true',
                   help='一覧画像をクラスごとに分ける(1枚が1クラスだけになる)')
    return p.parse_args()


def select(df, args, work):
    import pandas as pd
    if args.ids:
        ids = {int(i) for i in args.ids.split(',')}
        return df[df['cell_id'].isin(ids)]
    if args.cls:
        return df[df['obj_name'].isin(args.cls.split(','))]
    if args.what == 'regions':
        return df[df['obj_name'].isin(REGIONS)]
    if args.what == 'all':
        return df
    review_path = work / 'review.tsv'
    _common.need_file(review_path, 'run_ocr.py')
    ids = set(pd.read_csv(review_path, sep='\t')['cell_id']) if review_path.stat().st_size else set()
    return df[df['cell_id'].isin(ids)]


def crop(img, row, pad):
    x1, y1 = max(0, int(row['x1']) - pad), max(0, int(row['y1']) - pad)
    x2, y2 = min(img.width, int(row['x2']) + pad), min(img.height, int(row['y2']) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def _sheet_groups(tiles, by_class):
    """一覧画像に並べるまとまりを返す [(クラス名, タイル), ...]

    既定は全部を1つのまとまりにする(これまでどおり)．
    `--by-class` のときはクラスごとに分け，**1枚を1クラスだけ**にする．
    並びは最初に出てきた順(cell_id の順)を保つ．
    """
    if not by_class:
        return [('', tiles)]
    order, groups = [], {}
    for t in tiles:
        cls = t[3]
        if cls not in groups:
            order.append(cls)
            groups[cls] = []
        groups[cls].append(t)
    return [(cls, groups[cls]) for cls in order]


def make_sheet(tiles, path):
    """cell_id を焼き込んだタイルを1枚に並べる"""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('arial.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    label_h = 26
    tw = max(t[1].width for t in tiles)
    th = max(t[1].height for t in tiles) + label_h
    cols = min(COLS, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * (tw + 8) + 8, rows * (th + 8) + 8), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (cell_id, tile, label, *_) in enumerate(tiles):
        x = 8 + (i % cols) * (tw + 8)
        y = 8 + (i // cols) * (th + 8)
        draw.rectangle([x - 2, y - 2, x + tw + 1, y + th + 1], outline='gray')
        draw.text((x, y), f'#{cell_id} {label}', fill='blue', font=font)
        sheet.paste(tile, (x, y + label_h))
    sheet.save(path)


def main():
    args = parse_args()
    work, = _common.setup([args.workdir])
    from pathlib import Path
    work = Path(work)

    import pandas as pd
    from PIL import Image

    src = work / 'ocred.csv'
    if not src.is_file():
        src = work / 'located.csv'
    _common.need_file(src, 'run_pipeline.py')
    df = pd.read_csv(src)
    target = select(df, args, work)
    if target.empty:
        print('切り出すセルが無い')
        return

    out = work / 'crops'
    out.mkdir(exist_ok=True)
    for old in out.glob('*.png'):
        old.unlink()

    index, tiles, sheets = [], [], 0
    for image_path, group in target.groupby('source_image'):
        img = Image.open(image_path).convert('RGB')
        for _, row in group.iterrows():
            piece = crop(img, row, args.pad)
            if piece is None:
                continue
            cell_id = int(row['cell_id'])
            if row['obj_name'] in REGIONS:
                # 文章の領域は原寸のまま．小さくすると読めなくなる
                name = f'region_{cell_id}_{row["obj_name"]}.png'
                piece.save(out / name)
                index.append({'cell_id': cell_id, 'obj_name': row['obj_name'],
                              'file': name, 'kind': 'region'})
                continue
            scale = TILE_H / piece.height
            size = (min(TILE_W_MAX, max(1, int(piece.width * scale))), TILE_H)
            label = f'{row["obj_name"]} r{row.get("row")}c{row.get("col")}'
            tiles.append((cell_id, piece.resize(size, Image.LANCZOS), label,
                          str(row['obj_name'])))
            index.append({'cell_id': cell_id, 'obj_name': row['obj_name'],
                          'file': '', 'kind': 'tile'})
    # まとめた画像は，1枚に詰めすぎると読めないので分ける．
    # `--by-class` を付けると**1枚を1クラスだけ**にする．読む側が
    # 「このセルに来る字は `5 4 3 2 1 + r ・` だけ」と決め打ちできる
    # (字種を絞ると読みが良くなるのは ocr.retry_empty_comp() で実証済み)
    for cls, group_tiles in _sheet_groups(tiles, args.by_class):
        for i in range(0, len(group_tiles), PER_SHEET):
            sheets += 1
            chunk = group_tiles[i:i + PER_SHEET]
            name = (f'sheet_{sheets:02d}_{cls}.png' if args.by_class
                    else f'sheet_{sheets:02d}.png')
            make_sheet(chunk, out / name)
            ids = {c[0] for c in chunk}
            for rec in index:
                if rec['kind'] == 'tile' and rec['cell_id'] in ids:
                    rec['file'] = name

    pd.DataFrame(index).to_csv(out / 'index.tsv', sep='\t', index=False)
    print(f'切り出した: {len(index)} セル -> {out}')
    for f in sorted(out.glob('*.png')):
        print(f'  {f}')
    print('\n次: 画像を読み，cell_id と読んだ文字を fixes.tsv に書いて apply_text.py')


if __name__ == '__main__':
    sys.exit(main())
