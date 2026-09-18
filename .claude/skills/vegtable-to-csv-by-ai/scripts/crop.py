"""ページ画像の一部を切り出し，原寸の座標の目盛りを付けて保存する．

    # 読むための画像 (目盛り付き．縦 2 x 横 1 のタイルに分ける)
    python scripts/crop.py <ページ.png> --box X0 Y0 X1 Y1 --tiles 2x1 --out <置き場>

    # 表を 1 枚の画像として切り出す (折り込みの切り分け．目盛りなし・原寸)
    python scripts/crop.py <ページ.png> --box X0 Y0 X1 Y1 --raw --name tab12 --out <置き場>
    python scripts/crop.py <ページ.png> --raw --rotate 90 --name tab12 --out <置き場>   # 横倒しを起こす

- 座標はすべて**原寸の px** (見渡し画像や目盛りに書いてある数字)．`--box` を省くと全体．
  見渡しから目で読んだ座標はずれるので，`--box` は既定で上下左右に 80 px 広げて切る (`--pad`)．
- 目盛り付きの画像は，上下左右の余白に原寸の x・y を書く．**別のタイルの同じ行は，
  y の目盛りで対応を取る** (横に分けたタイルで，種名と値を結ぶのに使う)．
- `--max` は保存する画像の長辺 (既定 1800 px)．読み手はこれより大きい画像を縮めるので，
  大きくしても字は大きくならない．字を大きくしたいときは `--box` を小さくするか，タイルを増やす．
- 表示の縮小率 (保存した画像の px / 原寸の px) を必ず表示する．**0.6 を下回ったら，
  `・` と `+`，`1` と `I` の見分けが怪しくなる**ので，もっと細かく切る．
- `--rotate` は反時計回りの角度 (90・180・270)．`--raw` と組み合わせて使う．
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

MARGIN_X, MARGIN_Y = 64, 28


def _font(size):
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _step(scale):
    """表示で 40 px 以上あく，きりのよい原寸の刻み"""
    for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if s * scale >= 40:
            return s
    return 5000


def ruled(im, box, maxside=1800, grid=False):
    """box を切り出して縮め，余白に原寸の目盛りを書いた画像と縮小率を返す"""
    x0, y0, x1, y1 = box
    part = im.crop(box).convert('RGB')
    scale = min(1.0, (maxside - 2 * MARGIN_X) / max(part.width, part.height))
    w, h = max(1, round(part.width * scale)), max(1, round(part.height * scale))
    part = part.resize((w, h), Image.LANCZOS)
    out = Image.new('RGB', (w + 2 * MARGIN_X, h + 2 * MARGIN_Y), 'white')
    out.paste(part, (MARGIN_X, MARGIN_Y))
    d = ImageDraw.Draw(out)
    f = _font(15)
    step = _step(scale)
    col = (200, 0, 0)
    for y in range((y0 // step + 1) * step, y1, step):
        yy = MARGIN_Y + round((y - y0) * scale)
        d.line([(MARGIN_X - 8, yy), (MARGIN_X - 1, yy)], fill=col)
        d.line([(MARGIN_X + w, yy), (MARGIN_X + w + 7, yy)], fill=col)
        d.text((2, yy - 8), str(y), fill=col, font=f)
        d.text((MARGIN_X + w + 9, yy - 8), str(y), fill=col, font=f)
        if grid:
            d.line([(MARGIN_X, yy), (MARGIN_X + w, yy)], fill=(170, 210, 255))
    for x in range((x0 // step + 1) * step, x1, step):
        xx = MARGIN_X + round((x - x0) * scale)
        d.line([(xx, MARGIN_Y - 8), (xx, MARGIN_Y - 1)], fill=col)
        d.line([(xx, MARGIN_Y + h), (xx, MARGIN_Y + h + 7)], fill=col)
        d.text((xx - 14, 2), str(x), fill=col, font=f)
        d.text((xx - 14, MARGIN_Y + h + 9), str(x), fill=col, font=f)
        if grid:
            d.line([(xx, MARGIN_Y), (xx, MARGIN_Y + h)], fill=(170, 210, 255))
    return out, scale


def tiles(box, rows, cols, overlap):
    x0, y0, x1, y1 = box
    tw, th = (x1 - x0) / cols, (y1 - y0) / rows
    for r in range(rows):
        for c in range(cols):
            yield (r + 1, c + 1,
                   (max(x0, int(x0 + c * tw - overlap)), max(y0, int(y0 + r * th - overlap)),
                    min(x1, int(x0 + (c + 1) * tw + overlap)),
                    min(y1, int(y0 + (r + 1) * th + overlap))))


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('image')
    ap.add_argument('--box', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--tiles', default='1x1', help='縦 x 横 の枚数 (例: 3x2)')
    ap.add_argument('--overlap', type=int, default=60,
                    help='タイルを上下左右に広げる幅 (原寸 px．隣のタイルとは 2 倍重なる)')
    ap.add_argument('--pad', type=int, default=80,
                    help='--box を上下左右に広げる幅 (原寸 px．見渡しから目で読んだ範囲の誤差を吸収する)')
    ap.add_argument('--max', type=int, default=1800, help='保存する画像の長辺 (px)')
    ap.add_argument('--grid', action='store_true', help='目盛りの線を画像の上にも薄く引く')
    ap.add_argument('--raw', action='store_true', help='目盛りを付けず原寸で切り出す')
    ap.add_argument('--rotate', type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument('--name', help='出力の名前 (既定: 入力名と座標)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)

    im = Image.open(a.image)
    if a.box:
        x0, y0, x1, y1 = a.box
        box = (max(0, x0 - a.pad), max(0, y0 - a.pad),
               min(im.width, x1 + a.pad), min(im.height, y1 + a.pad))
    else:
        box = (0, 0, im.width, im.height)
    os.makedirs(a.out, exist_ok=True)
    stem = a.name or '%s_%d_%d_%d_%d' % ((os.path.splitext(os.path.basename(a.image))[0],) + box)

    if a.raw:
        part = im.crop(box)
        if a.rotate:
            part = part.rotate(a.rotate, expand=True)
        path = os.path.join(a.out, stem + '.png')
        part.save(path)
        with open(os.path.join(a.out, stem + '.box.txt'), 'w', encoding='utf-8') as f:
            f.write(f'box={box} rotate={a.rotate} source={a.image}\n')
        print(f'{path}: {part.width} x {part.height} px (原寸)．元の画像の box={box} '
              f'(--pad {a.pad} を含む) rotate={a.rotate}')
        return

    rows, cols = (int(v) for v in a.tiles.lower().split('x'))
    for r, c, tb in tiles(box, rows, cols, a.overlap):
        out, scale = ruled(im, tb, a.max, a.grid)
        suffix = f'_r{r}c{c}' if rows * cols > 1 else ''
        path = os.path.join(a.out, stem + suffix + '.png')
        out.save(path)
        warn = '  ← 0.6 未満．もっと細かく切る' if scale < 0.6 else ''
        print(f'{path}: box={tb} 縮小率 {scale:.3f}{warn}')


if __name__ == '__main__':
    main()
