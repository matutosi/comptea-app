"""組成表の入力 (PDF・PNG・JPG) を，1 ページ 1 枚の PNG にそろえる．

    python scripts/pages.py <入力> [<入力> ...] --out <作業ディレクトリ>

- PDF は 1 ページずつ書き出す．ページが 1 枚のスキャン画像だけなら，
  その画像を**原寸のまま**取り出す (描画し直すと解像度が変わるため)．
  それ以外は 300 dpi で描画する．
- PNG・JPG はそのまま写す (EXIF の向きは直す)．
- 書き出した各ページについて，全体を見渡すための縮小画像
  (`<名前>_overview.png`．目盛りの数字は原寸の px) も作る．
- 最後に，ページの一覧 (名前・幅・高さ・縮小率) を表示し，`pages.tsv` に書く．
"""
import argparse
import io
import os
import sys

from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop import ruled  # noqa: E402

Image.MAX_IMAGE_PIXELS = None               # A0 の折り込みは 1 億画素を超える


def pdf_pages(path):
    import pymupdf
    doc = pymupdf.open(path)
    for i, page in enumerate(doc, 1):
        imgs = page.get_images(full=True)
        im = None
        if len(imgs) == 1:
            info = doc.extract_image(imgs[0][0])
            try:
                im = Image.open(io.BytesIO(info['image']))
                im.load()
            except Exception:
                im = None
        if im is None:
            pix = page.get_pixmap(dpi=300)
            im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        yield i, im


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', required=True, help='作業ディレクトリ')
    ap.add_argument('--overview', type=int, default=1800,
                    help='見渡し画像の長辺 (px．既定 1800)')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for src in a.inputs:
        base = os.path.splitext(os.path.basename(src))[0]
        if src.lower().endswith('.pdf'):
            ims = [im for _, im in pdf_pages(src)]
            items = ([(base, ims[0])] if len(ims) == 1 else
                     [(f'{base}_p{i:02d}', im) for i, im in enumerate(ims, 1)])
        else:
            items = [(base, ImageOps.exif_transpose(Image.open(src)))]
        for name, im in items:
            if im.mode not in ('RGB', 'L'):
                im = im.convert('RGB')
            path = os.path.join(a.out, name + '.png')
            im.save(path)
            ov, scale = ruled(im, (0, 0, im.width, im.height), a.overview)
            ov.save(os.path.join(a.out, name + '_overview.png'))
            rows.append((name, im.width, im.height, scale))
    with open(os.path.join(a.out, 'pages.tsv'), 'w', encoding='utf-8') as f:
        f.write('page\twidth\theight\toverview_scale\n')
        for r in rows:
            f.write('%s\t%d\t%d\t%.3f\n' % r)
    for name, w, h, s in rows:
        print(f'{name}: {w} x {h} px  (見渡しの縮小率 {s:.3f})')


if __name__ == '__main__':
    main()
