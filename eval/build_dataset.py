"""既存の教師データに，新しい資料の断片を足したデータセットを組む

**既存の train / val の分け方はそのまま**にする．そうしないと，
学習前後を `eval_grid.py` で比べるときに，val に学習済みの画像が
混ざって成績が良く見えてしまう．

新しい断片は既定で train にだけ入れる．val に入れたいときは
`--val-every N` を渡す(N 枚に1枚を val へ)．

対象にするのは，`run_pipeline.py` の**要確認の警告が無い表**だけ
(`--corpus` の CSV で選ぶ)．格子が怪しい表を学ばせない．
"""
import argparse
import os
import shutil
import sys

import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# **入れていなくても動くようにする**(2026-09-07)．`cli/` と `tests/` は
# 同じことをしているが，ここだけ抜けており，パッケージを入れていない PC では
# 実行の**途中で**落ちていた(中核の import が関数の中にあるので --help は通る)
try:
    import comptea                                  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _data import use_data_dir                       # noqa: E402
import make_labels                              # noqa: E402

Image.MAX_IMAGE_PIXELS = None

SRC = 'labelme_data/YOLODataset'


def copy_existing(src, dst, repeat=1):
    """既存のデータセットを，分け方ごとそのまま写す

    `repeat` を上げると，**train だけ**を複製して枚数を水増しする．
    新しい資料の断片が多いと学習がそちらへ偏り，元の資料の成績が落ちる
    (2026-09-02 に測ったとき，kinki_047 の row recall が 1.000 → 0.51 に
     落ちた)．同じ画像を複製するだけなので，val は増やさない．
    """
    n = {}
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            s = os.path.join(src, kind, split)
            d = os.path.join(dst, kind, split)
            os.makedirs(d, exist_ok=True)
            if not os.path.isdir(s):
                continue
            for f in os.listdir(s):
                shutil.copy2(os.path.join(s, f), os.path.join(d, f))
                if split != 'train':
                    continue
                stem, ext = os.path.splitext(f)
                for k in range(2, repeat + 1):
                    shutil.copy2(os.path.join(s, f),
                                 os.path.join(d, f'{stem}_r{k}{ext}'))
        n[split] = len(os.listdir(os.path.join(dst, 'images', split)))
    return n


def add_table(workdir, image, dst, names, split, tile_w, tile_h):
    """1つの表から断片を作り，データセットへ足す．足した枚数を返す"""
    df_loc = pd.read_csv(os.path.join(workdir, 'located.csv'))
    im = Image.open(image)
    boxes = make_labels.region_boxes(df_loc, im)
    tiles = make_labels.plan_tiles(df_loc, im.size, tile_w, tile_h)
    comp = df_loc[df_loc['obj_name'] == 'comp']
    if comp.empty or not tiles:
        return 0
    name_x2 = float(comp['x1'].min())
    stem = os.path.basename(os.path.normpath(workdir))
    n = 0
    for i, tile in enumerate(tiles, 1):
        got = make_labels.clip_boxes(boxes, tile, name_x2)
        if not any(lab == 'row' for lab, *_ in got):
            continue
        s = split(n)
        png = os.path.join(dst, 'images', s, f'{stem}_x{i:02d}.png')
        size, _ = make_labels.write_tile(im, tile, got, png, name_x2,
                                         labelme=False)
        make_labels.write_yolo(
            got, size,
            os.path.join(dst, 'labels', s, f'{stem}_x{i:02d}.txt'), names)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description='新しい資料を足したデータセットを組む')
    p.add_argument('--corpus', required=True,
                   help='run_pipeline を通した結果の CSV(table, work, must …)')
    p.add_argument('--parts', required=True, help='表の画像を置いた場所')
    p.add_argument('--work', default='work', help='作業ディレクトリの親')
    p.add_argument('--src', default=SRC, help='元のデータセット')
    p.add_argument('--out', default='labelme_data/YOLODataset_s01115',
                   help='組み上げ先')
    p.add_argument('--max-must', type=int, default=0,
                   help='要確認の警告がこの数までの表を使う')
    p.add_argument('--keep-misaligned', action='store_true',
                   help='列の境が印字と合っていない表も使う(既定では捨てる)')
    p.add_argument('--min-rows', type=int, default=20,
                   help='行がこれ未満の表は使わない')
    p.add_argument('--max-cols', type=int, default=60,
                   help='列がこれを超える表は組めていないとみなす')
    p.add_argument('--repeat-existing', type=int, default=1,
                   help='元の train を何倍に水増しするか(偏りを抑える)')
    p.add_argument('--val-every', type=int, default=0,
                   help='新しい断片の N 枚に1枚を val へ(0 なら全部 train)')
    p.add_argument('--tile-w', type=int, default=make_labels.TILE_W)
    p.add_argument('--tile-h', type=int, default=make_labels.TILE_H)
    a = p.parse_args()
    # データの置き場へ移る(COMPTEA_DATA が無ければ，いまいる場所のまま)
    use_data_dir()

    with open(os.path.join(a.src, 'dataset.yaml'), encoding='utf-8') as f:
        names = yaml.safe_load(f)['names']

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    kept = copy_existing(a.src, a.out, a.repeat_existing)
    print(f'元のデータセット: train {kept["train"]} / val {kept["val"]}')

    df = pd.read_csv(a.corpus)
    use = df[df['ok'] & (df['must'] <= a.max_must)]
    if 'misaligned' in use.columns and not a.keep_misaligned:
        # **列の境が印字の隙間と合っていない表は学ばせない**．
        # 検出とは別の測り方で出した食い違いなので，これが立っている表は
        # 格子そのものが信用できない
        use = use[use['misaligned'] == 0]
    if 'cols' in use.columns:
        # 1行しかない・列が異様に多い表は組めていない
        use = use[(use['rows'] >= a.min_rows) & (use['cols'] <= a.max_cols)]
    print(f'使う表: {len(use)} / {len(df)}')

    def split(i):
        if a.val_every and i % a.val_every == 0:
            return 'val'
        return 'train'

    total = 0
    for _, r in use.iterrows():
        img = os.path.join(a.parts, f'{r["table"]}.png')
        wd = os.path.join(a.work, str(r['work']))
        if not (os.path.isfile(img) and os.path.isdir(wd)):
            print(f'  飛ばす(見つからない): {r["work"]}')
            continue
        n = add_table(wd, img, a.out, names, split, a.tile_w, a.tile_h)
        print(f'  {r["work"]}: 断片 {n} 枚')
        total += n

    with open(os.path.join(a.out, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump({'path': os.path.abspath(a.out),
                        'train': 'images/train', 'val': 'images/val',
                        'test': None, 'names': names}, f,
                       allow_unicode=True, sort_keys=False)
    for s in ('train', 'val'):
        k = len(os.listdir(os.path.join(a.out, 'images', s)))
        print(f'{s}: {k} 枚')
    print(f'足した断片: {total} 枚 -> {a.out}')


if __name__ == '__main__':
    main()
