"""段が丸ごと落ちていないかを，ラベルの無い画像も含めて数える

    py -3.12 scan_blocks.py --dir labelme_data --conf-col 20

'col' が1本も検出されないと，その段の組成セルは1つも作られない．
組成部が丸ごと空になるので，**取りこぼしのうち一番痛い**．
これは正解ラベルが無くても分かる失敗なので，ラベルのある 33 枚に限らず
手元の画像すべてで数えられる．閾値を決めるときの土俵をそろえるために使う．

eval_grid.py が「どれだけ正しく取れたか」を測るのに対し，
こちらは「丸ごと落ちていないか」だけを見る(ラベルが要らない)．

**本文や写真だけのページ(組成表でないページ)は数えない**．
手元の資料には隣のページの続きだけが載ったページが混ざっていて，
これを落ちた画像に数えると実力を過小に見せる(2026-09-01)．
判定は locate.looks_like_no_table() を見る．
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description='段が丸ごと落ちる件数を数える')
    p.add_argument('--dir', default='labelme_data', help='画像のあるディレクトリ')
    p.add_argument('--weights', default='weights/comptea.pt')
    p.add_argument('--conf', type=float, default=30)
    p.add_argument('--conf-col', type=float, default=20,
                   help='col だけの閾値(%%)．run_pipeline.py と同じ 20 が既定')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--csv', default=None)
    return p.parse_args()


def scan_one(image, args, eval_grid, locate):
    """1枚ぶんの結果を1行にまとめる"""
    rec = {'image': image.name, 'blocks': 0, 'blocks_no_comp': 0,
           'comp': 0, 'rows': 0, 'cols': 0, 'warnings': 0,
           'error': '', 'no_table': False}
    try:
        df, warns = eval_grid.run_one(image, args)
    except Exception as e:
        rec['error'] = type(e).__name__
        return rec
    rec['warnings'] = len(warns)
    if df is None or len(df) == 0:
        # 検出が0件．表のないページとみなす(locate.looks_like_no_table)
        rec['no_table'] = True
        return rec
    if locate.looks_like_no_table(df):
        rec['no_table'] = True
        return rec
    ranges, empty = eval_grid.block_ranges(df)
    comp = df[df['obj_name'] == 'comp']
    rec['blocks'] = len(ranges)
    rec['blocks_no_comp'] = len(empty)
    rec['comp'] = len(comp)
    if not comp.empty:
        rec['rows'] = int(comp.groupby('block')['row'].nunique().sum())
        rec['cols'] = int(comp.groupby('block')['col'].nunique().sum())
    return rec


def main():
    args = parse_args()
    # Windows の既定のコードページだと日本語が化ける
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    import eval_grid
    from comptea import locate

    d = Path(args.dir)
    images = sorted(p for p in d.iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not images:
        raise SystemExit('画像がない: ' + str(d))
    print('conf={} conf_col={} imgsz={}  画像 {} 枚'
          .format(args.conf, args.conf_col, args.imgsz, len(images)))

    rows = []
    for i, image in enumerate(images, 1):
        rec = scan_one(image, args, eval_grid, locate)
        rows.append(rec)
        if rec['no_table']:
            print('  {:>3}/{} {:<22} <- 組成表でないページ(数えない)'
                  .format(i, len(images), rec['image']))
            continue
        bad = rec['error'] or rec['blocks_no_comp']
        if bad:
            print('  {:>3}/{} {:<22} 段 {} / 組成の無い段 {} / comp {} {}'
                  .format(i, len(images), rec['image'], rec['blocks'],
                          rec['blocks_no_comp'], rec['comp'],
                          '<- ' + rec['error'] if rec['error'] else '<- 落ちている'))

    df = pd.DataFrame(rows)
    # 組成表でないページは土俵に載せない(載せると実力を過小に見せる)
    skipped = df[df['no_table']]
    tbl = df[~df['no_table']]
    ng = tbl[(tbl['blocks_no_comp'] > 0) | (tbl['error'] != '')]
    print('')
    print('=== まとめ ===')
    print('  画像            {} (組成表 {} / 表でない {})'
          .format(len(df), len(tbl), len(skipped)))
    print('  段の合計        {}'.format(int(tbl['blocks'].sum())))
    print('  組成の無い段    {}'.format(int(tbl['blocks_no_comp'].sum())))
    print('  落ちた画像      {}{}'
          .format(len(ng),
                  ' ({:.1%})'.format(len(ng) / len(tbl)) if len(tbl) else ''))
    print('  格子を作れず    {}'.format(int((tbl['error'] != '').sum())))
    print('  comp セル合計   {}'.format(int(tbl['comp'].sum())))
    if len(skipped):
        print('  表でないページ  ' + ', '.join(skipped['image']))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print('書いた: ' + args.csv)


if __name__ == '__main__':
    sys.exit(main())
