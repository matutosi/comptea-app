"""通した結果を 1 か所にまとめ，Need Check のセルを切り出して結び付ける

    python export_data.py WORKDIR... --out DIR [--tag NAME] [--no-crop]

WORKDIR は `run_pipeline.py` が作った作業ディレクトリか，その親
(親を渡すと，その下の作業ディレクトリを全部拾う)．

出力(--out の下)
    comp_table_long_<tag>.csv   縦持ちの本体(全部の表を縦に積む)
    plot_table_<tag>.csv        表頭の項目(1 行 = 1 地点)
    summary_<tag>.csv           表ごとの行・列・セル・OK・Need Check
    _need_check/<表>/<cell_id>.png       Need Check のセル
    _need_check/<表>/once_species.png    1回出現種の流し込み

**Need Check の行には `image` 列でフルパスを入れる**．
関門3の出力を人に渡すとき，どのセルを見ればよいかが CSV だけで分かる．
作業ディレクトリは一時のものなので，画像は出力先へ写しておく．

**1回出現種はセルを持たない**(流し込みを読み順に並べ直したもの)ので，
その領域を 1 枚切り出して，行から結び付ける．
"""
import argparse
import os
import sys

import _common

PAD = 6                 # セルを切り出すときの余白(px)
NEED_CHECK = 'Need Check'


def parse_args():
    p = argparse.ArgumentParser(description='通した結果をまとめて書き出す')
    p.add_argument('workdirs', nargs='+', help='作業ディレクトリか，その親')
    p.add_argument('--out', required=True, help='書き出し先')
    p.add_argument('--tag', default='all', help='ファイル名に付ける資料名')
    p.add_argument('--no-crop', action='store_true',
                   help='Need Check のセルを切り出さない')
    p.add_argument('--pad', type=int, default=PAD, help='切り出しの余白(px)')
    return p.parse_args()


def find_workdirs(paths):
    """作業ディレクトリを集める(親を渡されたら，その下を拾う)"""
    import glob

    out = []
    for p in paths:
        if os.path.isfile(os.path.join(p, 'comp_table_long.csv')):
            out.append(p)
            continue
        out += [d for d in sorted(glob.glob(os.path.join(p, '*')))
                if os.path.isfile(os.path.join(d, 'comp_table_long.csv'))]
    return out


def find_image(path, work):
    """記録された画像のパスが無ければ，同じ名前のものを近くから探す

    作業ディレクトリを別の場所へ写したり，元の場所を消したりすると，
    `detect.csv` に残るパスでは開けない．
    """
    if isinstance(path, str) and os.path.isfile(path):
        return path
    base = os.path.basename(str(path))
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..', '..', '..'))
    look = [work, os.path.dirname(work),
            os.path.join(root, 'yolo', 'labelme_data'),
            os.path.join(root, 'yolo', 'images')]
    for d in look:
        q = os.path.join(d, base)
        if os.path.isfile(q):
            return q
    return None


def plot_no(comp):
    """`comp_table.py` と同じ決まりで地点番号を振る

    縦持ちには `cell_id` が無いので，(行, 地点)でセルを引き当てる．
    段ごとに振り直すところまで合わせないと，折り返した紙面でずれる．
    """
    if 'block' in comp.columns:
        return comp.groupby('block')['col'].rank(method='dense').astype(int)
    return comp['col'].rank(method='dense').astype(int)


def collect(works, tag):
    """縦持ち・表頭・成績の 3 つを積み上げる"""
    import pandas as pd

    longs, plots, summary = [], [], []
    for w in works:
        name = os.path.basename(os.path.abspath(w))
        d = pd.read_csv(os.path.join(w, 'comp_table_long.csv'))
        d = d.rename(columns={'table': 'table_in_sheet'})
        d.insert(0, 'table', name)
        d.insert(0, 'corpus', tag)
        longs.append(d)
        p = os.path.join(w, 'plot_table.csv')
        if os.path.isfile(p):
            q = pd.read_csv(p)
            q = q.rename(columns={'table': 'table_in_sheet'})
            q.insert(0, 'table', name)
            q.insert(0, 'corpus', tag)
            plots.append(q)
        loc = os.path.join(w, 'located.csv')
        c = None
        if os.path.isfile(loc):
            g = pd.read_csv(loc)
            c = g[g['obj_name'] == 'comp']
        summary.append(dict(
            corpus=tag, table=name,
            rows=int(c['row'].nunique()) if c is not None else None,
            cols=int(c['col'].nunique()) if c is not None else None,
            cells=len(c) if c is not None else None,
            long=len(d), plots=int(d['plot'].nunique()),
            species=int(d['j_name'].nunique()) if 'j_name' in d else 0,
            ok=int((d['status'] == 'OK').sum()),
            need_check=int((d['status'] == NEED_CHECK).sum()),
            constancy=int(d['constancy'].notna().sum()) if 'constancy' in d else 0))
    return (pd.concat(longs, ignore_index=True) if longs else None,
            pd.concat(plots, ignore_index=True) if plots else None,
            pd.DataFrame(summary))


def crop_need_check(lg, works, out, pad):
    """Need Check のセルを切り出し，`image` 列にフルパスを入れる"""
    import pandas as pd
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    lg['image'] = ''
    n_cell = n_once = n_miss = 0
    for w in works:
        name = os.path.basename(os.path.abspath(w))
        sub = lg[(lg['table'] == name) & (lg['status'] == NEED_CHECK)]
        if sub.empty:
            continue
        f = os.path.join(w, 'ocred.csv')
        det = os.path.join(w, 'detect.csv')
        if not (os.path.isfile(f) and os.path.isfile(det)):
            n_miss += len(sub)
            continue
        d = pd.read_csv(f)
        comp = d[d['obj_name'] == 'comp'].copy()
        if comp.empty:
            n_miss += len(sub)
            continue
        comp['plot'] = plot_no(comp)
        key = {(int(a), int(b)): c
               for a, b, c in zip(comp['row'], comp['plot'], comp.itertuples())}
        src = find_image(pd.read_csv(det).iloc[0]['source_image'], w)
        if src is None:
            print(f'  画像が見つからない: {name}')
            n_miss += len(sub)
            continue
        img = Image.open(src).convert('RGB')
        dst = os.path.join(out, '_need_check', name)
        os.makedirs(dst, exist_ok=True)
        once_path = ''
        once = d[d['obj_name'] == 'once_species']
        if len(once) and sub['row_no'].isna().any():
            box = (max(0, int(once['x1'].min()) - pad),
                   max(0, int(once['y1'].min()) - pad),
                   min(img.width, int(once['x2'].max()) + pad),
                   min(img.height, int(once['y2'].max()) + pad))
            once_path = os.path.join(dst, 'once_species.png')
            img.crop(box).save(once_path)
        for idx, r in sub.iterrows():
            if pd.isna(r['row_no']):
                if once_path:
                    lg.at[idx, 'image'] = once_path
                    n_once += 1
                else:
                    n_miss += 1
                continue
            c = key.get((int(r['row_no']), int(r['plot'])))
            if c is None:
                n_miss += 1
                continue
            box = (max(0, int(c.x1) - pad), max(0, int(c.y1) - pad),
                   min(img.width, int(c.x2) + pad),
                   min(img.height, int(c.y2) + pad))
            p = os.path.join(dst, f'{int(c.cell_id)}.png')
            img.crop(box).save(p)
            lg.at[idx, 'image'] = p
            n_cell += 1
    return n_cell, n_once, n_miss


def main():
    args = parse_args()
    # **chdir する前に絶対パスへ直す**．`setup()` は中核の場所へ移るので，
    # 受け取った相対パスはそのままでは解決できない
    paths = _common.setup(list(args.workdirs) + [args.out])
    import pandas as pd                     # noqa: F401

    roots, out = paths[:-1], paths[-1]
    works = find_workdirs(roots)
    if not works:
        print('作業ディレクトリが見つからない(build_table.py まで通してから実行する)')
        sys.exit(1)
    args.out = out
    os.makedirs(out, exist_ok=True)
    lg, pt, sm = collect(works, args.tag)
    if lg is None:
        print('縦持ちが 1 つも無い')
        sys.exit(1)
    n_cell = n_once = n_miss = 0
    if not args.no_crop:
        n_cell, n_once, n_miss = crop_need_check(lg, works, args.out, args.pad)
    for fname, d in ((f'comp_table_long_{args.tag}.csv', lg),
                     (f'plot_table_{args.tag}.csv', pt),
                     (f'summary_{args.tag}.csv', sm)):
        if d is not None and len(d):
            d.to_csv(os.path.join(args.out, fname), index=False,
                     encoding='utf-8-sig')
            print(f'  {fname}: {len(d)} 行')
    print(f'{args.tag}: 表 {len(sm)}  縦持ち {int(sm["long"].sum())}  '
          f'OK {int(sm["ok"].sum())}  Need Check {int(sm["need_check"].sum())}')
    if not args.no_crop:
        print(f'  切り出し: セル {n_cell} 枚  1回出現種の領域に結んだ行 {n_once}'
              + (f'  結べなかった行 {n_miss}' if n_miss else ''))


if __name__ == '__main__':
    main()
