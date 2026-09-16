"""2 つの通しの結果を，表ごとに並べて比べる

    py -3.12 compare_runs.py <前の置き場> <後の置き場> [表の名前 ...]

置き場は，表ごとの作業ディレクトリ (`comp_table_long.csv`・`checks.txt` を
含むもの) を並べたディレクトリ．表の名前を省くと，両方にある表をすべて比べる．

出すもの (表ごと)
    要確認・見るべき の前後
    変わったセルの数と，**地点 1 に閉じているか** (直しが狙いどおりの場所に効いたか)
    `--ink` を付けると，箱の境にインクが乗るセルの数 (上半分・下半分)

**先に版と設定の違いを出す** (`run_info.json`)．2026-09-16 に，前の通しが
コミットしていない状態で回されていたのに気づかず，後退と取り違えた．

**判断はしない**．変わったセルは必ず中身を見る (`--cells` で並べる)．
要確認の数だけでは，直しの良し悪しは決まらない．
"""
import argparse
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import comptea                                  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='2 つの通しを表ごとに比べる')
    p.add_argument('before', help='前の通しの置き場')
    p.add_argument('after', help='後の通しの置き場')
    p.add_argument('tables', nargs='*', help='比べる表 (省くと両方にある表すべて)')
    p.add_argument('--ink', action='store_true',
                   help='箱の境にインクが乗るセルも数える (画像を開くので遅い)')
    p.add_argument('--cells', type=int, default=0,
                   help='表ごとに，変わったセルをこの数だけ並べる')
    return p.parse_args(argv)


def need_counts(work):
    """(要確認, 見るべき)．checks.txt が無ければ None"""
    p = Path(work) / 'checks.txt'
    if not p.is_file():
        return None
    s = io.open(p, encoding='utf-8', errors='replace').read()
    need = int(re.search(r'Need Check=(\d+)', s).group(1)) if 'Need Check=' in s else 0
    m = re.search(r'見るべきは\s*(\d+)', s)
    return need, int(m.group(1)) if m else need


def main(argv=None):
    import pandas as pd

    from comptea import compare
    from comptea.pipeline import common

    args = parse_args(argv)
    before, after = Path(args.before), Path(args.after)
    names = args.tables or sorted(
        d.name for d in before.iterdir()
        if (d / 'comp_table_long.csv').is_file()
        and (after / d.name / 'comp_table_long.csv').is_file())

    # 版と設定の違いを先に知らせる (最初の表で代表させる)
    if names:
        la = common.run_info_line(before / names[0])
        lb = common.run_info_line(after / names[0])
        if la != lb:
            print('!! 版か設定が違う (違いが直しによるものか確かめる)')
            print('   前: ' + (la.replace('\n', '\n       ') or '(記録なし)'))
            print('   後: ' + (lb.replace('\n', '\n       ') or '(記録なし)'))
            print()

    tot = {'na': 0, 'nb': 0, 'ma': 0, 'mb': 0, 'changed': 0, 'first': 0}
    for name in names:
        pa, pb = before / name, after / name
        if not ((pa / 'comp_table_long.csv').is_file()
                and (pb / 'comp_table_long.csv').is_file()):
            print(f'{name}: どちらかに結果が無い')
            continue
        a = pd.read_csv(pa / 'comp_table_long.csv', low_memory=False)
        b = pd.read_csv(pb / 'comp_table_long.csv', low_memory=False)
        d = compare.diff_long(a, b)
        in_first = d['by_plot'].get(1, 0)
        ca, cb = need_counts(pa) or (0, 0), need_counts(pb) or (0, 0)
        line = (f'{name}: 要確認 {ca[0]} → {cb[0]}  見るべき {ca[1]} → {cb[1]}  '
                f'変化 {d["changed"]} (地点 1 が {in_first})')
        if args.ink:
            line += '  ' + ink_line(pa, pb)
        print(line)
        for p, r, x, y in d['cells'][:args.cells]:
            print(f'    地点 {p} 行 {r}: {x!r} → {y!r}')
        tot['na'] += ca[0]; tot['nb'] += cb[0]
        tot['ma'] += ca[1]; tot['mb'] += cb[1]
        tot['changed'] += d['changed']; tot['first'] += in_first
    print(f'\n合計 {len(names)} 表: 要確認 {tot["na"]} → {tot["nb"]}  '
          f'見るべき {tot["ma"]} → {tot["mb"]}  '
          f'変化 {tot["changed"]} (地点 1 が {tot["first"]})')


def ink_line(pa, pb):
    """箱の境にインクが乗るセルの前後 (上半分・下半分)"""
    import pandas as pd
    from PIL import Image

    from comptea import compare, ink

    def cells(work):
        # **実際に読んだ箱** (`ocred.csv`) を優先する．段階 2 で読む箱だけを
        # 動かす直し (`left_rule`) は `located.csv` を変えないので，そちらで
        # 比べると差が出ない
        for fname in ('ocred.csv', 'located.csv'):
            f = Path(work) / fname
            if f.is_file():
                lo = pd.read_csv(f, low_memory=False)
                return lo, lo[lo['obj_name'] == 'comp']
        return None, None

    if cells(pa)[0] is None or cells(pb)[0] is None:
        return '(格子なし)'
    la, ca = cells(pa)
    _lb, cb = cells(pb)
    Image.MAX_IMAGE_PIXELS = None
    dark = ink.binarize(Image.open(la['source_image'].iloc[0]))
    (ta, na), (ba, nba) = compare.boundary_ink_halves(dark, ca)
    (tb, nb), (bb, nbb) = compare.boundary_ink_halves(dark, cb)
    return (f'境のインク 上 {ta}/{na} → {tb}/{nb}  下 {ba}/{nba} → {bb}/{nbb}')


if __name__ == '__main__':
    main()
