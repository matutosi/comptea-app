"""読み取り(OCR)と組み上げの精度を，人が書き起こした正解表と突き合わせて数える

    py -3.12 eval_read.py --truth truth/example.tsv --run work/example

やっていること
    1. 正解表(`truth/*.tsv`．原図を拡大して人が書き起こしたもの)を読む
    2. 通しの出力(`<run>/comp_table_long.csv`)を，同じ形((行, 地点) → 値)に直す
    3. 組成部のセル・和名・学名・階層のそれぞれで一致した数を数え，
       食い違いを並べる

**判断はしない**．読み方を変えたときに，前後で比べるための物差しとして使う．

物差しの限界(承知のうえで使う)
    - **正解表は画像を見て人(AI)が起こしたもの**なので，`--reader ai` の成績は
      同じ読み方で作った答えと比べることになり，甘く出る．
      EasyOCR どうしの比較や，補正の規則を変えたときの比較には偏りが無い．
    - **行と地点の番号は格子(切り出し)の結果を鍵にする**．格子が変われば
      対応が崩れるので，行数・地点数が合っているかを先に表示する．
      行番号は**表頭の行数によって丸ごとずれる**ので，値のある最初の行どうしを
      合わせて，ずれを打ち消してから比べる(`--offset` で手で指定もできる)．
    - 組成表そのものの誤植は直さない．**印字されているとおり**を正解とする．
      ただし，**補正が標準和名に直すのが妥当なもの**は `*_alt` 列に別解を書き，
      どちらでも正解とする(印字「クロツル」・標準和名「クロヅル」など)．
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# **入れていなくても動くようにする**(2026-09-07)．`cli/` と `tests/` は
# 同じことをしているが，ここだけ抜けており，パッケージを入れていない PC では
# 実行の**途中で**落ちていた(中核の import が関数の中にあるので --help は通る)
try:
    import comptea                                  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _data import use_data_dir                       # noqa: E402

def plot_cols(truth):
    """正解表にある地点の列(p1, p2, ...)を，番号の順に返す"""
    cols = [c for c in truth.columns
            if len(c) > 1 and c[0] == 'p' and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def parse_args():
    p = argparse.ArgumentParser(description='読み取りと組み上げの精度を数える')
    p.add_argument('--truth', required=True, help='正解表 (truth/*.tsv)')
    p.add_argument('--run', required=True, help='通しの出力のあるディレクトリ')
    p.add_argument('--csv', default=None, help='食い違いを書き出す')
    p.add_argument('--offset', type=int, default=None,
                   help='行番号のずれ(既定: 値のある最初の行どうしを合わせる)')
    p.add_argument('--all', action='store_true', help='食い違いを省略せず全部出す')
    return p.parse_args()


def read_truth(path):
    df = pd.read_csv(path, sep='\t', comment='#', dtype=str).fillna('')
    df['row_no'] = df['row_no'].astype(int)
    return df


def norm_cell(v):
    """組成部のセルの表記をそろえる('.'(非出現)と空白は空にする)"""
    v = (v or '').strip()
    if v in ('.', '・', '-', ''):
        return ''
    return v.replace('・', ';').replace(' ', '')


def norm_name(v):
    """種名の表記をそろえる(前後の空白と連続する空白だけ)"""
    return ' '.join((v or '').split())


def read_run(run):
    """通しの出力を (行, 地点) → 値 の形に直す"""
    long_csv = Path(run) / 'comp_table_long.csv'
    if not long_csv.exists():
        raise SystemExit('組み上げの結果が無い: {}'.format(long_csv))
    df = pd.read_csv(long_csv, dtype=str).fillna('')
    # 表の下の「1回出現種」は行番号を持たない(格子の外なので数えない)．
    # 正解表は表の本体だけを写している
    df['row_no'] = pd.to_numeric(df['row_no'], errors='coerce')
    n_once = int(df['row_no'].isna().sum())
    df = df.dropna(subset=['row_no']).copy()
    df['row_no'] = df['row_no'].astype(int)
    if n_once:
        print('  (1回出現種など，行番号を持たない {} 行は数えない)'.format(n_once))
    cells, names = {}, {}
    for r in df.itertuples():
        cells[(r.row_no, 'p' + str(r.plot))] = norm_cell(r.comp_raw)
        # 種名と階層は行に1つ．先に出た空でない値を採る
        cur = names.setdefault(r.row_no, {'j_name': '', 's_name': '', 'layer': ''})
        for k, v in (('j_name', r.j_name), ('s_name', r.s_name), ('layer', r.layer)):
            if not cur[k]:
                cur[k] = norm_name(v)
    return cells, names, df


def main():
    args = parse_args()
    # データの置き場へ移る(COMPTEA_DATA が無ければ，いまいる場所のまま)
    use_data_dir()
    truth = read_truth(args.truth)
    cells, names, long_df = read_run(args.run)

    body = truth[truth['kind'].isin(('species', 'cont'))]
    plots = plot_cols(truth)

    # 行番号のずれを打ち消す．表頭の行数が変わると本文の行番号も丸ごと動くため
    with_value = [r.row_no for r in body.itertuples()
                  if any(norm_cell(getattr(r, p)) for p in plots)]
    offset = args.offset
    if offset is None:
        offset = (int(long_df['row_no'].min()) - min(with_value)
                  if with_value and len(long_df) else 0)
    if offset:
        cells = {(k[0] - offset, k[1]): v for k, v in cells.items()}
        names = {k - offset: v for k, v in names.items()}

    print('=== 対象 ===')
    if offset:
        print('  行番号のずれ {:+d} を打ち消して比べる'.format(offset))
    print('  正解表 {} 行 (種 {} / 続き {} / 見出し {} / 空 {})'
          .format(len(truth),
                  int((truth['kind'] == 'species').sum()),
                  int((truth['kind'] == 'cont').sum()),
                  int((truth['kind'] == 'header').sum()),
                  int((truth['kind'] == 'blank').sum())))
    print('  出力の行番号 {} - {}  地点 {}'
          .format(long_df['row_no'].min(), long_df['row_no'].max(),
                  long_df['plot'].nunique()))

    miss = []
    # --- 組成部のセル ---
    n_ok = n_all = n_occ_ok = n_occ = 0
    for r in body.itertuples():
        for p in plots:
            t = norm_cell(getattr(r, p))
            g = cells.get((r.row_no, p), '')
            n_all += 1
            if t:
                n_occ += 1
            if t == g:
                n_ok += 1
                if t:
                    n_occ_ok += 1
            else:
                miss.append({'row_no': r.row_no, 'field': p,
                             'truth': t or '(非出現)', 'got': g or '(非出現)'})
    # --- 行ごとの項目 ---
    # **値のある行だけ**を数える．長い表は「地点 x 種」の形なので，
    # 組成部に値の無い行(学名だけが折り返して印字された行など)は
    # そもそも出力に現れない．数えると読み取りのせいでない差になる．
    field_ok = {k: [0, 0] for k in ('j_name', 's_name', 'layer')}
    scored = truth[(truth['kind'] == 'species')
                   & truth.apply(lambda r: any(norm_cell(r[p]) for p in plots),
                                 axis=1)]
    for r in scored.itertuples():
        got = names.get(r.row_no, {})
        for k in field_ok:
            t = norm_name(getattr(r, k))
            alt = norm_name(getattr(r, k + '_alt', ''))
            g = got.get(k, '')
            if k == 'layer' or t:
                field_ok[k][1] += 1
                if g == t or (alt and g == alt):
                    field_ok[k][0] += 1
                else:
                    miss.append({'row_no': r.row_no, 'field': k,
                                 'truth': t + (' / ' + alt if alt else '') or '(空)',
                                 'got': g or '(空)'})

    print('')
    print('=== 成績 ===')
    print('  組成部のセル   {:>3}/{:<3} {:.3f}   (非出現も含めた全セル)'
          .format(n_ok, n_all, n_ok / n_all if n_all else float('nan')))
    print('  うち値のあるセル {:>3}/{:<3} {:.3f}'
          .format(n_occ_ok, n_occ, n_occ_ok / n_occ if n_occ else float('nan')))
    for k, label in (('j_name', '和名'), ('s_name', '学名'), ('layer', '階層')):
        ok, tot = field_ok[k]
        print('  {:<12} {:>3}/{:<3} {:.3f}'
              .format(label, ok, tot, ok / tot if tot else float('nan')))

    print('')
    print('=== 食い違い {} 件 ==='.format(len(miss)))
    show = miss if args.all else miss[:30]
    for m in show:
        print('  行{:>3} {:<8} 正解 {:<40} 読み {}'
              .format(m['row_no'], m['field'], m['truth'], m['got']))
    if len(show) < len(miss):
        print('  ... 残り {} 件 (--all で全部出す)'.format(len(miss) - len(show)))
    if args.csv:
        pd.DataFrame(miss).to_csv(args.csv, index=False, encoding='utf-8-sig')
        print('書いた: ' + args.csv)


if __name__ == '__main__':
    sys.exit(main())
