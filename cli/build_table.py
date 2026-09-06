"""関門3: 縦持ちの表に組み，機械でできる検査をかける

    python build_table.py WORKDIR [--keep-absent]

出力
    comp_table_long.csv   1行 = 1地点 x 1種(これが正)
    comp_table_wide.csv   確認用(組成表の見た目)
    plot_table.csv        1行 = 1地点(表頭の属性)
    checks.txt            検査の結果(標準出力と同じ)

検査は**判定するだけ**．引っかかったものが誤りかどうかは，
画像に戻って目で確かめる(references/checkpoints.md 関門3)．
"""
import argparse
import sys

import _common


def parse_args():
    p = argparse.ArgumentParser(description='縦持ちに組んで検査する(関門3)')
    p.add_argument('workdir')
    p.add_argument('--keep-absent', action='store_true', help='非出現のセルも残す')
    return p.parse_args()


def check(df_long, df_plot, out):
    """機械でできる検査．進捗の記録で効くと分かっているものを並べる"""
    body = df_long[df_long['source'] == 'body']
    once = df_long[df_long['source'] == 'once']

    # 1. 地点番号の超過．「1回出現種」の目印(in N :)を読み違えると，
    #    存在しない地点番号が出る．いちばんよく効く検査
    n_plot = int(body['plot'].max()) if len(body) else 0
    over = sorted(set(once.loc[once['plot'] > n_plot, 'plot'])) if len(once) and n_plot else []
    out.append(f'[地点数] 本体 {n_plot} 地点')
    if over:
        out.append(f'  ! 1回出現種に，表に無い地点番号がある: {over}')
    elif len(once):
        out.append(f'  OK 1回出現種 {len(once)} 件は，すべて表の地点の範囲に収まる')

    # 2. 同じ地点に同じ種．1回出現種は定義上1地点1回なので誤りの徴候．
    #    ただし階層が違えば正当(同じ種が高木層と低木層に出る)
    key = ['plot', 'j_name']
    # **辞書にある名前だけを見る**(2026-09-06)．種として確からしくない値は，
    # 同じ文字列が何行にも並ぶので「同じ種が 2 つ」と鳴る．中身は
    # 種群の見出し(`亜群集区分種`)・候補が複数のまま(`クグ;クコ;…`)・
    # 1 文字の読み残りで，どれも種ではない(折り込みで 9 件のうち 7 件)
    import correct_text                       # setup() で yolo/ が通ってから

    named = df_long[df_long['j_name'].notna()].copy()
    known = correct_text.known_names(named['j_name'])
    dup = named[known].groupby(key).filter(lambda g: len(g) > 1)
    if len(dup):
        out.append(f'[重複] 同じ地点に同じ種が {dup.groupby(key).ngroups} 組')
        for (plot, name), g in dup.groupby(key):
            layers = list(g['layer'].fillna(''))
            ok = '階層違い' if len(set(layers)) == len(layers) else '要確認'
            out.append(f'  {ok}: 地点{plot} {name} 階層={layers}')
    else:
        out.append('[重複] なし')

    # 3. 被度として読めない形．OCR の誤りが残っている
    ng = df_long[df_long['status'] == 'Need Check']
    out.append(f'[被度] 読めない形 {len(ng)} 件')
    for _, r in ng.head(20).iterrows():
        out.append(f'  地点{r["plot"]} 行{r["row_no"]} {r["j_name"]} = {r["comp_raw"]!r}')

    # 4. 位置決めで気になった点が残っているセル
    note = df_long['note'].fillna('')
    n_note = int((note != '').sum())
    out.append(f'[位置] note の付いた行 {n_note} 件'
               + ('(内挿・ずらしたセル．overlay で色が付いている)' if n_note else ''))

    # 5. 表頭．値が1つも読めない地点があれば，読み直す
    if len(df_plot):
        cols = [c for c in df_plot.columns
                if c not in ('source_image', 'table', 'plot')]
        out.append(f'[表頭] {len(df_plot)} 地点 x {len(cols)} 項目')
        for _, r in df_plot.iterrows():
            miss = [c for c in cols if r[c] is None or r[c] != r[c]]
            if miss:
                out.append(f'  地点{r["plot"]}: 読めていない項目 {miss}')
    else:
        out.append('[表頭] 組み立てられなかった')


def table_id(work):
    """1ページに表が2つ以上あるときの，この置き場の表番号

    run_pipeline.py が `table.txt` を置く．無ければ表は1つ．
    """
    from pathlib import Path

    f = Path(work) / 'table.txt'
    if not f.is_file():
        return None
    return int(f.read_text(encoding='utf-8').split('/')[0].strip())


def main():
    args = parse_args()
    work, = _common.setup([args.workdir])
    from pathlib import Path
    work = Path(work)
    _common.need_file(work / 'ocred.csv', 'run_ocr.py')

    import pandas as pd
    import comp_table
    import plot_table

    df = pd.read_csv(work / 'ocred.csv')
    df_long = comp_table.comp_table(df, keep_absent=args.keep_absent)
    df_plot = plot_table.plot_table(df)

    # 1ページに表が2つ以上あるとき，どちらの表かを列に残す．
    # 画像のパスは同じで地点番号もそれぞれ 1 から振り直されるので，
    # これが無いと後で束ねたときに別の表の地点と衝突する
    table_no = table_id(work)
    if table_no:
        for d in (df_long, df_plot):
            if len(d):
                d.insert(1, 'table', table_no)

    df_long.to_csv(work / 'comp_table_long.csv', index=False)
    comp_table.to_wide(df_long).to_csv(work / 'comp_table_wide.csv', index=False)
    if len(df_plot):
        df_plot.to_csv(work / 'plot_table.csv', index=False)

    out = [f'rows   : {len(df_long)}  (body {int((df_long["source"] == "body").sum())}'
           f' / once {int((df_long["source"] == "once").sum())})']
    if len(df_long):
        out.append(f'plots  : {df_long["plot"].nunique()}')
        out.append(f'species: {df_long["row_no"].nunique()} 行')
        out.append('status : ' + ', '.join(
            f'{k}={v}' for k, v in df_long['status'].value_counts().items()))
    out.append('--- 検査 ---')
    check(df_long, df_plot, out)

    text = '\n'.join(out)
    print(text)
    _common.show_warnings(df_long.attrs.get('warnings', []), '--- comp_table の警告 ---')
    _common.show_warnings(df_plot.attrs.get('warnings', []), '--- plot_table の警告 ---')
    (work / 'checks.txt').write_text(text, encoding='utf-8')
    print(f'\n書いた: {work / "comp_table_long.csv"}')


if __name__ == '__main__':
    sys.exit(main())
