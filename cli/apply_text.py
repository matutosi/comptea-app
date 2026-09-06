"""段階2: 目で読んだ文字を ocred.csv に戻す

    python apply_text.py WORKDIR --tsv fixes.tsv

fixes.tsv は次の2列(1行目は見出し．タブ区切り)．

    cell_id	text
    12	オオツルウメモドキ
    35	+;2
    88	Tab. 85. コウヤマキ植林 調査番号：SO-216 ...

流し込んだあとは，EasyOCR で読んだときと**同じ補正**を通す
(辞書との編集距離，被度の正規表現)．読み方だけを差し替え，
判定の規則は動かさないため．どちらが読んだかは read_by 列に残る．

`run_ocr.py --reader both` で作った ocred.csv には EasyOCR の読みが
控えてある(`corrected_easyocr`)．そのときは**補正後の値で突き合わせ**，
食い違ったセルを並べる(`4;4` と `4・4` のような表記の違いで騒がないため)．
"""
import argparse
import sys

import _common


def parse_args():
    p = argparse.ArgumentParser(description='目視の読みを反映する')
    p.add_argument('workdir')
    p.add_argument('--tsv', required=True, help='cell_id と text のタブ区切り')
    p.add_argument('--by', default='ai', help='read_by に残す名前')
    return p.parse_args()


def main():
    args = parse_args()
    work, tsv = _common.setup([args.workdir, args.tsv])
    from pathlib import Path
    work = Path(work)
    _common.need_file(work / 'ocred.csv', 'run_ocr.py')
    _common.need_file(tsv, '読んだ結果を書いた TSV を用意')

    import pandas as pd
    import correct_text

    df = pd.read_csv(work / 'ocred.csv')
    fixes = pd.read_csv(tsv, sep='\t', dtype={'cell_id': int}, keep_default_na=False)
    if 'text' not in fixes.columns:
        raise SystemExit('TSV に text 列が無い')
    if 'read_by' not in df.columns:
        df['read_by'] = 'easyocr'

    known = set(df['cell_id'])
    unknown = sorted(set(fixes['cell_id']) - known)
    if unknown:
        raise SystemExit(f'ocred.csv に無い cell_id がある: {unknown[:10]}')

    changed = []
    for _, fix in fixes.iterrows():
        i = df.index[df['cell_id'] == fix['cell_id']][0]
        before = df.at[i, 'text']
        text = fix['text']
        fixed = correct_text.correct_cell(df.at[i, 'obj_name'], text)
        df.at[i, 'text'] = text
        df.at[i, 'corrected'] = fixed['corrected']
        df.at[i, 'status'] = fixed['status']
        df.at[i, 'read_by'] = args.by
        changed.append((fix['cell_id'], df.at[i, 'obj_name'], before,
                        fixed['corrected'], fixed['status']))

    # --reader both のときは，EasyOCR の読みと突き合わせる
    disagree = []
    if 'corrected_easyocr' in df.columns:
        def _text(v):
            """欠測(NaN)は空として扱う('nan' という字にしない)"""
            return '' if pd.isna(v) else str(v).strip()

        for cell_id, *_ in changed:
            i = df.index[df['cell_id'] == cell_id][0]
            a = _text(df.at[i, 'corrected_easyocr'])
            b = _text(df.at[i, 'corrected'])
            if a != b:
                disagree.append((cell_id, df.at[i, 'obj_name'], a, b))
        df['agree'] = ''
        for cell_id, *_ in changed:
            i = df.index[df['cell_id'] == cell_id][0]
            df.at[i, 'agree'] = ('disagree'
                                 if cell_id in {d[0] for d in disagree} else 'ok')

    df.to_csv(work / 'ocred.csv', index=False)
    print(f'反映: {len(changed)} セル')
    for cell_id, obj, before, after, status in changed:
        mark = '  ' if status in (None, 'OK') else '! '
        print(f'{mark}#{cell_id:<5} {obj:<14} {str(before)!r:<24} -> {str(after)!r} [{status}]')
    if 'corrected_easyocr' in df.columns:
        n = len(changed)
        print('\n--- EasyOCR との突き合わせ (補正後の値で比べる) ---')
        print(f'  一致 {n - len(disagree)}/{n}   食い違い {len(disagree)}')
        for cell_id, obj, a, b in disagree[:40]:
            print(f'  #{cell_id:<5} {obj:<14} easyocr {a!r:<24} -> ai {b!r}')
        if len(disagree) > 40:
            print(f'  ... 残り {len(disagree) - 40} 件 (ocred.csv の agree 列を見る)')

    still = [c for c in changed if c[4] in ('Need Check', 'multi')]
    if still:
        print(f'\n直しても読めない形が {len(still)} 件ある．画像をもう一度見る')
    print('\n次: build_table.py で表に組む')


if __name__ == '__main__':
    sys.exit(main())
