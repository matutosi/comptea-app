"""表の載っていないページから「1回出現種」の続きと注記を拾う

    段階1  python cli/read_once_page.py IMAGE --out DIR
           塊の位置を決め，切り出して `once_block.png` に置く
    段階2  python cli/read_once_page.py IMAGE --out DIR --text DIR/once.txt \
               --table kinki_045 --start-plot 5 [--note DIR/note.txt]
           読んだ文字列から縦持ちの行と注記を作る

**読むのは人か AI**．学名は斜体で，EasyOCR では 0.62 までしか合わない
(2026-09-01 に測定)．ここは位置を決めて切り出すまでを担う．

なぜ要るか
    組成表の下の流し込みが紙面に収まらないと，次のページへあふれる．
    あふれた先は本文や写真だけのページなので，`looks_like_no_table()` が
    ページごと捨てていた(2026-09-07 にユーザ指摘で判明．88 枚中 9 枚)．

出力(--out の下)
    once_block.png   塊を切り出した画像(これを読む)
    lines.csv        ページの行と位置(範囲の決め方を確かめるため)
    once_rows.csv    縦持ちの行(comp_table_long と同じ列)
    site_notes.csv   地点ごとの調査地・調査年月日・出典
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comptea import once_page, parse_text  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='ページをまたいだ1回出現種を拾う')
    p.add_argument('image', help='表の載っていないページの画像')
    p.add_argument('--out', required=True, help='出力先のディレクトリ')
    p.add_argument('--text', default=None,
                   help='読んだ文字列のファイル(段階2)')
    p.add_argument('--note', default=None,
                   help='注記の文字列のファイル(省略すると --text の残りを見る)')
    p.add_argument('--table', default=None,
                   help='続き元の表の名前(comp_table_long の table に入る)')
    p.add_argument('--start-plot', type=int, default=None,
                   help='目印より前の断片を入れる地点番号(前ページの最後の地点)')
    return p.parse_args(argv)


def crop_block(image, out_dir):
    """塊を見つけて切り出す(段階1)"""
    import pandas as pd
    from PIL import Image

    found = once_page.find_block(image)
    if found is None:
        print('この画像に「1回出現種」の塊は見あたらない')
        return None
    pd.DataFrame(found['lines']).to_csv(
        os.path.join(out_dir, 'lines.csv'), index=False, encoding='utf-8-sig')
    img = Image.open(image)
    box = tuple(int(v) for v in found['box'])
    img.crop(box).save(os.path.join(out_dir, 'once_block.png'))
    kind = '目印あり' if found['has_mark'] else '前のページからの続き'
    print(f'塊 {kind}  行 {found["span"][0]}-{found["span"][1] - 1}  外形 {box}')
    if found['note']:
        print(f'注記: {found["note"][:120]}')
        with open(os.path.join(out_dir, 'note.txt'), 'w',
                  encoding='utf-8') as f:
            f.write(found['note'])
    return found


def build_rows(text, note, table, start_plot):
    """読んだ文字列から縦持ちの行と注記を作る(段階2)"""
    import pandas as pd

    rows = parse_text.parse_once_species(text, start_plot=start_plot)
    for r in rows:
        r['table'] = table or ''
        r['row_no'] = None
        r['cover'], _, r['sociability'] = r.pop('comp_raw').partition(';')
        r['constancy'] = ''
        r['status'] = 'Need Check'
        r['note'] = 'ページをまたいだ1回出現種'
        r['source'] = 'once'
    cols = ['table', 'plot', 'row_no', 'j_name', 's_name', 'layer', 'cover',
            'sociability', 'constancy', 'status', 'note', 'source']
    df = pd.DataFrame(rows, columns=cols)
    notes = parse_text.parse_site_notes(note) if note else []
    for n in notes:
        n['table'] = table or ''
    df_note = pd.DataFrame(notes, columns=['table', 'plot', 'field', 'value'])
    return df, df_note


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.text is None:
        crop_block(args.image, args.out)
        return 0
    with open(args.text, encoding='utf-8') as f:
        text = f.read()
    note = ''
    note_path = args.note or os.path.join(args.out, 'note.txt')
    if os.path.exists(note_path):
        with open(note_path, encoding='utf-8') as f:
            note = f.read()
    df, df_note = build_rows(text, note, args.table, args.start_plot)
    df.to_csv(os.path.join(args.out, 'once_rows.csv'), index=False,
              encoding='utf-8-sig')
    df_note.to_csv(os.path.join(args.out, 'site_notes.csv'), index=False,
                   encoding='utf-8-sig')
    plots = sorted(set(df['plot'])) if len(df) else []
    print(f'縦持ち {len(df)} 行(地点 {plots})  注記 {len(df_note)} 件')
    return 0


if __name__ == '__main__':
    sys.exit(main())
