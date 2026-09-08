"""枝番の付いたページを1つの表につなぐ

    python cli/link_pages.py WORKDIR... --out DIR

WORKDIR は `run_pipeline.py` が作った置き場か，その親(親を渡すと下を全部拾う)．

なぜ要るか
    組成表の下の流し込み(出現1回の種・調査地・調査年月日・出典)は，紙面に
    収まらないと次のページへあふれる．どのページが続きかはノンブルの順で
    分かるはずだが，**紙面の配置の都合で崩れる**(`kinki_081` = p.383 と
    `kinki_082` = p.382)．そこで**ファイル名の枝番で明示する**(2026-09-08 決定)．

        組成表だけ          xxx.jpg
        組成表と続き        xxx-1.jpg(組成表)  xxx-2.jpg(続き)

何をするか
    枝番の順に**流し込みを1本の文字列としてつないでから，一度だけ解析する**．
    つなぐことで，次の3つが**自動で**解ける(1 ページずつ解析すると解けない)．

    - 地点の引き継ぎ．`in 5:` は文字列の中に残る
    - ページで割れた種名．`Lycopodium` + `serratum トウゲシバ` が地続きになる
    - 前ページに残った注記の見出し(`Nachweis` だけが前のページにある形)

出力(--out の下)
    comp_table_long_linked.csv   縦持ちの行(`source` は 'once')
    site_notes_linked.csv        地点ごとの調査地・調査年月日・出典
    link_report.tsv              表ごとの枝番・つないだ文字数・警告
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comptea import page_group, parse_text  # noqa: E402

# 流し込みとして読む領域．`crop_cells.py` の REGIONS と同じ
REGIONS = ('once_species',)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='枝番のページを1つの表につなぐ')
    p.add_argument('workdirs', nargs='+', help='置き場か，その親')
    p.add_argument('--out', required=True, help='出力先')
    return p.parse_args(argv)


def collect(workdirs):
    """置き場を集める(親を渡されたら下を拾う)"""
    from pathlib import Path

    out = []
    for w in workdirs:
        p = Path(w)
        if (p / 'located.csv').is_file() or (p / 'continuation.txt').is_file():
            out.append(p)
            continue
        out += [d for d in sorted(p.iterdir())
                if d.is_dir() and ((d / 'located.csv').is_file()
                                   or (d / 'continuation.txt').is_file())]
    return out


def once_text(work):
    """その置き場の流し込みを読んだ文字列(無ければ空)

    表のあるページは `ocred.csv` の `once_species` の行に入っている．
    続きのページは `once_text.txt`(読んだ人が置く)を見る．
    """
    import pandas as pd

    txt = work / 'once_text.txt'
    if txt.is_file():
        return txt.read_text(encoding='utf-8').strip()
    ocred = work / 'ocred.csv'
    if not ocred.is_file():
        return ''
    df = pd.read_csv(ocred)
    if 'obj_name' not in df.columns:
        return ''
    region = df[df['obj_name'].isin(REGIONS)]
    if region.empty:
        return ''
    col = 'corrected' if 'corrected' in df.columns else 'text'
    vals = [v for v in region[col] if isinstance(v, str) and v.strip()]
    return ' '.join(vals).strip()


def note_text(work):
    """その置き場の注記(調査地・調査年月日・出典)"""
    p = work / 'note.txt'
    return p.read_text(encoding='utf-8').strip() if p.is_file() else ''


def link(items):
    """枝番の順に，流し込みと注記をつなぐ

    Returns:
        (つないだ流し込み, つないだ注記)
    """
    once = [t for t in (once_text(w) for _, w in items) if t]
    note = [t for t in (note_text(w) for _, w in items) if t]
    return (' '.join(once), ' '.join(note))


def main(argv=None):
    import pandas as pd

    args = parse_args(argv)
    works = collect(args.workdirs)
    if not works:
        raise SystemExit('置き場が見つからない')
    groups = page_group.group_parts([w.name for w in works])
    by_name = {w.name: w for w in works}

    rows, notes, report = [], [], []
    for table, items in sorted(groups.items()):
        pairs = [(part, by_name[name]) for part, name in items]
        text, note = link(pairs)
        warn = []
        gaps = page_group.missing_parts(items)
        if gaps:
            warn.append(f'枝番の抜け {gaps}')
        if len(items) > 1 and not text:
            warn.append('つないだ流し込みが空(続きのページをまだ読んでいない)')
        # **組成表そのものが 2 ページ以上に渡ることもある**(2026-09-08)．
        # ここがつなぐのは流し込みと注記だけで，**本体の行は page ごとに出る**．
        # 地点の並びが合っているかは人が見るしかないので，知らせる
        grids = [w.name for _, w in pairs if (w / 'located.csv').is_file()]
        if len(grids) > 1:
            warn.append(f'組成表が {len(grids)} ページにある({", ".join(grids)})．'
                        '本体の行はページごとに出る(ここではまとめない)．'
                        '地点の並びが合っているか目で確かめる')
        for r in parse_text.parse_once_species(text):
            cover, _, soc = r.pop('comp_raw').partition(';')
            rows.append({'table': table, 'plot': r['plot'], 'row_no': None,
                         'j_name': r['j_name'], 's_name': r['s_name'],
                         'layer': r['layer'], 'cover': cover,
                         'sociability': soc, 'constancy': r['constancy'] or '',
                         'status': 'Need Check', 'source': 'once',
                         'note': 'つないだ流し込みから'})
        for n in parse_text.parse_site_notes(note):
            notes.append({'table': table, 'plot': n['plot'],
                          'field': n['field'], 'value': n['value']})
        report.append({'table': table,
                       'parts': ','.join(str(p) for p, _ in items),
                       'pages': ','.join(n for _, n in items),
                       'once_chars': len(text), 'note_chars': len(note),
                       'warning': '; '.join(warn)})

    os.makedirs(args.out, exist_ok=True)
    cols = ['table', 'plot', 'row_no', 'j_name', 's_name', 'layer', 'cover',
            'sociability', 'constancy', 'status', 'source', 'note']
    pd.DataFrame(rows, columns=cols).to_csv(
        os.path.join(args.out, 'comp_table_long_linked.csv'),
        index=False, encoding='utf-8-sig')
    pd.DataFrame(notes, columns=['table', 'plot', 'field', 'value']).to_csv(
        os.path.join(args.out, 'site_notes_linked.csv'),
        index=False, encoding='utf-8-sig')
    pd.DataFrame(report).to_csv(
        os.path.join(args.out, 'link_report.tsv'), sep='\t',
        index=False, encoding='utf-8-sig')

    multi = [r for r in report if ',' in r['parts']]
    print(f'表 {len(report)}(うち枝番でつないだ表 {len(multi)})')
    print(f'縦持ち {len(rows)} 行 / 注記 {len(notes)} 件')
    for r in report:
        if r['warning']:
            print(f'  ! {r["table"]}: {r["warning"]}')
    print(f'書いた: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
