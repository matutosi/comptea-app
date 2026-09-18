"""転記表 (画像と同じ形に書き写した TSV) から，縦持ちの 2 つの CSV を作る．

    python scripts/to_csv.py <置き場>

<置き場>/transcripts/<表>/ の次のファイルを読み，<置き場>/sites.csv・composition.csv を書く．
書き方は references/transcript-format.md．

    meta.tsv    key<TAB>value (file・table・title_ja・title_raw)
    header.tsv  表頭．item_ja item_de unit <地点1> <地点2> …  (セルは印字のまま)
    body.tsv    本体．kind group_ja group_de s_name j_name layer <地点1> … note [status]
    once.tsv    1 回出現の種．plot s_name j_name layer value note [status]
    notes.tsv   注記．plot item_ja item_de value unit value_raw status note

値の分解 (`+・2` → 被度 `+`・群度 `2`，`+` → 群度 `1` を補う，`IV(+-3)` → 常在度) は
ここで行う．読めない値・末尾に `?` を付けた値は status=check，階層が `S;K` のように
1 つに決まらない行は status=multi にする．
"""
import argparse
import csv
import os
import re
import sys

SEP = r'\s*[・･·•∙.,;:-]\s*'
COV = r'[54321+r]'
ABSENT = {'', '・', '･', '·', '•', '∙', '.', '-', '—', '–'}
ROMAN = r'(?:V|IV|III|II|I)'
FIX = str.maketrans({'＋': '+', '十': '+', '，': ',', '．': '.', '－': '-', '―': '-', '—': '-',
                     '（': '(', '）': ')', '　': ' '})
MONTHS = {'jan': 1, 'feb': 2, 'mär': 3, 'mar': 3, 'märz': 3, 'apr': 4, 'mai': 5, 'may': 5,
          'jun': 6, 'juni': 6, 'jul': 7, 'juli': 7, 'aug': 8, 'sep': 9, 'sept': 9,
          'okt': 10, 'oct': 10, 'nov': 11, 'dez': 12, 'dec': 12}

SITE_COLS = ['file', 'table', 'plot', 'item_ja', 'item_de', 'value', 'unit',
             'value_raw', 'source', 'status', 'note']
COMP_COLS = ['file', 'table', 'plot', 'group_ja', 'group_de', 's_name', 'j_name',
             'layer', 'cover', 'sociability', 'constancy', 'value_raw', 'source',
             'status', 'note']


def parse_value(raw):
    """印字の値を (cover, sociability, constancy, status, note) に分ける．非出現は None"""
    s = raw.translate(FIX).strip()
    unsure = s.endswith('?')
    s = s.rstrip('?').strip()
    if s in ABSENT:
        if unsure:                                    # ? だけ (読めない) か ・? (非出現か怪しい)
            return ('', '', '', 'check', '読めない' if not s else '非出現か自信なし')
        return None
    notes = []
    m = re.fullmatch(r'\((.+)\)', s)
    if m:                                             # (+) 括弧付き
        s = m.group(1).strip()
        notes.append('括弧付き')
    out = None
    m = re.fullmatch(rf'({ROMAN})(?:\s*\(\s*({COV})(?:\s*-\s*({COV}))?\s*\))?', s)
    if m:                                             # 常在度 IV(+-3)
        cov = m.group(2) or ''
        if m.group(3):
            cov += '-' + m.group(3)
        out = (cov, '', m.group(1))
    m = None if out else re.fullmatch(rf'({COV})\s*\(\s*([1-5])\s*-\s*([1-5])\s*\)', s)
    if m:                                             # 2(3-4) 単独地点の群度の範囲
        out = (m.group(1), f'{m.group(2)}-{m.group(3)}', '')
    m = None if out else re.fullmatch(rf'({COV})(?:{SEP}([1-5]))?', s)
    if m:                                             # 4・2 / + / r
        out = (m.group(1), m.group(2) or '1', '')
    if out is None:
        return ('', '', '', 'check', ';'.join(notes + [f'値として読めない: {raw}']))
    return out + ('check' if unsure else 'ok', ';'.join(notes + (['読みに自信なし'] if unsure else [])))


PLOTS_SEEN = set()


def cell_note(note, plot):
    """行の note のうち，その地点に掛かるものだけを返す．`2: 下線` は地点 2 だけ"""
    out = []
    for part in (x.strip() for x in note.split(';')):
        m = re.match(r'^([^:：\s]+)\s*[:：]\s*(.*)$', part)
        if m and m.group(1) in PLOTS_SEEN:
            if m.group(1) == plot:
                out.append(m.group(2))
        elif part:
            out.append(part)
    return ';'.join(out)


def merge(status, note, row, plot=None):
    """行の status 列・note (地点の指定つき)・複数の階層を，セルの status と note に重ねる"""
    extra = cell_note(row.get('note', ''), plot) if plot is not None else row.get('note', '')
    notes = [x for x in (note, extra) if x]
    if row.get('status', '').strip() == 'check' or extra.startswith('check'):
        status = 'check'
    if ';' in row.get('layer', '') and status == 'ok':
        status = 'multi'                              # 値はあるが階層が 1 つに決まらない
    return status, ';'.join(notes)


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.reader(f, delimiter='\t') if r and not r[0].startswith('#')]
    if not rows:
        return []
    head = [h.strip() for h in rows[0]]
    return [dict(zip(head, [c.strip() for c in r] + [''] * (len(head) - len(r)))) for r in rows[1:]]


def plots_of(head, fixed):
    return [h for h in head if h not in fixed and h]


def norm_site(item_ja, item_de, raw, year=None):
    """表頭の値を整える．(value, status, note)"""
    s = raw.translate(FIX).strip()
    unsure = s.endswith('?')
    s = s.rstrip('?').strip()
    st = 'check' if unsure else 'ok'
    note = '読みに自信なし' if unsure else ''
    if s in ('-', '—', '–'):
        return '-', st, note
    key = item_ja + ' ' + item_de.lower()
    if '年月日' in key or '月日' in key or 'datum' in key:
        d = norm_date(s, year)
        if d:
            return d, st, note
        return s, 'check', '日付として読めない'
    if '調査番号' in key or 'feld' in key:
        return re.sub(r'\s+', '-', s), st, note
    if re.fullmatch(r'\d+\s*[.・･·•]\s*\d+', s):     # 0. 5 / 1・8 (小数点が中黒の印字)
        return re.sub(r'\s*[.・･·•]\s*', '.', s), st, note
    return s, st, note


def norm_date(s, year=None):
    s = s.strip().rstrip('.')
    m = re.search(r'(\d{1,2})\.?\s*([A-Za-zäÄ]+)\.?\s*(\d{4})', s)           # 6. Juli 1983
    if m:
        w = m.group(2).lower()
        mon = MONTHS.get(w) or MONTHS.get(w[:4]) or MONTHS.get(w[:3])
        if mon:
            return f'{int(m.group(3)):04d}-{mon:02d}-{int(m.group(1)):02d}'
    m = re.search(r'(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})', s)
    if m:
        return f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'
    nums = re.findall(r"'?\d+", s)
    if len(nums) == 3:                                # '83 8 3
        y, mo, d = nums
        y = int(y.lstrip("'"))
        y = y + 1900 if y < 100 else y
        return f'{y:04d}-{int(mo):02d}-{int(d):02d}'
    if len(nums) == 2 and year:                       # 年は見出しにある (Datum d. Aufn. (1983))
        return f'{int(year):04d}-{int(nums[0]):02d}-{int(nums[1]):02d}'
    return None


def build(tdir):
    meta = {r['key']: r['value'] for r in read_tsv(os.path.join(tdir, 'meta.tsv'))}
    file, table = meta.get('file', ''), meta.get('table', os.path.basename(tdir))
    sites, comp = [], []
    if meta.get('title_ja') or meta.get('title_raw'):
        sites.append(dict(file=file, table=table, plot='', item_ja='表題', item_de='Titel',
                          value=meta.get('title_ja', ''), unit='', value_raw=meta.get('title_raw', ''),
                          source='title', status='ok', note=''))

    header = read_tsv(os.path.join(tdir, 'header.tsv'))
    body = read_tsv(os.path.join(tdir, 'body.tsv'))
    fixed_head = {'item_ja', 'item_de', 'unit', 'note', 'status'}
    fixed_body = {'kind', 'group_ja', 'group_de', 's_name', 'j_name', 'layer', 'note', 'status'}
    all_plots = []
    for rows, fixed in ((header, fixed_head), (body, fixed_body)):
        if rows:
            all_plots += [p for p in plots_of(list(rows[0].keys()), fixed) if p not in all_plots]
    PLOTS_SEEN.clear()
    PLOTS_SEEN.update(all_plots)
    if header:
        plots = plots_of(list(header[0].keys()), fixed_head)
        for r in header:
            ym = re.search(r'(19|20)\d\d', r.get('item_de', '') + r.get('item_ja', ''))
            for p in plots:
                raw = r.get(p, '')
                if raw == '':
                    continue
                v, st, note = norm_site(r['item_ja'], r['item_de'], raw, ym and ym.group(0))
                st, note = merge(st, note, r, p)
                sites.append(dict(file=file, table=table, plot=p, item_ja=r['item_ja'],
                                  item_de=r['item_de'], value=v, unit=r.get('unit', ''),
                                  value_raw=raw, source='header', status=st, note=note))

    for r in read_tsv(os.path.join(tdir, 'notes.tsv')):
        # plot が空 = 全地点に共通．地点ごとに展開する (地点が分からなければ空のまま)
        for p in ([r.get('plot', '')] if r.get('plot', '') or not all_plots else all_plots):
            sites.append(dict(file=file, table=table, plot=p, item_ja=r.get('item_ja', ''),
                              item_de=r.get('item_de', ''), value=r.get('value', ''),
                              unit=r.get('unit', ''), value_raw=r.get('value_raw', ''),
                              source='note', status=r.get('status') or 'ok', note=r.get('note', '')))

    if body:
        plots = plots_of(list(body[0].keys()), fixed_body)
        g_ja = g_de = s_prev = j_prev = ''
        for r in body:
            if r.get('kind') == 'group':
                g_ja, g_de = r.get('group_ja', ''), r.get('group_de', '')
                continue
            s, j = r.get('s_name', ''), r.get('j_name', '')
            if not s and not j:                       # 階層ごとの続きの行
                s, j = s_prev, j_prev
            s_prev, j_prev = s, j
            for p in plots:
                pv = parse_value(r.get(p, ''))
                if pv is None:
                    continue
                cov, soc, const, st, note = pv
                st, note = merge(st, note, r, p)
                comp.append(dict(file=file, table=table, plot=p, group_ja=g_ja, group_de=g_de,
                                 s_name=s, j_name=j, layer=r.get('layer', ''), cover=cov,
                                 sociability=soc, constancy=const, value_raw=r.get(p, ''),
                                 source='body', status=st, note=note))

    for r in read_tsv(os.path.join(tdir, 'once.tsv')):
        pv = parse_value(r.get('value', '')) or ('', '', '', 'check', '値が空')
        cov, soc, const, st, note = pv
        st, note = merge(st, note, r)
        comp.append(dict(file=file, table=table, plot=r.get('plot', ''), group_ja='', group_de='',
                         s_name=r.get('s_name', ''), j_name=r.get('j_name', ''),
                         layer=r.get('layer', ''), cover=cov, sociability=soc, constancy=const,
                         value_raw=r.get('value', ''), source='once', status=st, note=note))
    return sites, comp


def write(path, cols, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dir')
    a = ap.parse_args(argv)
    root = os.path.join(a.dir, 'transcripts')
    sites, comp = [], []
    for name in sorted(os.listdir(root)):
        tdir = os.path.join(root, name)
        if os.path.isdir(tdir):
            s, c = build(tdir)
            sites += s
            comp += c
            print(f'{name}: sites {len(s)} 行・composition {len(c)} 行')
    write(os.path.join(a.dir, 'sites.csv'), SITE_COLS, sites)
    write(os.path.join(a.dir, 'composition.csv'), COMP_COLS, comp)
    print(f'計: sites {len(sites)} 行・composition {len(comp)} 行 → {a.dir}')


if __name__ == '__main__':
    main()
