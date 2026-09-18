"""読み取った 2 つの CSV (sites.csv・composition.csv) を検算する．

    python scripts/check.py <置き場>          # <置き場>/sites.csv と composition.csv を見る

見るもの (どれも「読み違いの手がかり」で，出たら**画像に戻って確かめる**)．

1. 列がそろっているか
2. 値の形: 被度 (`5 4 3 2 1 + r`)・群度 (`1`〜`5`)・常在度 (`I`〜`V`)・階層・和名
3. 同じ (表・地点・学名・和名・階層) の重複
4. composition の地点が sites にあるか
5. **地点ごとの種数と，表頭の「出現種数」(Artenzahl) が合うか** (いちばん効く検算)．
   種は学名 (無ければ和名) で数え，階層が違っても 1 種と数える．1 回出現の種も含める．
   合わないときは，別の数え方 (種 × 階層，本体だけ，見出し × 種 × 和名) の数も並べる．
   `--list 表:地点` で，その地点で数えた種の一覧を出せる．
   どれかと合えば読みではなく数え方の違い，どれとも合わなければ読み落としか行のずれを疑う
6. 1 回出現の種が，本体の同じ地点に同じ種として出ていないか

終わりに件数を表示する．問題が 0 件でも，読み違いが無いことの証明にはならない
(正しい形になる読み違い，たとえば `+` と `1` の取り違えは形では見つからない)．
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

SITE_COLS = ['file', 'table', 'plot', 'item_ja', 'item_de', 'value', 'unit',
             'value_raw', 'source', 'status', 'note']
COMP_COLS = ['file', 'table', 'plot', 'group_ja', 'group_de', 's_name', 'j_name',
             'layer', 'cover', 'sociability', 'constancy', 'value_raw', 'source',
             'status', 'note']

COVER = re.compile(r'^(?:[54321+r])(?:-(?:[54321+r]))?$')     # 範囲 (+-3) も認める
SOC = re.compile(r'^[1-5](?:-[1-5])?$')
CONST = re.compile(r'^(?:V|IV|III|II|I)$')
LAYER = re.compile(r'^(?:(?:[TBSKH][12]?|M)(?:;(?:[TBSKH][12]?|M))*)$')
JNAME = re.compile(r'^[゠-ヿー・一-鿿\dA-Za-z .()（）の？?]+$')
COUNT_JA = ('出現種数', '種数')
COUNT_DE = ('artenzahl',)


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('dir')
    ap.add_argument('--list', metavar='表:地点',
                    help='その地点で数えた種の一覧を出して終わる (例: Tab.12:5)．出現種数が合わないときに使う')
    a = ap.parse_args(argv)
    sites = read(os.path.join(a.dir, 'sites.csv'))
    comp = read(os.path.join(a.dir, 'composition.csv'))
    probs = []

    if a.list:
        table, plot = a.list.rsplit(':', 1)
        rows = [r for r in comp if r['table'] == table and r['plot'] == plot]
        for i, r in enumerate(sorted(rows, key=lambda r: (r['source'], r['s_name'].lower())), 1):
            print(f"{i:3d} {r['source']:4s} {r['layer']:5s} {r['s_name']} {r['j_name']} "
                  f"{r['value_raw']}  [{r['group_ja']}]")
        return 0

    def bad(kind, msg):
        probs.append((kind, msg))

    # 1. 列
    for name, rows, cols in (('sites', sites, SITE_COLS), ('composition', comp, COMP_COLS)):
        if rows:
            miss = [c for c in cols if c not in rows[0]]
            if miss:
                bad('列', f'{name}.csv に列が無い: {miss}')

    # 2. 値の形
    for i, r in enumerate(comp, 2):
        where = f"composition {i} 行 ({r.get('table')} 地点 {r.get('plot')} {r.get('s_name') or r.get('j_name')})"
        cv, sc, ct, ly = (r.get(k, '').strip() for k in ('cover', 'sociability', 'constancy', 'layer'))
        if not cv and not ct:
            bad('値', f'{where}: 被度も常在度も空 (非出現の行は書かない)')
        if cv and not COVER.match(cv):
            bad('値', f'{where}: 被度 {cv!r} が 5 4 3 2 1 + r でない')
        if sc and not SOC.match(sc):
            bad('値', f'{where}: 群度 {sc!r} が 1〜5 でない')
        if ct and not CONST.match(ct):
            bad('値', f'{where}: 常在度 {ct!r} が I〜V でない')
        if ly and not LAYER.match(ly):
            bad('値', f'{where}: 階層 {ly!r} の形が違う (T1 T2 B1 B2 S K H M．複数は ;)')
        jn = r.get('j_name', '').strip()
        if jn and not JNAME.match(jn):
            bad('値', f'{where}: 和名 {jn!r} にカタカナ・漢字以外の字がある')
        if not r.get('s_name', '').strip() and not jn:
            bad('値', f'{where}: 学名も和名も空')
        if r.get('source') not in ('body', 'once'):
            bad('値', f"{where}: source {r.get('source')!r} は body か once")

    # 3. 重複
    seen = defaultdict(int)
    for r in comp:
        seen[(r['table'], r['plot'], r['s_name'].strip(), r['j_name'].strip(), r['layer'].strip())] += 1
    for k, n in seen.items():
        if n > 1:
            bad('重複', f'{k} が {n} 行')

    # 4. 地点
    plots = {(r['table'], r['plot']) for r in sites if r.get('plot', '').strip()}
    for k in sorted({(r['table'], r['plot']) for r in comp}):
        if k not in plots:
            bad('地点', f'composition の {k} が sites.csv に無い')

    # 5. 出現種数
    counted = defaultdict(set)          # 種 (本体 + 1 回出現)
    by_layer = defaultdict(set)         # 種 × 階層
    body_only = defaultdict(set)        # 種 (本体だけ)
    by_row = defaultdict(set)           # 見出し × 学名 × 和名 (同じ種が 2 つの見出しに載ると 2)
    for r in comp:
        if r.get('constancy', '').strip():
            continue                                  # 常在度の列は地点ではない
        key = (r['table'], r['plot'])
        sp = r['s_name'].strip().lower() or r['j_name'].strip()
        counted[key].add(sp)
        by_layer[key].add((sp, r['layer'].strip()))
        by_row[key].add((r.get('group_ja', ''), sp, r['j_name'].strip()))
        if r.get('source') == 'body':
            body_only[key].add(sp)
    n_ok = n_ng = 0
    for r in sites:
        ja, de = r.get('item_ja', ''), r.get('item_de', '').lower()
        if not (any(w in ja for w in COUNT_JA) or any(w in de for w in COUNT_DE)):
            continue
        try:
            want = int(float(r['value']))
        except ValueError:
            continue
        key = (r['table'], r['plot'])
        got = len(counted.get(key, ()))
        if got == want:
            n_ok += 1
        else:
            n_ng += 1
            alt = (f"種×階層 {len(by_layer.get(key, ()))}・本体だけの種 {len(body_only.get(key, ()))}・"
                   f"見出し×種×和名 {len(by_row.get(key, ()))}")
            bad('種数', f"{r['table']} 地点 {r['plot']}: 出現種数 {want} に対し読んだ種 {got} "
                        f"({got - want:+d})．別の数え方: {alt}")

    # 6. 1 回出現の種が本体にも
    body = {(r['table'], r['plot'], r['s_name'].strip().lower() or r['j_name'].strip())
            for r in comp if r.get('source') == 'body'}
    for r in comp:
        if r.get('source') == 'once':
            k = (r['table'], r['plot'], r['s_name'].strip().lower() or r['j_name'].strip())
            if k in body:
                bad('1回出現', f'{k} が本体にもある')

    for kind, msg in probs:
        print(f'[{kind}] {msg}')
    n_check = sum(1 for r in comp + sites if r.get('status', '').strip() == 'check')
    n_multi = sum(1 for r in comp if r.get('status', '').strip() == 'multi')
    print(f'--- sites {len(sites)} 行・composition {len(comp)} 行 '
          f'(本体 {sum(r.get("source") == "body" for r in comp)}・'
          f'1 回出現 {sum(r.get("source") == "once" for r in comp)})')
    print(f'--- 出現種数の照合: 一致 {n_ok}・不一致 {n_ng}．問題 {len(probs)} 件．'
          f'status=check {n_check} 行・multi {n_multi} 行')
    if n_ng:
        print('    合わない地点は `--list 表:地点` で数えた種の一覧を出し，画像の列と突き合わせる')
    return 1 if probs else 0


if __name__ == '__main__':
    sys.exit(main())
