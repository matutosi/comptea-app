# -*- coding: utf-8 -*-
"""維管束植物和名チェックリスト ver. 1.10 から，名前の辞書を作る

    py -3.12 download_species_names.py [--src <xlsx>]

書き出すもの(いずれも yolo/ に置く)．
    j_name.txt   和名の一覧
    s_name.txt   学名の一覧
    js_name.txt  **和名 -> 学名 の対応**(タブ区切り．学名が複数なら ';' で連ねる)

データ：山ノ内 崇志・首藤 光太郎・大澤 剛士・米倉 浩司・加藤 将・志賀 隆. 2019.
「維管束植物和名チェックリスト」
(https://gbif.jp/activities/checklist/wamei_checklist_110)

**取得は User-Agent を付けて行う**．付けないと 403 で弾かれる(2026-09-01 に確認)．
"""
import argparse
import io
import urllib.request

import pandas as pd

URL = ('https://gbif.jp/activities/checklist/wamei_checklist_110/excel/'
       'wamei_checklist_ver.1.10.xlsx')
UA = 'Mozilla/5.0 (compatible; comptea/1.0)'


def fetch(src=None):
    """xlsx を読む(src を指定すれば手元のファイルを使う)"""
    if src:
        return pd.ExcelFile(src)
    req = urllib.request.Request(URL, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=180) as r:
        return pd.ExcelFile(io.BytesIO(r.read()))


def write_lines(path, items):
    """**並べてから書く**．集合のまま書くと実行のたびに順序が変わり，
    `correct_name()` の同点候補の優先順(辞書の並び順)まで変わってしまう
    (2026-09-01)．
    """
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        for item in sorted(items):
            f.write(item + '\n')


def download_species_names(src=None):
    xl = fetch(src)
    df_jp = xl.parse('Hub_data')
    df_sn = xl.parse('JN_dataset')

    write_lines('j_name.txt', set(df_jp['all_name']))
    write_lines('s_name.txt', set(df_sn['scientific name without author']))

    # 和名 -> 学名．別名(another name)からも引けるようにする
    pair = {}
    for jn, an, sn in zip(df_sn['common name'], df_sn['another name'],
                          df_sn['scientific name without author']):
        if not isinstance(sn, str) or not sn.strip():
            continue
        for name in (jn, an):
            if isinstance(name, str) and name.strip():
                pair.setdefault(name.strip(), set()).add(sn.strip())
    with io.open('js_name.txt', 'w', encoding='utf-8', newline='\n') as f:
        for jn in sorted(pair):
            f.write('%s\t%s\n' % (jn, ';'.join(sorted(pair[jn]))))
    print('j_name %d / s_name %d / js_name %d'
          % (len(set(df_jp['all_name'])),
             len(set(df_sn['scientific name without author'])), len(pair)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='名前の辞書を作る')
    p.add_argument('--src', default=None, help='手元の xlsx を使う')
    download_species_names(p.parse_args().src)
