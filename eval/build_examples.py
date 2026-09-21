"""見本の zip (examples/*.zip) を，作業ディレクトリから作り直す

    py -3.12 eval/build_examples.py <作業ディレクトリ>

手元のパスは入れない (`model` は名前だけ，`source_image` は画像の名前だけ)．
"""
import os
import re
import sys
import zipfile

PAT = re.compile(r'[A-Za-z]:[\\/][^,\t\r\n"]*')
SETS = {'examples/sample_grid.zip': ('located.csv', 'detect.csv', 'summary.txt'),
        'examples/sample_read.zip': ('ocred.csv', 'plot_table.csv', 'review.tsv',
                                     'located.csv', 'detect.csv')}


def leaf(m):
    return m.group(0).replace(chr(92), '/').rsplit('/', 1)[-1]


def main():
    work = sys.argv[1]
    for name, members in SETS.items():
        with zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED) as z:
            for m in members:
                text = open(os.path.join(work, m), encoding='utf-8-sig').read()
                new, n = PAT.subn(leaf, text)
                print(f'{name} {m}: パスを {n} か所直した')
                z.writestr(m, new)


if __name__ == '__main__':
    main()
