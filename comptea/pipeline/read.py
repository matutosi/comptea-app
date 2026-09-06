"""段階2の下ごしらえ: セルを読み，辞書と規則で補正する

    python run_ocr.py WORKDIR [--reader easyocr|ai|both]

読み方は3つ．**どれを選んでも補正は同じ**(`correct_text.correct_cell()`)．
読み方だけを差し替え，判定の規則は動かさない．誰が読んだかは `read_by` に残る．

    easyocr (既定)  EasyOCR で全セルを読む．安く，再現性がある
    ai              EasyOCR を使わず，**全セルを AI が読む**ものとして段階2へ回す
                    (古い印刷で EasyOCR が崩れる資料向け)
    both            EasyOCR で読んだうえで全セルを AI にも回し，
                    **補正後の値が食い違ったセルだけ**を目視に残す(費用は倍)

出力
    ocred.csv     located.csv に text / corrected / status を足したもの
    review.tsv    目視に回すセルの一覧(cell_id 付き．これを見て次を決める)

**読めなかったセルも行として残す**．cell_id が飛ぶと段階2で指せなくなるため．
`ai` と `both` の次の手順は `crop_cells.py` → 目で読む → `apply_text.py --by ai`．
"""
import argparse
import sys

from . import common as _common

# 目視に回す理由．cell_id とともに review.tsv に書く
REASONS = {
    'empty': '読めなかった(インクはあるのに文字が取れていない)',
    'need_check': '値として読めない形',
    'suggested': '辞書から直したが候補が1つ(別種に化けうる)',
    'multi': '候補が複数',
    'region': '文章として読む領域(表頭・1回出現種)',
    'note': '位置決めで気になった点がある',
    'unread': 'まだ読んでいない(--reader ai)',
    'crosscheck': 'EasyOCR と AI の両方で読む(--reader both)',
}
# 文章として読む領域．格子に切れないので，まとめて読んで構文解析する
REGIONS = ('header', 'once_species')
# ai / both で AI に回すクラス(文章の領域は元から回している)
READ_CLASSES = ('comp', 'species_col', 'sname', 'layer', 'plot_row', 'header_col')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='OCRと補正(段階2の下ごしらえ)')
    p.add_argument('workdir', help='run_pipeline.py が作った作業ディレクトリ')
    p.add_argument('--only', default=None,
                   help='読み直すクラスを絞る(例 comp,species_col)')
    p.add_argument('--reader', default='easyocr',
                   choices=['easyocr', 'ai', 'both'],
                   help='読み方(既定 easyocr)．ai と both は段階2で AI が読む')
    return p.parse_args(argv)


def ink_ratio(read):
    """セルごとのインクの割合(黒画素 / 面積)

    二値化は**画像ぜんたいで1回**行う．セルごとに大津の方法をかけると，
    白紙のセルでも雑音が半分に割れて「インクがある」ことになってしまう．
    """
    import cv2
    import numpy as np

    ratios = {}
    for image_path, group in read.groupby('source_image'):
        gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = bw.shape
        for _, row in group.iterrows():
            x1, y1 = max(0, int(row['x1'])), max(0, int(row['y1']))
            x2, y2 = min(w, int(row['x2'])), min(h, int(row['y2']))
            if x2 <= x1 or y2 <= y1:
                continue
            cell = bw[y1:y2, x1:x2]
            ratios[row['cell_id']] = float(cell.sum()) / 255.0 / cell.size
    return ratios


def suspect_empty(read, ratios):
    """読めなかった comp セルのうち，中身がありそうなものの cell_id

    組成部は**非出現('・')のセルが多数**で，そこは文字が取れないのが普通．
    全部を目視に回すと本当に見たいものが埋もれる．
    そこで，点だけのセルの標準(中央値)と比べて明らかに濃いものだけ挙げる．
    """
    import statistics

    # NaN を str() すると 'nan' になる．文字列かどうかで見る
    empty = [r['cell_id'] for _, r in read.iterrows()
             if r['obj_name'] == 'comp' and r['cell_id'] in ratios
             and not (isinstance(r.get('text'), str) and r['text'].strip())]
    if len(empty) < 4:
        return set(empty)
    base = statistics.median(ratios[c] for c in empty)
    limit = max(3 * base, 0.01)
    return {c for c in empty if ratios[c] > limit}


def reasons_for(row, suspect, reader='easyocr'):
    """このセルを目視に回すか．回すなら理由を並べる"""
    out = []
    text = row.get('text')
    has_text = isinstance(text, str) and text.strip()
    if reader in ('ai', 'both') and row['obj_name'] in READ_CLASSES:
        # 読み方の指定で，AI に全セルを回す(非出現のセルも含める．
        # 「点しか無い」ことを確かめるのも読みのうち)
        out.append('unread' if reader == 'ai' else 'crosscheck')
    if row['obj_name'] in REGIONS:
        out.append('region')
    elif not has_text:
        # 組成部の空セルは非出現が普通なので，濃いものだけを挙げる
        if row['obj_name'] != 'comp' or row['cell_id'] in suspect:
            out.append('empty')
    status = row.get('status')
    if status in ('Need Check', 'suggested', 'multi'):
        out.append({'Need Check': 'need_check'}.get(status, status))
    note = row.get('note')
    if isinstance(note, str) and note.strip():
        out.append('note')
    return out


def main(argv=None):
    args = parse_args(argv)
    work, = _common.setup([args.workdir])
    from pathlib import Path
    work = Path(work)
    _common.need_file(work / 'located.csv', 'run_pipeline.py')

    import pandas as pd
    from comptea import correct_text
    if args.reader != 'ai':
        from comptea import ocr  # import に時間がかかる(EasyOCR のモデルを読む)

    df = pd.read_csv(work / 'located.csv')
    # --only で読み直すときは，前の結果を残したまま該当セルだけ差し替える
    base = None
    if args.only and (work / 'ocred.csv').is_file():
        base = pd.read_csv(work / 'ocred.csv')
    if args.only:
        keep = set(args.only.split(','))
        target = df[df['obj_name'].isin(keep)].copy()
    else:
        target = df.copy()

    if args.reader == 'ai':
        # EasyOCR は動かさない．全セルを「まだ読んでいない」状態で並べ，段階2へ回す
        print(f'読み方 ai: {len(target)} セルを AI に回す(EasyOCR は動かさない)')
        read = target.copy().sort_values('cell_id')
        read['text'] = None
    else:
        print(f'OCR: {len(target)} セル ...')
        read = []
        for _, group in target.groupby('source_image'):
            read.append(ocr.ocr_images_df(group.copy()))
        read = pd.concat(read).sort_values('cell_id')
        read = read.drop(columns=['img_base64'], errors='ignore')
    if 'text' not in read.columns:
        read['text'] = None

    for i, row in read.iterrows():
        fixed = correct_text.correct_cell(row['obj_name'], row['text'])
        read.loc[i, 'corrected'] = fixed['corrected']
        read.loc[i, 'status'] = fixed['status']
    # AI が読み直したセルを後から見分けられるようにする
    read['read_by'] = '(未読)' if args.reader == 'ai' else 'easyocr'
    if args.reader == 'both':
        # AI の読みで上書きされても比べられるよう，EasyOCR の結果を控えておく
        read['text_easyocr'] = read['text']
        read['corrected_easyocr'] = read['corrected']
    if base is not None:
        read = (pd.concat([base[~base['cell_id'].isin(read['cell_id'])], read])
                  .sort_values('cell_id'))
    read.to_csv(work / 'ocred.csv', index=False)

    print('--- クラスごと ---')
    for name, g in read.groupby('obj_name'):
        empty = int((g['text'].fillna('').astype(str).str.strip() == '').sum())
        print(f'  {name:<14} {len(g):>4} セル   読めなかった {empty}')
    print('--- status ---')
    for status, n in read['status'].value_counts(dropna=False).items():
        print(f'  {str(status):<14} {n:>4}')

    ratios = ink_ratio(read)
    suspect = suspect_empty(read, ratios)
    rows = []
    for _, row in read.iterrows():
        rs = reasons_for(row, suspect, args.reader)
        if rs:
            rows.append({
                'cell_id': row['cell_id'], 'obj_name': row['obj_name'],
                'block': row.get('block'), 'row': row.get('row'), 'col': row.get('col'),
                'text': row.get('text'), 'corrected': row.get('corrected'),
                'reason': ','.join(rs),
            })
    review = pd.DataFrame(rows)
    review.to_csv(work / 'review.tsv', sep='\t', index=False)
    print(f'--- 目視に回す: {len(review)} セル ---')
    for reason, n in (review['reason'].value_counts().items() if len(review) else []):
        print(f'  {reason:<24} {n:>4}')
    for key, text in REASONS.items():
        if len(review) and review['reason'].str.contains(key).any():
            print(f'  * {key}: {text}')

    print(f'\n書いた: {work / "ocred.csv"} / {work / "review.tsv"}')
    print('次: crop_cells.py で切り出して目で読む(段階2)')
    if args.reader in ('ai', 'both'):
        print('  python crop_cells.py {} --what review'.format(work))
        print('  読んだ結果を TSV にして apply_text.py {} --tsv fixes.tsv --by ai'
              .format(work))
        if args.reader == 'both':
            print('  apply_text.py が EasyOCR の読みと突き合わせ，'
                  '補正後の値が食い違ったセルを挙げる')




if __name__ == '__main__':      # python -m comptea.pipeline.<段>
    sys.exit(main())
