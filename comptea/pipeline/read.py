"""段階2の下ごしらえ: セルを読み，辞書と規則で補正する

    python run_ocr.py WORKDIR [--reader easyocr|ai|both]

読み方は3つ．**どれを選んでも補正は同じ**(`correct_text.correct_cell()`)．
読み方だけを差し替え，判定の規則は動かさない．誰が読んだかは `read_by` に残る．

    easyocr (既定)  EasyOCR で全セルを読む．安く，再現性がある
    multi           EasyOCR で読んだうえで，**入っている読み手で領域を読み直す**．
                    クラスごとの順で質の通る読みを採る(`read_region.pick`)．
                    2026-09-12 の実測: 学名の一致が 13 → 24・11 → 31 (真値の 2 表)，
                    辞書に当たるセルが 学名 462 → 548・和名 277 → 428 (12 表)
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
    'short_name': '読みが短いので採らなかった(候補は suggest 列)',
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


# `multi` で読み直すクラス．**領域としてまとまっている所**だけにする．
# 組成 (`comp`) は細かい格子なので，いまのところ対象にしない
BLEND_CLASSES = ('sname', 'species_col', 'header_item', 'header_item_ja')


def usable_readers(readers):
    """入っている読み手だけを返す"""
    return {k: r for k, r in (readers or {}).items()
            if getattr(r, 'available', lambda: True)()}


RETRY_CLASS = 'comp'
# 読み直しに付ける余白 (px)．
#
# **広げる案は，測って取り下げた** (2026-09-14)．セル単位では効くように
# 見える (余白 3 px で 0%・6 px で 14〜16%・10 px で 17〜18% が読める) のに，
# **工程を通すと悪くなる**: 本体の要確認が 17_p1 で 114 → 132，
# 05_p2 で 93 → 98 に増えた．広げた読みは `correct_comp` を通るが**隣の値**
# のことがあり，そこから列の種類の判定 (`column_head_kinds`・
# `constancy_share`) が動いて，他のセルが要確認に回る．
# **セル単位の「読めた」と，工程の要確認の数は別物**．
RETRY_PAD = 3
# **変わらなくなるまで繰り返す巡の上限** (2026-09-15)．
# 1 巡目で要確認でなかったセルが，読み直しで他のセルが直ったあとに
# 要確認として残ることがあるので，変化が無くなるまで回す．
# 2 巡目は要確認のセルだけが対象なので費用は小さい．
# (「17_p1 でだけ 43 セル取りこぼす」の主因は別だった —
# **判定の付いていないセルを対象から外していた**．`retry_targets` で直した)
RETRY_ROUNDS = 3


def retry_targets(df, cls=RETRY_CLASS):
    """読み直す対象のセル (要確認と，**判定の付いていない字のあるセル**)

    `correct_comp` が何も返さない読み (点線・記号だけ) では `status` が空のまま
    残り，**段階 3 で初めて要確認になる** (`comp_table` が `validate_comp` で
    判定し直すため)．要確認だけを選ぶと，そのセルは読み直しの対象から外れていた
    — 17_p1 は段階 2 で 349・段階 3 で 360 と 11 件ずれ，その中身がこれだった
    (2026-09-15 に突き合わせて分かった)．

    **空のセルは混ぜない**．17_p1 は組成 63,232 のうち 59,784 が空で，
    混ぜると読み直しが桁違いに重くなる (空のセルは `retry_empty_comp` の持ち場)．
    """
    import pandas as pd

    if df is None or not len(df) or 'status' not in df.columns:
        return pd.Series(False, index=getattr(df, 'index', None))
    need = df['status'] == 'Need Check'
    if 'text' in df.columns:
        # `astype('string')` は None も float の nan も欠測のまま残す
        # (`astype(str)` だと 'None'・'nan' という**字のある値**になる)
        text = df['text'].astype('string').str.strip()
        need = need | (df['status'].isna() & text.notna() & (text != ''))
    return (df['obj_name'] == cls) & need


def retry_cells(img, df, reader, cls=RETRY_CLASS, pad=RETRY_PAD):
    """**読めなかった組成のセルを，まとめて読み直す**

    2026-09-14 の実測: 読めなかったセルは **NDLOCR-Lite がよく読む**
    (22_p2 で 77%・17_p1 で 53%・05_p2 で 26%)．EasyOCR は 66 個中 5 個，
    yomitoku は 0 個だった．**1 セル 1 画像でまとめて渡す**と 0.6 秒/セル
    (1 セルずつ呼ぶと 15 秒．別プロセスの起動と模型の読み込みのため)．

    領域をまとめて読む形は駄目だった (488 セル中 53 個しか割り当たらない)．

    **読んだ文字列は `text_ndl` に残す** (2026-09-15)．補正で落ちた読みも残すので，
    あとで補正の規則を良くしたときに，**OCR をやり直さずに当て直せる**．
    """
    from comptea import correct_text

    if df is None or not len(df) or reader is None:
        return df
    if not getattr(reader, 'available', lambda: True)():
        return df
    if not hasattr(reader, 'read_crops'):
        return df
    out = df.copy()
    if 'status' not in out.columns:
        return out
    hit = retry_targets(out, cls)
    if not hit.any():
        return out
    sub = out[hit]
    boxes = list(zip(sub['x1'], sub['y1'], sub['x2'], sub['y2']))
    got = reader.read_crops(img, boxes, pad=pad) or []
    if 'text_ndl' not in out.columns:
        out['text_ndl'] = ''
    n = 0
    for i, text in zip(sub.index, got):
        if not text:
            continue
        # **読みそのものを残す** (2026-09-15)．補正で落ちた読みも残す —
        # あとで補正の規則を良くしたとき，OCR をやり直さずに当て直せる
        out.at[i, 'text_ndl'] = str(text)
        fixed = correct_text.correct_comp(str(text))
        if not fixed or fixed.get('status') != 'OK':
            continue
        out.at[i, 'corrected'] = fixed['corrected']
        out.at[i, 'status'] = 'OK'
        now = str(out.at[i, 'note'] or '')
        out.at[i, 'note'] = f'{now};ndl' if now else 'ndl'
        n += 1
    if n:
        print(f'読み直し: 組成の {int(hit.sum())} セルのうち {n} セルが読めた'
              ' (NDLOCR-Lite)')
    return out


def retry_until_stable(img, df, reader, cls=RETRY_CLASS, pad=RETRY_PAD,
                       rounds=RETRY_ROUNDS):
    """**変わらなくなるまで**読み直す (2026-09-15)

    **変化が無くなるまで回す**．2 巡目以降は残った要確認のセルだけが
    対象なので，費用は巡ごとに小さくなる．読み直す対象は `retry_targets`
    (要確認と，判定の付いていない字のあるセル)．

    Returns:
        (格子, 回った巡の数)
    """
    out = df
    for i in range(1, max(1, rounds) + 1):
        before = int(retry_targets(out, cls).sum())
        if not before:
            return out, i - 1
        out = retry_cells(img, out, reader, cls=cls, pad=pad)
        after = int(retry_targets(out, cls).sum())
        if after == before:
            return out, i
    return out, rounds


def recorrect_cells(df, cls=RETRY_CLASS):
    """**保存した読みに，いまの補正を当て直す** (OCR はやり直さない．2026-09-15)

    読み直し (`retry_cells`) の読みは `text_ndl` に残してある．補正の規則を
    良くしたあと，**もう一度 OCR を回さずに**要確認を減らせる
    (1 表あたり 4 分 → 1 秒)．

    当てるのは**要確認のセルだけ**．OK のセルには触らない
    (読めているものを補正の版で揺らさない)．

    Returns:
        (格子, 直した数)
    """
    from comptea import correct_text

    if df is None or not len(df) or 'text_ndl' not in df.columns:
        return df, 0
    out = df.copy()
    hit = ((out['obj_name'] == cls) & (out['status'] == 'Need Check')
           & out['text_ndl'].astype(str).str.strip().ne(''))
    n = 0
    for i in out[hit].index:
        fixed = correct_text.correct_comp(str(out.at[i, 'text_ndl']))
        if not fixed or fixed.get('status') != 'OK':
            continue
        out.at[i, 'corrected'] = fixed['corrected']
        out.at[i, 'status'] = 'OK'
        now = str(out.at[i, 'note'] or '')
        out.at[i, 'note'] = f'{now};ndl' if now else 'ndl'
        n += 1
    return out, n


def blend(img, read, readers, ok=None):
    """EasyOCR の読みに，**領域を読み直した結果**を重ねる

    クラスごとの順 (`read_region.ORDER`) で，質の通る読みを採ります．
    どこから採ったかは `read_by` に残します．
    """
    from comptea import read_region
    out = read.copy()
    if 'read_by' not in out.columns:
        out['read_by'] = 'easy'
    use = usable_readers(readers)
    if not use:
        return out
    for cls in BLEND_CLASSES:
        cells = out[out['obj_name'] == cls]
        if cells.empty:
            continue
        maps = {'easy': {c.cell_id: c.text for c in cells.itertuples()
                         if isinstance(c.text, str) and c.text.strip()}}
        for tag, r in use.items():
            # **領域が大きければ分割して読む** (折込をそのまま渡すと縮小されて
            # 読みが崩れる．7012x9214 は辞書に当たる和名が 1 個だった)
            maps[tag] = read_region.read_cells(img, cells, r)
        got, src = read_region.pick(maps, cls, ok=ok, with_source=True)
        for cid, text in got.items():
            m = out['cell_id'] == cid
            out.loc[m, 'text'] = text
            out.loc[m, 'read_by'] = src.get(cid, 'easy')
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='OCRと補正(段階2の下ごしらえ)')
    p.add_argument('workdir', help='run_pipeline.py が作った作業ディレクトリ')
    p.add_argument('--only', default=None,
                   help='読み直すクラスを絞る(例 comp,species_col)')
    p.add_argument('--no-retry', action='store_true',
                   help='読めなかった組成のセルを読み直さない (既定は読み直す)')
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'],
                   help='読み手を動かす装置(既定は環境から決める)．'
                        'GPU が無ければ cpu')
    p.add_argument('--reader', default='easyocr',
                   choices=['easyocr', 'multi', 'ai', 'both'],
                   help='読み方(既定 easyocr)．multi は他の読み手も使う．'
                        'ai と both は段階2で AI が読む')
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
    if row.get('suggest'):
        # 短くて採らなかった読み．候補は出ているので，見れば早く決まる
        out.append('short_name')
    status = row.get('status')
    if status in ('Need Check', 'suggested', 'multi'):
        out.append({'Need Check': 'need_check'}.get(status, status))
    note = row.get('note')
    if isinstance(note, str) and note.strip():
        out.append('note')
    return out


def push_left_rule(df, work):
    """**組成部の左端の列を，左の縦罫線より右へ押し出す** (切り出しだけ．2026-09-16)

    格子 (`located.csv`) はそのまま残し，読む箱だけを動かす
    (理由は `comptea/left_rule.py`)．罫線が当たらない表では何もしない．
    """
    from comptea import ink, left_rule, source

    comp = df[df['obj_name'] == 'comp']
    if comp.empty or 'col' not in comp.columns:
        return df
    try:
        img = source.open_image(df, workdir=work)
    except FileNotFoundError:
        return df
    dark = ink.binarize(img)
    rule = left_rule.fit_rule(dark, comp)
    if rule is None:
        return df
    out, moved = left_rule.push_first_column(df, rule, dark=dark)
    if moved:
        print(f'組成部の左端の列 {moved} セルの左の境を，縦罫線より右へ押し出した'
              f'(罫線 x = {rule[0] * 1000:+.1f}/1000・y + {rule[1]:.0f})．'
              '格子のファイルは変えず，読む箱だけを動かす')
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
    df = push_left_rule(df, work)
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

    # **入っている読み手で領域を読み直す** (`--reader multi`)．
    # 読み手が無ければ何も起きないので，入れていない環境でも動く
    if args.reader == 'multi':
        from PIL import Image
        from comptea import correct_text as _ct
        from comptea import ndl, source, yomi
        readers = usable_readers({'yomi': yomi.YomiReader(device=args.device),
                                  'ndl': ndl.NdlReader(device=args.device)})
        if not readers:
            print('読み方 multi: 他の読み手が入っていないので EasyOCR のまま')
        else:
            print(f'読み方 multi: {sorted(readers)} で領域を読み直す')
            bad = ('Need Check', 'multi', 'suggested')

            def _ok(t, cls=None):
                return _ct.correct_cell(cls, t).get('status') not in bad

            done = []
            for image_path, group in read.groupby('source_image'):
                try:
                    img = Image.open(image_path)
                except Exception:                       # noqa: BLE001
                    done.append(group)
                    continue
                done.append(blend(img, group, readers,
                                  ok=lambda t, c=None: True))
            read = pd.concat(done).sort_values('cell_id')

    # 短い読みは，候補があっても採らずに印字を残す(correct_text.MIN_ADOPT_LEN)．
    # その候補は `suggest` に入って来るので，目視のために持ち回る
    read['suggest'] = None
    for i, row in read.iterrows():
        fixed = correct_text.correct_cell(row['obj_name'], row['text'])
        read.loc[i, 'corrected'] = fixed['corrected']
        read.loc[i, 'status'] = fixed['status']
        if fixed.get('suggest'):
            read.loc[i, 'suggest'] = fixed['suggest']
    # **読めなかった組成のセルを読み直す** (NDLOCR-Lite)．
    # 2026-09-14 の実測で，EasyOCR が読めなかったセルの 26〜77% が読める
    if not args.no_retry:
        from comptea import ndl, source as _source

        rd = ndl.NdlReader(device=args.device)
        if rd.available():
            try:
                _img = _source.open_image(read, workdir=work)
            except Exception:
                _img = None
            if _img is not None:
                read, _rounds = retry_until_stable(_img, read, rd)
    # AI が読み直したセルを後から見分けられるようにする
    if 'read_by' not in read.columns:
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
                'suggest': row.get('suggest'),
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
