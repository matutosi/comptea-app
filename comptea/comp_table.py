"""OCRの結果を縦持ち(long format)の表に組み立てる.

入力は ocr_web.py / comptea_web.py のOCRページが作るデータフレーム.
    obj_name  : species_col(和名) / sname(学名) / layer(階層) / comp(被度・群度)
    col, row  : locateのグリッドから振った列番号・行番号
    corrected : 補正後の文字列(無ければ text を使う)

出力は1行が「1地点 × 1種」の縦持ち.
    source_image, plot, row_no, j_name, s_name, layer,
    cover, sociability, comp_raw, status, note

noteはlocate.pyが付ける位置決めの気になる点(interpolated / snapped / on_text).

組成表の並びは「学名・和名・階層の列があり，その右が地点」という前提
(R版 comp_table.R の memo と同じ). 地点番号は comp の列番号を左から1,2,...と振り直す.
"""
import re

import pandas as pd

import correct_text
import parse_text

# OCR結果のobj_nameと，出力する列名の対応
ATTR_COLUMNS = {
    'species_col': 'j_name',
    'sname'      : 's_name',
    'layer'      : 'layer',
}
CLASS_COMP = 'comp'
# 値のあるセルのこの割合が「常在度(被度の範囲)」なら，常在度表とみなして知らせる
CONSTANCY_MIN = 0.2


def split_comp(text, kind=None):
    """セルの文字列を被度と群度に分ける

    「被度・群度」の順で書かれ，correct_comp()で';'区切りに正規化されている前提．
    '+'は'+・1'，'r'は'r・1'の省略形なので，群度1を補う．
    空文字は非出現('・'はcorrect_comp()で除かれるため空になる)．

    値として読めるかの判定は correct_text.validate_comp() に任せる
    (判定のルールを1箇所にまとめるため)．

    Args:
        text: 補正後のセルの文字列．例 '4;2' '5' '+' 'r'
        kind: その列の種類．'constancy'(群落の要約) / 'plot'(単独地点) / None
    Returns:
        (cover, sociability, status)
        cover      : 被度．非出現のときNone
        sociability: 群度．書かれていないときNone
        status     : 'OK' / 'Need Check' / 'absent'(非出現)
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None, None, 'absent'
    text = str(text)
    # 非出現の印だけのセルは空と同じに扱う．
    # correct_comp() は '・' を除いて空文字にするが，**空文字は CSV を往復すると
    # 欠測になり，_text_value() が補正前の text('・') に戻してしまう**．
    # AI がセルを読んで '・' と書いたときに，値として読めない扱いになるのを防ぐ
    # (2026-09-01．--reader ai を通して分かった)．
    if text.strip().strip('・･.-‐') == '':
        return None, None, 'absent'
    # 括弧の付いたセル(`IV(+-3)`・`1(+)`)．**どちらの意味かは列で決まる**
    # (2026-09-05)．`kind='constancy'` なら常在度(括弧の中が被度の範囲，群度は無い)，
    # `kind='plot'` なら単独地点の被度・群度．決まらないときは常在度として扱う
    con = correct_text.correct_constancy(text)
    if con is not None:
        head, inner = con.split('(')[0], con[con.index('(') + 1:-1]
        if kind == 'plot':
            cover = correct_text.ROMAN2DIGIT.get(head, head)
            return cover, inner, 'OK'
        return inner, None, 'OK'
    parts = [p for p in text.split(';') if p != '']
    if not parts:
        return None, None, 'absent'
    cover = parts[0]
    if len(parts) >= 2:
        sociability = parts[1]
    elif cover in ('+', 'r'):
        sociability = '1'  # 省略形なので補う
    else:
        sociability = None
    status = 'OK' if correct_text.validate_comp(text) else 'Need Check'
    return cover, sociability, status


def _text_value(df: pd.DataFrame) -> pd.Series:
    """correctedを使い，無い行はtextで補う"""
    if 'corrected' not in df.columns:
        return df['text'] if 'text' in df.columns else pd.Series([None] * len(df), index=df.index)
    if 'text' not in df.columns:
        return df['corrected']
    return df['corrected'].where(df['corrected'].notna(), df['text'])


def species_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """行番号ごとの和名・学名・階層を取り出す

    Args:
        df: OCR結果．obj_name, row, corrected(またはtext)を持つ
    Returns:
        row_no を索引に持ち，j_name / s_name / layer を列に持つDataFrame
        (検出されなかった列は作られない)
    """
    attrs = df[df['obj_name'].isin(ATTR_COLUMNS)].copy()
    if attrs.empty:
        return pd.DataFrame(columns=['row_no'])
    attrs['value'] = _text_value(attrs)
    # 同じ行に同じobj_nameが複数あるときは左のものを採る
    attrs = attrs.sort_values(['row', 'col']).drop_duplicates(['row', 'obj_name'])
    wide = attrs.pivot(index='row', columns='obj_name', values='value')
    wide = wide.rename(columns=ATTR_COLUMNS).reset_index().rename(columns={'row': 'row_no'})
    wide.columns.name = None
    return wide


HEAD_VOTE_MIN = 3      # 列の種類を決めるのに要る，括弧付きのセルの数


def column_head_kinds(comp, col='col', value='comp_raw', vote_min=HEAD_VOTE_MIN):
    """列ごとに，括弧付きのセルが「常在度」か「単独地点の被度」かを決める

    **1 つの表に両方が混ざる**(2026-09-05 に実物で確認)．s01115_22_p2 は
    25 列のうち 9 列が算用数字の頭(`2(3-4)` = 被度2・群度3-4)，残りが
    ローマ数字の頭(`IV(+-3)` = 常在度IV・被度の範囲 +-3)で，列ごとに
    きれいに分かれる(混ざる列の割合は 0.00-0.09 か 0.88-1.00 の両極)．
    11_p1 も 14 列が 7 列ずつに分かれる．群落の要約列と単独地点の列を
    1 つの表に並べた組み方で，**セルだけを見て意味は決められない**．

    列の多数決で決めるので，1 文字の `1` と `I` の読み違いもここで直る．

    Returns:
        行ごとの 'constancy' / 'plot' / None(票が足りない列)．
        `comp` と同じ索引の Series
    """
    keys = ['block', col] if 'block' in comp.columns else [col]
    kind = pd.Series([None] * len(comp), index=comp.index, dtype=object)
    for _, sub in comp.groupby(keys, dropna=False):
        n_roman = n_digit = 0
        for t in sub[value]:
            if not isinstance(t, str):
                continue
            v = correct_text.correct_constancy(t)
            if v is None:
                continue
            head = v.split('(')[0]
            if head in correct_text.ROMAN2DIGIT:
                n_roman += 1
            elif head.isdigit():
                n_digit += 1
        if n_roman + n_digit < vote_min:
            continue
        kind.loc[sub.index] = 'constancy' if n_roman >= n_digit else 'plot'
    return kind


def constancy_share(values):
    """値のあるセルのうち，常在度(`IV(+-3)`)の割合を返す

    頭が**ローマ数字のものだけ**を数える(算用数字の頭は単独地点の被度)．
    """
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return 0.0
    n = 0
    for v in vals:
        c = correct_text.correct_constancy(v)
        if c and c.split('(')[0] in correct_text.ROMAN2DIGIT:
            n += 1
    return n / len(vals)


def repair_split_values(comp, col_value='comp_raw', col_plot='plot',
                        col_row='row', constancy=False):
    """隣り合うセルにまたがって組まれた値を，行の中で分け直す

    印字が列の境からずれていると，**地点1の値の末尾と地点2の値の先頭が
    くっついて**組まれることがある(`example.jpg` の行18 は `[1・] [11・1]`)．
    切り出しは幾何で決まるので，そのまま切ると `1` と `1;1;1` になる．
    境を動かしても直らない(2026-09-01 に測って確かめた)ので，**読みの側で直す**．

    片方が**3つ以上に割れていて値として読めない**とき，端の1つを隣へ移す．
    **移した結果，両方が値として読めるときだけ**直すので，
    読めるものを壊すことはない．直したセルには note に `moved` を付ける．

    Returns:
        (comp, 直した箇所の数)
    """
    if col_value not in comp.columns or col_plot not in comp.columns:
        return comp, 0
    keys = [k for k in ('source_image', 'block', col_row) if k in comp.columns]
    if not keys:
        return comp, 0
    comp = comp.copy()
    if 'note' not in comp.columns:
        comp['note'] = ''
    n_fixed = 0
    for _, g in comp.groupby(keys, dropna=False):
        g = g.sort_values(col_plot)
        idx = list(g.index)
        for i in range(len(idx) - 1):
            ia, ib = idx[i], idx[i + 1]
            a = str(comp.at[ia, col_value] or '')
            b = str(comp.at[ib, col_value] or '')
            pa = [p for p in a.split(';') if p]
            pb = [p for p in b.split(';') if p]
            # **常在度は繋ぎ直す**(2026-09-04)．`III(+-1)` は 8 文字あるので
            # 列の境で割れやすく，`III(` と `+-1)` のように 2 つのセルに
            # 分かれる(s01115_22_p2 で 15 件)．被度のように「1 文字を隣へ移す」
            # のではなく，**両方を繋いで長い方のセルに入れ，短い方は空にする**
            # **片方でも値として読めるなら繋がない**(2026-09-04 に実物で確かめた)．
            # 「常在度表なら裸の被度は無いはずだから繋いでよい」と緩めたところ，
            # `1(+)` と `2(+)` という**別々の値**を `II(+-1)` に繋いだ
            # (s01115_22_p2 の 15 件のうち少なくとも 3 件が誤り)．
            # 割れた値は片側が `III(` のように**それだけでは値にならない**
            if not (correct_text.validate_comp(a) or correct_text.validate_comp(b)):
                ra, rb = a.replace(';', ''), b.replace(';', '')
                joined = correct_text.correct_constancy(ra + rb)
                # **両側に中身の字がある**ときだけ繋ぐ．片方が括弧だけだと，
                # OCR が落とした分を見落としたまま値らしい形になる
                # (s01115_22_p2 の `4(1` + `}` は，印字 `IV(1-3)` なのに
                #  `IV(1)` になった．2026-09-04 に実物で確かめた)
                if joined and not all(re.search(r'[IVivl|Tt1-5+r]', s) for s in (ra, rb)):
                    joined = None
                if joined:
                    if len(a) >= len(b):
                        comp.at[ia, col_value], comp.at[ib, col_value] = joined, ''
                    else:
                        comp.at[ia, col_value], comp.at[ib, col_value] = '', joined
                    for j in (ia, ib):
                        note = str(comp.at[j, 'note'] or '').strip(';')
                        comp.at[j, 'note'] = (note + ';' if note else '') + 'moved'
                    n_fixed += 1
                    continue
            new_a = new_b = None
            if len(pb) > 2 and len(pa) == 1:
                # 右のセルの先頭が，左のセルの値の続き
                new_a, new_b = ';'.join(pa + pb[:1]), ';'.join(pb[1:])
            elif len(pa) > 2 and len(pb) == 1:
                # 左のセルの末尾が，右のセルの値の先頭
                new_a, new_b = ';'.join(pa[:-1]), ';'.join(pa[-1:] + pb)
            if new_a is None:
                continue
            if not (correct_text.validate_comp(new_a)
                    and correct_text.validate_comp(new_b)):
                continue
            comp.at[ia, col_value] = new_a
            comp.at[ib, col_value] = new_b
            for j in (ia, ib):
                note = str(comp.at[j, 'note'] or '').strip(';')
                comp.at[j, 'note'] = (note + ';' if note else '') + 'moved'
            n_fixed += 1
    return comp, n_fixed


def fill_sname_from_jname(attrs):
    """学名が**空の行だけ**，和名から補う

    **印字されている学名は触らない**(2026-09-01 決定)．引ける学名は
    いまの分類の名前なので，古い資料の印字とは食い違うことがあり，
    置き換えると「印字されていたもの」が分からなくなる．
    空の行(学名が前の行に折り返して印字された行など)を埋めるためだけに使う．

    Returns:
        (attrs, 補った row_no の集合, 決められなかった件数)
    """
    if 'j_name' not in attrs.columns or 's_name' not in attrs.columns:
        return attrs, set(), 0
    filled, n_multi = set(), 0
    for i, r in attrs.iterrows():
        s = r.get('s_name')
        if isinstance(s, str) and s.strip():
            continue
        name, why = correct_text.sname_from_jname(r.get('j_name'))
        if why == 'ok':
            attrs.at[i, 's_name'] = name
            filled.add(r['row_no'])
        elif why == 'multi':
            n_multi += 1
    return attrs, filled, n_multi


def _name_unsure(corrected):
    """種名の補正が当てにならないか

    辞書に無い(Need Check)か，候補が複数ある(';' で並ぶ)とき．
    """
    if not corrected:
        return True
    return (corrected.get('status') == 'Need Check'
            or ';' in str(corrected.get('corrected') or ''))


def _sname_broken(s_name):
    """学名に別の語が混じっていないか

    地点の目印が読めないと(`in6` -> `inG`)，そこが切れずに
    学名の前にくっついてくる．学名に数字やコロンは出ない．
    """
    return bool(s_name) and bool(re.search(r'[0-9:：;]', str(s_name)))


def once_species_rows(df: pd.DataFrame, text_col: str = 'corrected') -> pd.DataFrame:
    """
    表の下の「1回出現種」を，本体と同じ縦持ちの行にする

    1地点にしか出なかった種は，表を小さくするために本体から外され，
    表の下に文章として列挙される(docs/vegetation_science.md 2節)．
    読めば本体と同じ情報なので，同じ形に直して足す．
    出典は `source` 列で区別する．

    種名は OCR の誤りを含むので，本体と同じく辞書との編集距離で直す．
    """
    if text_col not in df.columns:
        text_col = 'text'
    region = df[df['obj_name'] == 'once_species'] if 'obj_name' in df else df.iloc[0:0]
    if region.empty or text_col not in df.columns:
        return pd.DataFrame()
    text = region.iloc[0][text_col]
    if not isinstance(text, str) or not text.strip():
        return pd.DataFrame()
    rows = []
    for r in parse_text.parse_once_species(text):
        cover, sociability, status = split_comp(r['comp_raw'])
        j = correct_text.correct_name(r['j_name'], target='j_name')
        s = correct_text.correct_name(r['s_name'], target='s_name') if r['s_name'] else None
        rows.append({
            'source_image': region.iloc[0].get('source_image'),
            'plot': r['plot'], 'row_no': None,
            'j_name': j['corrected'] if j else r['j_name'],
            's_name': s['corrected'] if s else r['s_name'],
            'layer': r['layer'],
            'cover': cover, 'sociability': sociability,
            'comp_raw': r['comp_raw'],
            # 種名が辞書に無いもの，候補が複数のもの，
            # 学名に別の語が混じったものは目視へ回す
            'status': ('Need Check'
                       if _name_unsure(j) or _sname_broken(r['s_name'])
                       else status),
            'note': '', 'source': 'once',
        })
    out = pd.DataFrame(rows)
    # **まったく同じ記載が 2 度取り込まれることがある**(2026-09-06)．
    # 流し込みを読み順に並べ直す段で，同じ断片を二重に拾う紙面がある
    # (s01115_15_p2 の「地点9 アラカシ K +」など 6 件)．
    # 同じ地点・種・階層・値なら情報は増えないので 1 つにまとめる．
    # 重複の検査が鳴る原因にもなっていた
    if len(out):
        out = out.drop_duplicates(
            subset=['plot', 'j_name', 'layer', 'comp_raw']).reset_index(drop=True)
    return out


def comp_table(df: pd.DataFrame, keep_absent: bool = False) -> pd.DataFrame:
    """OCR結果を縦持ちの表に組み立てる

    Args:
        df: OCR結果．obj_name, col, row, corrected(またはtext)を持つ
        keep_absent: Trueなら非出現のセルも行として残す(既定はFalseで落とす)
    Returns:
        縦持ちのDataFrame．
        組み立てに困ったことは `df.attrs['warnings']` にリストで入る．
    """
    warnings = []
    required = ['obj_name', 'col', 'row']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'必要な列がない: {missing}')

    comp = df[df['obj_name'] == CLASS_COMP].copy()
    if comp.empty:
        warnings.append(
            f"'{CLASS_COMP}' が1件も無いため，組成の値を組み立てられない．"
            'OCRの段階でセルが作られているか確認する．')
        res = pd.DataFrame(columns=[
            'source_image', 'plot', 'row_no', 'j_name', 's_name', 'layer',
            'cover', 'sociability', 'comp_raw', 'status'])
        res.attrs['warnings'] = warnings
        return res

    # 地点番号: compの列番号を左から1,2,...と振り直す．
    # **段ごとに振り直す**．1つの地点の表を左右2段に折り返したページでは，
    # 同じ地点が段ごとに現れるので，段をまたいで通しにすると
    # 別の地点として数えてしまう(kinki_047 は1地点が2地点になっていた)．
    # `number_cells()` も同じ決まりで列に番号を振っている (2026-09-01)．
    if 'block' in comp.columns:
        comp['plot'] = (comp.groupby('block')['col']
                        .rank(method='dense').astype(int))
    else:
        comp['plot'] = comp['col'].rank(method='dense').astype(int)
    comp['comp_raw'] = _text_value(comp)
    # 隣り合うセルにまたがって組まれた値を，行の中で分け直す
    # (常在度表かどうかで繋ぎ方の慎重さを変えるので，先に割合を測る)
    is_constancy = constancy_share(comp['comp_raw']) >= CONSTANCY_MIN
    comp, n_moved = repair_split_values(comp, constancy=is_constancy)
    if n_moved:
        warnings.append(
            f'{n_moved} 箇所は，隣り合うセルにまたがって組まれた値だったので'
            "行の中で分け直した(note に 'moved' を付けてある)．"
            '印字が列の境からずれている行なので，関門3で目を通す．')
    # 括弧付きのセルの意味は列で決まる(常在度か，単独地点の被度・群度か)
    kinds = column_head_kinds(comp)
    split = [split_comp(t, k) for t, k in zip(comp['comp_raw'], kinds)]
    comp['cover'] = [s[0] for s in split]
    comp['sociability'] = [s[1] for s in split]
    comp['status'] = [s[2] for s in split]
    # **常在度の列**(地点でなく群落，セルが「常在度(被度の範囲)」)の判定．
    # 常在度は `constancy` 列に分けて入れ，警告で知らせる(2026-09-04)
    def _constancy(text, kind):
        if kind == 'plot':
            return None
        v = correct_text.correct_constancy(text)
        if not v:
            return None
        head = v.split('(')[0]
        if head.isdigit():
            # 常在度の列の中の算用数字は，ローマ数字の読み違いとみなす
            return correct_text.DIGIT2ROMAN.get(head) if kind == 'constancy' else None
        return head

    con = pd.Series([_constancy(t, k) for t, k in zip(comp['comp_raw'], kinds)],
                    index=comp.index)
    comp['constancy'] = con
    kcol = (comp.assign(_k=kinds).dropna(subset=['_k'])
            .drop_duplicates(['block', 'col'] if 'block' in comp.columns else ['col'])
            ['_k'].value_counts())
    n_con_col, n_col = int(kcol.get('constancy', 0)), int(kcol.sum())
    if n_con_col and n_con_col < n_col:
        warnings.append(
            f'**この表は常在度の列と単独地点の列が混ざっている**'
            f'(括弧付きの列 {n_col} のうち {n_con_col} が常在度)．'
            '常在度の列は `constancy` に常在度・`cover` に被度の範囲が入り，'
            '単独地点の列は `cover`・`sociability` に被度と群度が入る．'
            '列の種類は列ごとの多数決で決めているので，関門3で目を通す．')
    n_con = int(con.notna().sum())
    n_val = int((comp['status'] != 'absent').sum())
    if n_con and n_val and n_con / n_val >= CONSTANCY_MIN:
        warnings.append(
            f'**この表は常在度表とみられる**(値のあるセル {n_val} のうち {n_con} が'
            '「常在度(被度の範囲)」)．**列は地点ではなく群落**なので，'
            '`plot` は群落の並び順，`cover` は被度の**範囲**，'
            '常在度は `constancy` 列に入れてある．地点ごとの被度は得られない．')
    comp = comp.rename(columns={'row': 'row_no'})

    # 行ごとの種名・階層をくっつける
    attrs = species_attrs(df)
    for name in ATTR_COLUMNS.values():
        if name not in attrs.columns:
            warnings.append(f"'{name}' が1件も無いため，この列は空になる．")
            attrs[name] = None
    # 学名が空の行を，和名から補う(印字がある行は触らない)
    attrs, filled, n_multi = fill_sname_from_jname(attrs)
    if filled:
        warnings.append(
            f'学名が空の {len(filled)} 行を，和名から補った(印字は置き換えていない)．'
            "note に 'sname_from_jname' を付けてある．"
            '引ける学名は**いまの分類の名前**なので，古い資料の印字とは'
            '食い違うことがある．')
    if n_multi:
        warnings.append(
            f'学名が空の {n_multi} 行は，和名から引いても学名を1つに決められなかった'
            '(属や種が分かれる和名)．空のままにした．')
    res = comp.merge(attrs, on='row_no', how='left')

    if not keep_absent:
        res = res[res['status'] != 'absent']

    if 'source_image' not in res.columns:
        res['source_image'] = None
    # noteは位置決めで気になった点(locate.py)．そのまま持ち越す
    if 'note' not in res.columns:
        res['note'] = ''
    if filled:
        mark = res['row_no'].isin(filled)
        res.loc[mark, 'note'] = (res.loc[mark, 'note'].fillna('').astype(str)
                                 .str.strip(';')
                                 .apply(lambda v: (v + ';' if v else '')
                                        + 'sname_from_jname'))
    res['source'] = 'body'
    res = res[[
        'source_image', 'plot', 'row_no', 'j_name', 's_name', 'layer',
        'cover', 'sociability', 'constancy', 'comp_raw', 'status', 'note', 'source',
    ]].sort_values(['source_image', 'plot', 'row_no']).reset_index(drop=True)

    # 表の下の「1回出現種」を同じ縦持ちの行として足す
    once = once_species_rows(df)
    if len(once):
        warnings.append(
            f'表の下の「1回出現種」から {len(once)} 行を足した'
            "(source 列が 'once')．")
        res = pd.concat([res, once], ignore_index=True)

    layer = res['layer'].dropna()
    n_layer = sum(not correct_text.validate_layer(str(v)) for v in layer.unique() if v != '')
    if n_layer:
        warnings.append(
            f'階層として読めない値が {n_layer} 種類ある．'
            'OCRページで layer の status を確認する．')

    n_check = (res['status'] == 'Need Check').sum()
    if n_check:
        warnings.append(
            f'被度・群度として読めないセルが {n_check} 件ある．'
            "status が 'Need Check' の行を確認する．")

    n_note = int((res['note'].fillna('') != '').sum())
    if n_note:
        warnings.append(
            f'位置決めで気になった点のあるセルが {n_note} 件ある．'
            "note 列(interpolated / snapped / on_text)を確認する．")
    res.attrs['warnings'] = warnings
    return res


def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """縦持ちを組成表の見た目(地点が列)に戻す

    確認用．出力の正は縦持ちの方とする．
    """
    if df_long.empty:
        return df_long
    value = df_long['cover'].fillna('')
    sociability = df_long['sociability'].fillna('')
    df = df_long.assign(
        value=[c if s in ('', '1') else f'{c}・{s}' for c, s in zip(value, sociability)])
    return df.pivot_table(
        index=['row_no', 'j_name', 's_name', 'layer'],
        columns='plot', values='value', aggfunc='first').reset_index()


if __name__ == '__main__':
    df = pd.read_csv('ocred2.csv', index_col=0)
    long_df = comp_table(df)
    for w in long_df.attrs.get('warnings', []):
        print(f'[警告] {w}')
    print(long_df.head(15).to_string())
    print(f'\n{len(long_df)} 行')
    print(long_df['status'].value_counts().to_dict())
