"""表の下の注記から，地点ごとの調査地・調査年月日・出典を取る (site_notes.py)

未着手だった課題「**表の下の地点情報を読む**」(2026-09-08 ユーザ指示) のための段．
表頭に無い地点の情報は，**表の下の注記**にあります．注記は文章なので格子に切れず，
`comptea.yomi` などのレイアウト解析で**段落として**取り出します．

    from comptea import site_notes, yomi
    paras = yomi.YomiReader().read_paragraphs(im)      # [{'box', 'text'}]
    recs = site_notes.site_info(paras)                 # 地点ごとの locality/date/…

**レイアウト解析は 1 つの注記を複数の段落に割ります**．s01115_19_p3 の注記は

    「調査地 Lage: Lfd. Nr.1-10: … 神戸市山田町菊水山 Juni 1983), 11-15: Berg Nagam」
    「obanoyama-cho, Stadt Kobe 神戸市篠原伯母野山町長峰山」
    「(25, Juni 1983).」

の 3 段落に割れていました．**続きの段落には見出し語が無い**ので，見出しだけで
選ぶと日付が欠け，地名も途中で切れます．そこで**見出しの段落から始め，すぐ下に
続く段落を，次の見出しか本文に当たるまでつなぎます**．
"""
import os
import re

from . import parse_text, row_kinds

# 注記の見出し (独文と和文．`row_kinds.NOTE_RE` より広く取る)
HEAD = re.compile(r'(Lage\s*d\.|Lage\s*[:：]|Fundorte|Datum\s*d\.|'
                  r'Nachweis\s*d\.|Aufnahme|調査地|調査年月日|既発表|'
                  r'原調査資料|出典)', re.I)
GAP = 1.5               # 続きとみなす段落の間隔 (前の段落の高さの倍数)


def _text(p):
    return (p.get('text') if isinstance(p, dict) else getattr(p, 'text', '')) or ''


def _box(p):
    b = p.get('box') if isinstance(p, dict) else getattr(p, 'box', None)
    return [float(v) for v in (b or (0, 0, 0, 0))]


def is_head(text, kinds=('note',)):
    """その段落が注記の**始まり**か"""
    t = (text or '').strip()
    if not t:
        return False
    if 'note' in kinds and (HEAD.search(t) or row_kinds.kind_of_text(t) == 'note'):
        return True
    if 'once' in kinds and (row_kinds.ONCE_JA.search(t)
                            or row_kinds.kind_of_text(t) == 'once'):
        return True
    return False


def pick(paras, kinds=('note',), gap=GAP):
    """段落の並びから注記を選び，**続きをつないで**返す

    Args:
        paras: [{'box': (x1, y1, x2, y2), 'text': 文字列}]．y の順でなくてよい
        kinds: 'note' (調査地・出典…) と 'once' (出現 1 回の種)
    Returns:
        つないだ注記の文字列のリスト
    """
    items = sorted(((_box(p), _text(p)) for p in (paras or []) if _text(p).strip()),
                   key=lambda z: (z[0][1], z[0][0]))
    out = []
    cur = None
    prev = None
    for b, t in items:
        h = max(1.0, b[3] - b[1])
        near = prev is not None and (b[1] - prev[3]) <= prev_h * gap
        if is_head(t, kinds=kinds):
            if cur is not None:
                out.append(cur)
            cur = t.strip()
        elif cur is not None and near:
            cur += '\n' + t.strip()             # 見出しの無い続き
        elif cur is not None:
            out.append(cur)
            cur = None
        prev, prev_h = b, h
    if cur is not None:
        out.append(cur)
    return out


def site_info(paras, gap=GAP):
    """注記から，地点ごとの調査地・調査年月日・出典を取る

    Returns:
        `parse_text.parse_site_notes` と同じ形
        [{'plot': 地点番号 or None, 'field': 'locality'|'date'|'source_ref',
          'value': 文字列}, ...]
    """
    notes = pick(paras, kinds=('note',), gap=gap)
    if not notes:
        return []
    recs = parse_text.parse_site_notes('\n'.join(notes))
    # **長すぎる日付は日付でない**．「調査年月日」の見出しの下に「出現 1 回の種」の
    # 段落が続く紙面があり (s01115_09_p1)，種の列挙が丸ごと値になっていた．
    recs = [r for r in recs
            if r.get('field') != 'date' or len(r.get('value') or '') <= DATE_MAX]
    return with_dates(recs)


# 日付は「調査地」の文の中に**括弧付きで**書かれるのが普通で，独立した
# 「調査年月日」の見出しは少ない (注記画像 10 枚で `date` は 4 件だけだった)．
#   「Berg Kikusui, Shimada-cho, Stadt Kobe 神戸市山田町菊水山 (23. Juni 1983)」
# 既存の `parse_text.parse_site_notes` は**見出しで場を分ける**作りなので，
# 値に埋もれた日付はここで分ける (既存の解析には手を入れない)．

MONTH = (r'Jan|Feb|M(?:ä|a)r|Apr|Mai|May|Jun|Jul|Aug|Sep|Okt|Oct|Nov|Dez|Dec'
         r'|[1１][0-2０-２]?月|[1-9１-９]月')
# (日) 月 年 / 月 年 / 年 だけ．年は 4 桁か「'83」
_DATE = re.compile(
    r'(?P<date>'
    r'(?:\d{1,2}\s*[.,、]?\s*)?(?:' + MONTH + r')[a-zé]*\.?\s*[,，]?\s*'
    r"(?:\d{4}|'\d{2})"
    r'|(?:\d{1,2}\s*[.,、]?\s*)?(?:' + MONTH + r')[a-zé]*\.?'
    r"|(?:19|20)\d{2}|'\d{2}"
    r')\s*[).．]*\s*$')
DATE_MAX = 24          # 日付はこの字数まで (長いものは種名の列挙などの巻き添え)


def split_date(value):
    """調査地の文を (地名, 日付) に分ける．日付が無ければ (そのまま, None)

    末尾の括弧や「6 Nov. 1973」のような並びを日付とみなします．
    地名の中の数字 (「2 丁目」) は日付にしません — 月か 4 桁の年が要ります．
    """
    s = (value or '').strip()
    if not s:
        return s, None
    body = s.rstrip(' .．,，')
    # 末尾の括弧を先に見る
    m = re.search(r'[(（]\s*(?P<in>[^()（）]*)\s*[)）]\s*[.．]?\s*$', body)
    if m:
        inner = m.group('in').strip()
        if len(inner) <= DATE_MAX and _DATE.search(inner + ' '):
            return body[:m.start()].strip(' .,，'), inner.strip(' .,，')
        # 括弧の中が日付でなければ，**その中は見ない** (中の年号を拾わない)
        body = body[:m.start()].rstrip(' .．,，')
    m = _DATE.search(body)
    if m:
        head = body[:m.start()].strip(' .,，([（')
        if head and len(m.group('date')) <= DATE_MAX:
            return head, m.group('date').strip(' .,，')
    return s.strip(' .,，'), None


def with_dates(recs):
    """地点情報の `locality` から日付を分け，`date` として足す

    **すでにその地点に日付があれば足しません** (見出しから取れたものを優先)．
    """
    has = {r.get('plot') for r in (recs or []) if r.get('field') == 'date'}
    out = []
    for r in (recs or []):
        if r.get('field') != 'locality':
            out.append(r)
            continue
        head, date = split_date(r.get('value'))
        out.append(dict(r, value=head))
        if date and r.get('plot') not in has:
            out.append({'plot': r.get('plot'), 'field': 'date', 'value': date})
            has.add(r.get('plot'))
    return out


# 注記の切り出しの名前．折込は表の画像の隣 (`<画像>_note.png`)，
# 続きのページは置き場の中 (`note_block.png`)
NOTE_SUFFIX = '_note.png'
NOTE_BLOCK = 'note_block.png'
CACHE_SUFFIX = '.layout.txt'


def block_text(work, name=NOTE_BLOCK, kinds=('note',), reader=None,
               use_yomi=True, gap=GAP, cache=True):
    """置き場の切り出し画像をレイアウト解析で読み，選んだ文をつないで返す

    枝番 `-1` のページは注記が次のページ (`-2`) にあり，
    `cli/link_pages.py` がつなぎます．そこが読んでいたのは EasyOCR の
    `note.txt` でした．**切り出しをレイアウト解析で読む方がよく取れます**
    (2026-09-13 の実測．本のページ 80 表で注記が取れた表 68)．

    読みは `<画像>.layout.txt` に残し，2 度目からは読み直しません
    (1 枚 12〜20 秒かかるため)．
    """
    img = os.path.join(str(work), name)
    keep = img + CACHE_SUFFIX
    if cache and os.path.isfile(keep):
        with open(keep, encoding='utf-8') as f:
            return f.read().strip()
    if not os.path.isfile(img):
        return ''
    if reader is None and use_yomi:
        reader = _yomi_or_none()
    if reader is None:
        return ''
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img) as im:
        paras = reader.read_paragraphs(im.convert('RGB'))
    text = '\n'.join(pick(paras, kinds=kinds, gap=gap)).strip()
    if cache:
        with open(keep, 'w', encoding='utf-8') as f:
            f.write(text)
    return text


PAGE_CACHE = 'page_note.layout.txt'


def source_image(work):
    """その置き場の格子が使った画像 (`located.csv` の `source_image`)"""
    f = os.path.join(str(work), 'located.csv')
    if not os.path.isfile(f):
        return None
    import pandas as pd

    try:
        df = pd.read_csv(f, usecols=['source_image'])
    except Exception:
        return None
    got = df['source_image'].dropna()
    return str(got.iloc[0]) if len(got) else None


def work_note_text(work, reader=None, use_yomi=True, gap=GAP, page=True):
    """その置き場の注記の文字列 (切り出しが無ければページ自身を読む)

    枝番 `-2` の注記は**文の途中から始まる続き**のことがあり
    (010-2 は「4-6 : Kumihama-cho …」で始まる)，見出しは表のページ側に
    あります．`link_pages` が両方をつなげるよう，**表のページも読みます**．

    **1 枚に 2 表ある紙面では読みません** (どちらの表の注記か分けられない)．
    """
    got = block_text(work, reader=reader, use_yomi=use_yomi, gap=gap)
    if got or not page:
        return got
    if os.path.isfile(os.path.join(str(work), 'table.txt')):
        return ''
    keep = os.path.join(str(work), PAGE_CACHE)
    if os.path.isfile(keep):
        with open(keep, encoding='utf-8') as f:
            return f.read().strip()
    img = source_image(work)
    if not img or not os.path.isfile(img):
        return ''
    if reader is None and use_yomi:
        reader = _yomi_or_none()
    if reader is None:
        return ''
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(img) as im:
        paras = reader.read_paragraphs(im.convert('RGB'))
    text = '\n'.join(pick(paras, kinds=('note',), gap=gap)).strip()
    with open(keep, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def once_species(paras, gap=GAP):
    """注記から「1 回出現の種」を取る

    折込の注記画像には，地点情報と「1 回出現の種」が並んでいます．

    **同じ (地点, 和名, 階層, 被度) が重なって出る**ので 1 件にします
    (長い文を解析するため．2026-09-13 の実測で 50 件中 10 件・108 件中 10 件)．
    **階層や被度が違えば別の行**として残します
    (同じ種が高木層と低木層に出るのは紙面どおり)．
    """
    out, seen = [], set()
    for text in pick(paras, kinds=('once',), gap=gap):
        try:
            got = parse_text.parse_once_species(text)
        except Exception:
            continue
        for r in got:
            key = (r.get('plot'), r.get('j_name'), r.get('s_name'),
                   r.get('layer'), r.get('comp_raw'))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


def _letters(text):
    return re.sub(r'[^a-z]', '', (text or '').lower())


def _fill_sname(r, status, correct_text):
    """学名を和名から補う (**印字は置き換えない**)

    2026-09-13 の実測 (折込 254 件): 和名が辞書に当たるのに学名が当たらない
    のが 68 件あり，そのすべてで和名から引ける．しかし **58 件は本当に
    別の名前**だった (古い資料の印字と現在の分類の違い)．
    補ってよいのは**印字の読みが引いた名前の頭に収まる**ものだけ．
    """
    j_name = (r.get('j_name') or '').strip()
    printed = (r.get('s_name') or '').strip()
    if not j_name:
        return r
    if printed and 'Need Check' not in status and 'suggested' not in status:
        return r                            # 印字が辞書に当たっている
    name, why = correct_text.sname_from_jname(j_name)
    if why != 'ok':
        return r
    if not printed:
        r['s_name'] = name
        r['note'] = '学名は和名から引いた'
        return r
    a, b = _letters(printed), _letters(name)
    if a and b.startswith(a):
        # **読みが途中で切れている**．同じ名前なのでつなぐ
        r['s_name'] = name
        r['note'] = '学名は和名から補った (読みが途中で切れていた)'
    elif a and not a.startswith(b):
        # **別の名前**．印字を残し，引いた名前は別に控える
        r['s_name_ref'] = name
        r['note'] = '和名から引いた学名と食い違う (印字を残した)'
    return r


def correct_once(recs):
    """「1 回出現の種」の名前を辞書で直す

    2026-09-13 の実測 (折込 10 枚・254 件): **形が整っているのは 241 件 (95%)**
    で，崩れは 13 件しかない．**弱点は形でなく字の読み違い**
    (`ASPlenium oligophlebium`・`Lindera strychnl`)なので，工程の本体と
    同じ `correct_text.correct_name` に通します．

    **印字されている学名は置き換えません** (2026-09-01 の決定)．
    和名から引けるのは**いまの分類の名前**で，古い資料の印字とは食い違う
    (イタドリ: 印字 `Polygonum cuspidatum` / 引くと `Fallopia japonica`)．
    空のときだけ埋め，`note` にそう書きます．
    """
    from . import correct_text

    out = []
    for r in (recs or []):
        r = dict(r)
        status = []
        for key, target in (('j_name', 'j_name'), ('s_name', 's_name')):
            got = correct_text.correct_name(r.get(key), target=target)
            if got:
                r[key] = got['corrected']
                status.append(got['status'])
        r = _fill_sname(r, status, correct_text)
        # **いちばん確かでない方を採る** (どちらかが怪しければ目視に回す)
        for want in ('Need Check', 'suggested', 'OK'):
            if want in status:
                r['status'] = want
                break
        out.append(r)
    return out


# --- 表頭の表へ差し込む ----------------------------------------------------
#
# 表頭から取れる項目は紙面によって欠ける (**表頭の無い表もある**)．
# 注記は**欠けた所を埋めるため**のものなので，**表頭の値は上書きしない**．
# 出典 (`source_ref`) は表頭には無く，注記にしかない．

FIELDS = ('locality', 'date', 'source_ref')


def _blank(v):
    return v is None or v != v or str(v).strip() == ''


def merge_plots(df_plot, recs):
    """注記から取った地点情報を `plot_table` へ差し込む

    Args:
        df_plot: `plot_table.plot_table` の返す表 (`plot` 列が要る)
        recs: `site_info` の返す [{'plot', 'field', 'value'}]
    Returns:
        写しを返す．`attrs['warnings']` に，表に無い地点番号などを入れる
    """
    out = df_plot.copy()
    warns = list(out.attrs.get('warnings', []))
    out.attrs['warnings'] = warns
    if not len(out) or not recs or 'plot' not in out.columns:
        return out

    plots = set(out['plot'])
    unknown = sorted({r.get('plot') for r in recs
                      if r.get('plot') is not None and r.get('plot') not in plots})
    if unknown:
        warns.append(f'注記の地点番号が表に無い: {unknown} (入れなかった)')

    for r in recs:
        field, value = r.get('field'), r.get('value')
        if field not in FIELDS or _blank(value):
            continue
        if field not in out.columns:
            out[field] = None
        p = r.get('plot')
        # **地点の無い注記は全地点に当てる** (1 地点ぶんしか書かれていない紙面)
        rows = out.index if p is None else out.index[out['plot'] == p]
        for i in rows:
            if _blank(out.at[i, field]):
                out.at[i, field] = value
    return out


# --- 工程につなぐ ----------------------------------------------------------
#
# 注記の画像は，表の画像の隣に `<表の画像>_note.png` として置かれる
# (`split_sheet.note_boxes`．**表の画像に含めない**のは，画像が高くなると
# 検出器の入力の縮尺が変わって列や階層の枠が動くため)．
# 続きのページ (枝番 `-2`) では，置き場の `note_block.png`．

def find_note_image(image, work=None):
    """この表の注記の画像．無ければ None"""
    if image:
        stem, _ext = os.path.splitext(str(image))
        p = stem + NOTE_SUFFIX
        if os.path.isfile(p):
            return p
    if work:
        p = os.path.join(str(work), NOTE_BLOCK)
        if os.path.isfile(p):
            return p
    return None


def _yomi_or_none():
    """yomitoku を入れてあれば読み手を返す (入れていなければ None)"""
    try:
        from . import yomi
    except Exception:
        return None
    r = yomi.YomiReader()
    return r if r.available() else None


def read_site_info(image, reader=None, use_yomi=True, gap=GAP):
    """注記の画像を**レイアウト解析で段落として**読み，地点情報にする

    読み手を入れていない環境では `[]` を返します (工程は今までどおり動く)．
    """
    if reader is None and use_yomi:
        reader = _yomi_or_none()
    if reader is None or not image or not os.path.isfile(str(image)):
        return []
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image) as im:
        paras = reader.read_paragraphs(im.convert('RGB'))
    return site_info(paras, gap=gap)


def apply_notes(df_plot, image=None, work=None, reader=None, use_yomi=True,
                page=False):
    """注記を読んで `plot_table` へ差し込む (工程から呼ぶ入口)

    Args:
        page: 切り出した注記が無いとき，**ページ自身**を読むか．
            本のページ (s01114) は注記がページの下にあり切り出されていないので
            これが要る．**1 枚に 2 表ある紙面では，どちらの表の注記か
            分けられない**ので渡さないこと
    Returns:
        (差し込んだ表, 報告の 1 行)．注記が無ければ (元の表, '')
    """
    path = find_note_image(image, work=work)
    if not path and page and image and os.path.isfile(str(image)):
        path = str(image)
    if not path:
        return df_plot, ''
    recs = read_site_info(path, reader=reader, use_yomi=use_yomi)
    if not recs:
        return df_plot, ''
    out = merge_plots(df_plot, recs)
    got = {f: sum(1 for r in recs if r.get('field') == f) for f in FIELDS}
    got = ', '.join(f'{k} {v}' for k, v in got.items() if v)
    return out, f'[注記] {os.path.basename(path)} から {got}'
