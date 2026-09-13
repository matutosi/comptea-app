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
