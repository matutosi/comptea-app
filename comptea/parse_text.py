"""流し込みの文章になっている部分を読み，構造化する

対象は2つ．どちらも行・列に切れないので，領域を丸ごとOCRしてから解析する．
    header(文章形式)  1地点の表の表頭．「調査番号：SO-216, 調査面積：100m²…」
    once_species      表の下の「出現1回の種」の列挙

OCRは断片で返るので，まず**読み順につなぐ**(reading_order)．
項目や種名が行をまたぐことがあるため，つないでから解析する．
"""
import re
import unicodedata

import numpy as np


def reading_order(results, line_tol: float = 0.6):
    """
    EasyOCRの結果を読み順につなぐ

    断片は箱ごとに返るので，縦の重なりで行にまとめ，行の中はxの順に並べる．
    行のあいだは空白でつなぐ(項目や種名が行をまたぐため)．

    Args:
        results: reader.readtext() の戻り値 [(box, text, conf), ...]
        line_tol: 同じ行とみなす縦のずれ．文字の高さに対する割合
    Returns:
        つないだ文字列
    """
    return ' '.join(ln['text'] for ln in group_lines(results, line_tol))


def group_lines(results, line_tol: float = 0.6):
    """EasyOCRの断片を1行ずつにまとめる

    `reading_order()` が中でつないでいた処理を，**行のまま**取り出せるように
    切り出したもの．表の載っていないページから「1回出現種」の塊だけを
    抜き出すには，行ごとの文字と位置の両方が要る(2026-09-07)．

    Args:
        results: reader.readtext() の戻り値 [(box, text, conf), ...]
        line_tol: 同じ行とみなす縦のずれ．文字の高さに対する割合
    Returns:
        [{'text': 行の文字, 'y1': 上, 'y2': 下, 'x1': 左, 'x2': 右}, ...]
        上から順に並ぶ
    """
    items = []
    for box, text, *_ in results:
        if not text:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), max(ys), min(xs), max(xs), text))
    if not items:
        return []
    heights = [b - a for a, b, _, _, _ in items]
    tol = float(np.median(heights)) * line_tol
    items.sort(key=lambda t: (t[0], t[2]))
    lines, current, base = [], [items[0]], items[0][0]
    for it in items[1:]:
        if it[0] - base <= tol:
            current.append(it)
        else:
            lines.append(current)
            current, base = [it], it[0]
    lines.append(current)
    out = []
    for line in lines:
        line = sorted(line, key=lambda t: t[2])
        out.append({'text': ' '.join(t[4] for t in line),
                    'y1': min(t[0] for t in line),
                    'y2': max(t[1] for t in line),
                    'x1': min(t[2] for t in line),
                    'x2': max(t[3] for t in line)})
    return out


def join_hyphenated(text: str) -> str:
    """行をまたいで分かれた語をつなぐ

    `Set- aria viridis` のように，学名がハイフンで分断されることがある．
    """
    return re.sub(r'([A-Za-z])-\s+([a-z])', r'\1\2', text)


def normalize(text: str) -> str:
    """全角・空白をそろえる"""
    if not isinstance(text, str):
        return ''
    s = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', ' ', s).strip()


# 「調査番号：SO-216」の区切り．全角・半角のコロンが混じる
_SEP = r'[:：]'


def parse_header_text(text: str, aliases):
    """
    文章形式の表頭から「項目 -> 値」を取り出す

    `Feld-Nr. 調査番号：SO-216, Größe d. Probefläche 調査面積：100m².
     Höhe ü. Meer 海抜高：640m. Exposition 方位：W. …` という形．
    ドイツ語と和名が並ぶが，**和名を手がかりに切る**(表形式と同じ語彙が使える)．

    値の終わりは「次の項目名が始まるところ」とする．
    句読点で切ると `100m². Höhe` のように値の中の点で切れてしまう．

    Args:
        text: 領域から読んだ文字列
        aliases: [(列名, [和名, ...]), ...] plot_table.ITEM_ALIASES を渡す
    Returns:
        {列名: 値の文字列}
    """
    # 項目名は行をまたいで分かれる(`高木層の` + `植被率`)．
    # 和文の空白は行つなぎで入ったものなので，落としてから探す
    s = normalize(text).replace(' ', '')
    # 和名の出現位置を集める
    hits = []
    for key, words in aliases:
        for w in words:
            for m in re.finditer(re.escape(w), s):
                hits.append((m.start(), m.end(), key, w))
    # 階層ごとの項目(高木層の高さ など)も拾う
    # 階層名は和文字と数字だけ．空白を落としてあるので，
    # 制限しないと直前のドイツ語(Baumschicht高木層)まで飲み込む
    for m in re.finditer(r'([ぁ-んァ-ヶ一-龥0-9]{1,6}層)(?:の)?(高さ|植被率)', s):
        key = f'{m.group(1)}_{"height" if m.group(2) == "高さ" else "cover"}'
        hits.append((m.start(), m.end(), key, m.group(0)))
    if not hits:
        return {}
    # 同じ位置に複数当たったら長い語を採る(海抜高度 と 海抜高)
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    picked, last_end = [], -1
    for start, end, key, word in hits:
        if start >= last_end:
            picked.append((start, end, key))
            last_end = end
    out = {}
    for i, (start, end, key) in enumerate(picked):
        stop = picked[i + 1][0] if i + 1 < len(picked) else len(s)
        chunk = s[end:stop]
        m = re.match(rf'\s*{_SEP}\s*(.*)', chunk, flags=re.S)
        value = (m.group(1) if m else chunk).strip()
        # 次の項目のドイツ語名が値の後ろに付いてくるので落とす
        # 値の後ろには次の項目のドイツ語名が続くが，ここでは切らない．
        # 言語で切ろうとすると `IOOm?`(=100m²) の I や O を語の一部と見て
        # 値まで削ってしまう．**項目ごとの規則に値を探させる**方が確実
        # (correct_text.correct_header_value)．
        value = value.strip().rstrip('.,，。 ')
        if value and key not in out:
            out[key] = value
    return _split_aspect_slope(out)


# 「in 5 :」「in Lfd.-Nr. 1 :」で地点が変わる．
# OCRでは `in7`(空白なし) `einmalin`(前の語とくっつく) `LfdNr5`(点が消える)
# のように崩れるので，**`in` か `Nr` のあとの数字**を目印にする．
# 地点番号は小さいので2桁までに限る(学名の中の in を拾わないため)
_PLOT_MARK = re.compile(r'(?:in|Nr)\.?\s*[:：;.,\-]?\s*(\d{1,2})\s*[:：;.,]?', re.I)
# 階層(B/T/S/K/H/M)＋被度．階層は無いこともある．`S—1・2` `K—+` `+・2` `1・2`
# `+` は `十`，区切りのダッシュは `一` と読まれる(どちらも漢字)
_COVER = re.compile(
    r'(?:([BTSKHM][12]?)\s*[—–\-ー―一]\s*)?'
    # 被度が英字のとき(r)は単語の末尾を拾わないようにする．
    # `var.` の `.` が `,` と読まれて切れると，残った `var` の r を被度と見てしまう
    r'(?<![A-Za-z])'
    # 被度が + r 十 のときは区切りが消えることがある(`十2` = `+・2`)．
    # 数字どうしでは `12` と `1・2` を区別できないので，区切りを必須にする
    r'(?:([+r十])\s*[・.,]?\s*([54321])?|([54321])\s*(?:[・.,]\s*([54321]))?)'
    r'\s*$', re.I)
# 被度の中のカンマ(`1,1` や `+,2` の区切り)では種を分けない．
# 中黒がカンマと読まれるため，被度になりうる文字(数字と + r 十)の
# あいだにあるカンマは区切りとみなさない
_ENTRY_SPLIT = re.compile(r'(?<![\d+r十])[,，]|[,，](?!\s*\d)')
# 和名はカタカナ(まれに漢字)．学名はラテン文字
_JNAME = re.compile(r'[ァ-ヶー]{2,}')
# 常在度表の1回出現種は，被度ではなく `常在度(被度)` で書かれる
# (`ケチヂミザサ II(+)`．kinki_064 で実物を確認．2026-09-07)．
# 頭のローマ数字と括弧の中は `correct_text.correct_constancy()` が整える
_CONST_TAIL = re.compile(
    r'(?:([BTSKHM][12]?)\s*[—–\-ー―一]\s*)?'
    r'([IVivl|Tt\[【〔1-5]{1,4}|[+r])\s*[(\[{【〔し]\s*'
    r'([^)\]}】〕]*?)\s*[)\]}】〕]\s*$')


def _split_constancy(m):
    """`II(+・2)` の当たりを (階層, 常在度, 被度) に分ける

    整え方は `correct_text.correct_constancy()` に1つだけ置いてある．
    そちらが形として認めなければ，読めなかったものとして常在度は付けない．
    """
    from . import correct_text

    fixed = correct_text.correct_constancy(m.group(0)[len(m.group(1) or ''):]
                                           .lstrip(' —–-ー―一'))
    if not fixed:
        return m.group(1), None, normalize(m.group(3)).replace('十', '+')
    constancy, _, inner = fixed.partition('(')
    inner = inner.rstrip(')')
    raw = inner.replace('・', ';').replace('十', '+')
    return m.group(1), constancy, raw


def parse_once_species(text: str, start_plot=None):
    """
    「1回出現種」の列挙から，1件ずつ取り出す

    `出現1回の種 … in 1: Bidens pilosa コセンダングサ +・2, Cirsium maritimum
     ハマアザミ +, in 2: …` という形．

    同じ種に階層が2つ続くことがある(`ヒメドコロ S—+, K—+`)．
    後ろの断片には和名が無いので，直前の種に付ける．

    塊が紙面に収まらないと**次のページへあふれる**(2026-09-07)．
    あふれた先は `in N:` の目印から始まらず，前のページの地点の続きになる．
    `start_plot` にその地点番号を渡すと，最初の目印より前の断片をそこへ入れる．
    渡さなければ，目印より前は捨てる(これまでどおり)．

    Args:
        text      : 読み取った文字列
        start_plot: 目印より前の断片を入れる地点番号(続きのページで使う)
    Returns:
        [{'plot': 地点番号, 'j_name': 和名, 's_name': 学名,
          'layer': 階層 or None, 'comp_raw': 被度の文字列}, ...]
    """
    s = join_hyphenated(normalize(text))
    # 和名が行またぎで割れる(`ク` + `ロマツ`)．
    # カタカナのあいだの空白は行つなぎで入ったものなので落とす
    s = re.sub(r'(?<=[ァ-ヶー])\s+(?=[ァ-ヶー])', '', s)
    marks = list(_PLOT_MARK.finditer(s))
    # (地点番号, その地点の断片) に切り分ける．
    # 続きのページでは，最初の目印より前も1つの塊として扱う
    segments = []
    if start_plot is not None:
        head = s[:marks[0].start()] if marks else s
        if head.strip():
            segments.append((int(start_plot), head))
    elif not marks:
        return []
    for i, m in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        segments.append((int(m.group(1)), s[m.end():stop]))
    out = []
    for plot, body in segments:
        prev = None
        for frag in _ENTRY_SPLIT.split(body):
            frag = frag.strip().rstrip('.。')
            if not frag:
                continue
            # 常在度表の1回出現種は `常在度(被度)`．先に見る
            # (`I(1)` は被度の形にも読めてしまうため)
            const = _CONST_TAIL.search(frag)
            if const is not None:
                layer, constancy, raw = _split_constancy(const)
                cut = const.start()
            else:
                cover = _COVER.search(frag)
                if cover is None:
                    continue
                # 被度だけを残す(階層は layer に分けて持つ)．十 は + の誤読
                head_cover = (cover.group(2)
                              or cover.group(4)).replace('十', '+')
                soc = cover.group(3) or cover.group(5)
                raw = f'{head_cover};{soc}' if soc else head_cover
                layer, constancy, cut = cover.group(1), None, cover.start()
            head = frag[:cut].strip()
            jm = _JNAME.search(head)
            if jm is None:
                # 和名が無い断片は，直前の種の別の階層
                if prev is not None:
                    out.append({**prev, 'layer': layer, 'comp_raw': raw,
                                'constancy': constancy})
                continue
            j_name = jm.group(0)
            s_name = head[:jm.start()].strip(' .,')
            prev = {'plot': plot, 'j_name': j_name, 's_name': s_name or None}
            out.append({**prev, 'layer': layer, 'comp_raw': raw,
                        'constancy': constancy})
    return out


def _split_aspect_slope(items):
    """`方位及び傾斜：S 2°` のように2項目で値を1つ共有する形をほどく

    前の項目の値が接続詞だけになるので，それを手がかりにする．
    """
    if items.get('aspect') and re.fullmatch(r'(及び|および|と|・|/)+',
                                            items['aspect']):
        m = re.match(r'\s*([NSEWnsew]+)\s*(.*)', items.get('slope', ''))
        if m:
            items['aspect'] = m.group(1).upper()
            items['slope'] = m.group(2).strip()
        else:
            del items['aspect']
    return items


# 表の下に付く注記の見出し．ドイツ語と日本語が並ぶので，どちらでも拾う．
# 値は `Lfd. Nr. 1,2: …, 3: …` のように地点ごとに並ぶことも，
# 1地点の表では地点の指定なしに1つだけ書かれることもある
_NOTE_FIELDS = (
    ('locality', r'(?:Lage(?:\s*d\.?\s*Aufn\.?)?|Fundorte?|調査地)'),
    ('date', r'(?:Datum(?:\s*d\.?\s*Aufn\.?)?|調査年月日)'),
    # 印字は `Nachweis d. Vegetationsaufnahmen` `…aufnahme` `…anfnahmen`(誤植)
    # と揺れるので，`Vegetation` から先は見ない
    ('source_ref', r'(?:Nachweis(?:\s*d\.?\s*Vegetation\w*)?|既発表資料名?)'),
)
_NOTE_HEAD = re.compile(
    '(?:' + '|'.join(f'(?P<{k}>{v})' for k, v in _NOTE_FIELDS)
    + r')\s*[:：]?', re.I)
# `Lfd. Nr.` `Lfd.-Nr.` `Nr.` の前置き．地点番号の前に付くだけで意味は無い
_LFD = re.compile(r'(?:Lfd\.?\s*[-–—]?\s*)?Nr\.?\s*', re.I)
# 地点の指定．`1` `1,2` `1~5` `1—9` `8, 10`．3桁の数(西暦)は地点ではない
_PLOT_SPEC = re.compile(
    r'(?<!\d)(\d{1,2}(?!\d)(?:\s*[,，~〜\-–—]\s*\d{1,2}(?!\d))*)\s*[:：]')


# 出典の値の形．見出し(`Nachweis` / `既発表資料`)が前のページに残ると，
# 出典の並びが調査地の節に続いてしまう．値の形で見分ける．
# 調査地に年が入るのは括弧の中だけ(`… 箱石 (6. Juli 1983)`)なので，
# **年で終わる**ものだけを出典とみなす
_SOURCE_REF = re.compile(r'原調査資料|既発表資料|Original|(?:19|20)\d{2}\s*$', re.I)


def _expand_plots(spec: str):
    """`1,2` `1~5` `8, 10` を地点番号の一覧にほどく"""
    out = []
    for part in re.split(r'[,，]', spec):
        part = part.strip()
        m = re.fullmatch(r'(\d{1,2})\s*[~〜\-–—]\s*(\d{1,2})', part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.extend(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def parse_site_notes(text: str):
    """表の下の注記から，地点ごとの調査地・調査年月日・出典を取り出す

    `Lage d. Aufn. 調査地: Lfd. Nr. 1,2: Ryujin-mura, Hidaka-gun 日高郡龍神村,
     3: Tsuchiyama-cho …. Datum d. Aufn. 調査年月日: 6 Nov. 1973.
     Nachweis d. Vegetationsaufnahmen 既発表資料 1, 3: Präf. Shiga 滋賀県 1978`

    地点の指定が無いときは `plot` を `None` にして1件だけ返す
    (1地点の表や，表の全体に掛かる注記)．

    Returns:
        [{'plot': 地点番号 or None, 'field': 'locality'|'date'|'source_ref',
          'value': 文字列}, ...]
    """
    s = normalize(text)
    heads = list(_NOTE_HEAD.finditer(s))
    if not heads:
        return []
    fields = dict(_NOTE_FIELDS)
    out = []
    for i, h in enumerate(heads):
        field = h.lastgroup
        if field not in fields:
            continue
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(s)
        body = _LFD.sub('', s[h.end():stop]).strip(' .,，:：')
        marks = list(_PLOT_SPEC.finditer(body))
        if not marks:
            if body:
                out.append({'plot': None, 'field': field, 'value': body})
            continue
        for j, m in enumerate(marks):
            end = marks[j + 1].start() if j + 1 < len(marks) else len(body)
            value = body[m.end():end].strip(' .,，')
            here = ('source_ref' if field == 'locality'
                    and _SOURCE_REF.search(value) else field)
            for plot in _expand_plots(m.group(1)):
                out.append({'plot': plot, 'field': here, 'value': value})
    return out
