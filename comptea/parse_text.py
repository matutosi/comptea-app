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
    items = []
    for box, text, *_ in results:
        if not text:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), max(ys), min(xs), text))
    if not items:
        return ''
    heights = [b - a for a, b, _, _ in items]
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
    return ' '.join(' '.join(t[3] for t in sorted(line, key=lambda t: t[2]))
                    for line in lines)


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


def parse_once_species(text: str):
    """
    「1回出現種」の列挙から，1件ずつ取り出す

    `出現1回の種 … in 1: Bidens pilosa コセンダングサ +・2, Cirsium maritimum
     ハマアザミ +, in 2: …` という形．

    同じ種に階層が2つ続くことがある(`ヒメドコロ S—+, K—+`)．
    後ろの断片には和名が無いので，直前の種に付ける．

    Returns:
        [{'plot': 地点番号, 'j_name': 和名, 's_name': 学名,
          'layer': 階層 or None, 'comp_raw': 被度の文字列}, ...]
    """
    s = join_hyphenated(normalize(text))
    # 和名が行またぎで割れる(`ク` + `ロマツ`)．
    # カタカナのあいだの空白は行つなぎで入ったものなので落とす
    s = re.sub(r'(?<=[ァ-ヶー])\s+(?=[ァ-ヶー])', '', s)
    marks = list(_PLOT_MARK.finditer(s))
    if not marks:
        return []
    out = []
    for i, m in enumerate(marks):
        plot = int(m.group(1))
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        body = s[m.end():stop]
        prev = None
        for frag in _ENTRY_SPLIT.split(body):
            frag = frag.strip().rstrip('.。')
            if not frag:
                continue
            cover = _COVER.search(frag)
            if cover is None:
                continue
            # 被度だけを残す(階層は layer に分けて持つ)．十 は + の誤読
            head_cover = (cover.group(2) or cover.group(4)).replace('十', '+')
            soc = cover.group(3) or cover.group(5)
            raw = f'{head_cover};{soc}' if soc else head_cover
            head = frag[:cover.start()].strip()
            jm = _JNAME.search(head)
            if jm is None:
                # 和名が無い断片は，直前の種の別の階層
                if prev is not None:
                    out.append({**prev, 'layer': cover.group(1),
                                'comp_raw': raw})
                continue
            j_name = jm.group(0)
            s_name = head[:jm.start()].strip(' .,')
            prev = {'plot': plot, 'j_name': j_name, 's_name': s_name or None}
            out.append({**prev, 'layer': cover.group(1), 'comp_raw': raw})
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
