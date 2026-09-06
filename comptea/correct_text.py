import re
import unicodedata
import json
from functools import lru_cache
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

def correct_layer(str):
    """
    階層の文字列を修正

    半角・大文字に変換
    数字を除外
    ';'を間にいれる

    Args:
        str: 文字列
    Returns:
        修正した文字列と検証の結果
        status: 'OK'    階層1つとして読めた
                'multi' 複数の階層('S;K'など)．誤りとは限らないが目視で確かめる
                'Need Check' 階層として読めない
                None    空(セルが空)
    """
    corrected = unicodedata.normalize("NFKC", str)
    corrected = corrected.upper()
    # 区切りは後で入れ直すので，元から入っている ';' もここで落とす．
    # 落とさないと 'S;K' が 'S;;;K' になる(2026-09-01)
    corrected = re.sub('[-_*･・.,+ ;]', '', corrected)
    corrected = re.sub('5', 'S', corrected)
    corrected = ';'.join(corrected)
    corrected = re.sub('([BTSKH]);([12])', '\\1\\2', corrected)
    if corrected == '':
        return {'corrected': corrected, 'status': None}
    if not validate_layer(corrected):
        status = 'Need Check'
    elif ';' in corrected:
        status = 'multi'
    else:
        status = 'OK'
    return {'corrected': corrected, 'status': status}

def validate_layer(str):
    """
    階層の文字列が'[BTSKH][12]?'に合致しているか

    ';'区切りで複数の階層が入ることがあるため，区切って1つずつ見る．
    細分は'S1' 'S2'のように1桁までなので'[12]?'とする．
    部分一致だと'あS'のような誤読が通ってしまうので全体一致で見る．

    Args:
        str: 文字列
    Returns:
        合致するとき: True
        合致しない: False
    """
    str = str.split(';')
    pattern = '[BTSKH][12]?'
    matched = [bool(re.fullmatch(pattern, s)) for s in str]
    matched = all(matched) # is all True?
    return matched

# 常在度表のセル(2026-09-04)．折り込みの 68 表のうち 11_p1・22_p2 の 2 表は
# **常在度表**で，列が地点ではなく群落，セルが「常在度(被度の範囲)」になる
# (`IV(+-3)`・`II(+-1)`)．常在度は V-I の5段階
# (docs/vegetation_science.md 「常在度」)．
# **頭の算用数字をローマ数字に読み替えてはいけない**(2026-09-05 に実物で確認)．
# 同じ表に `1(+)`(数字の 1．被度1・群度+)と `I(+)`(ローマ数字．常在度I)が
# **どちらも印字されている**．字形も別で(1 は旗付き，I は上下に serif)，
# EasyOCR も区別して読めている．どちらの列かは列ごとに決まるので，
# 読み替えは `comp_table.column_head_kinds()` が列の多数決でおこなう．
# ここでは**印字どおり**返し，明らかな誤読の字だけを直す．
# 閉じ括弧は落ちることが多い(`III(+-1`)ので無くても認める
# 頭の `[`・`【` は**開き括弧ではなくローマ数字の I** の誤読(2026-09-05 に
# 実物で確認．`[II(+}` の印字は `III(+)`)．頭の位置に開き括弧は来ないので，
# 取り違えの心配はない(`[+-1)` のように括弧が続かない形は，下の正規表現で外れる)
_ROMAN = {'l': 'I', '|': 'I', 'i': 'I', 'v': 'V', 'T': 'I', 't': 'I',
          '[': 'I', '【': 'I', '〔': 'I'}
# ローマ数字と算用数字の対応(列の種類が決まったあとで使う)
ROMAN2DIGIT = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5'}
DIGIT2ROMAN = {v: k for k, v in ROMAN2DIGIT.items()}
# 開き括弧は `( [ { 【 〔 し`，閉じ括弧は `) ] } 】 〕` と読まれる
# (`し` は `4し+-3)` の実例．この形のセルに仮名は来ない)．
# 閉じ括弧のあとに 1 字ぶんの誤読が付くことがある(`3(+-1);`)ので，末尾は緩く見る
CONSTANCY_RE = re.compile(
    r'^([IVivl|Tt\[【〔1-5]{1,4}|[+r])\s*[(\[{【〔し]\s*'
    r'([^)\]}】〕]*?)\s*([)\]}】〕].?)?$')
COVER_RANGE_RE = re.compile(r'(?:[54321]|[+r])(?:-(?:[54321]|[+r]))?')


def correct_constancy(text):
    """`頭(括弧の中)` の形のセルを整えて返す．違えば None

    頭は常在度のローマ数字(I-V)か，単独地点の列の被度(算用数字 1-5)か，
    `+`・`r`．括弧の中は被度の範囲(`+`・`+-1`・`1-2` など)．
    **どちらの意味かはここでは決めない**(列ごとに決まる．上の注記を見る)．
    """
    s = unicodedata.normalize('NFKC', str(text)).strip()
    s = re.sub('[‐−–—ー]', '-', s).replace('十', '+')
    m = CONSTANCY_RE.match(s)
    if not m:
        return None
    head = ''.join(_ROMAN.get(c, c) for c in m.group(1))
    # 数字が 2 つ以上並ぶ頭は，ローマ数字の誤読とみなす(`11` → `II`)．
    # 印字の頭が算用数字なら 1 桁しかない
    if len(head) >= 2 and head.isdigit():
        head = ''.join(DIGIT2ROMAN.get(c, c) for c in head)
    # 常在度は V-I の5段階(docs/vegetation_science.md)．
    # **I に満たない散発の出現は `+`・`r`** と印字される(s01115_11_p1 で実物を確認)．
    # 算用数字 1 桁の頭は，単独地点の列の**被度**(`1(+)` = 被度1・群度+)
    if not re.fullmatch(r'I{1,3}|IV|V|[1-5]|[+r]', head):
        return None
    inner = re.sub(r'[\s.,;・･]', '', m.group(2))
    # **括弧の中にローマ数字は来ない**(被度の範囲だけ)ので，`I`・`l`・`|` は
    # 算用数字の 1 の読み違い(2026-09-05．`+(+-I` は印字 `+(+-1)`)．
    # ただし**範囲の後ろ側だけ**に直す．前側は `+` の読み違いのことがあり
    # (印字 `III(+-4)` が `III(l-4)`)，`1` にすると黙って別の値になる．
    # 前側が読めないセルは値にせず，目視に回す
    inner = re.sub(r'(?<=-)[Il|i]', '1', inner)
    # **括弧の中は被度の範囲**なので，記号が 2 つ並んでいたら間の `-` が
    # 落ちたものとみなす(`(+2)` → `(+-2)`．印字は `III(+-2)`．2026-09-04)
    if re.fullmatch(r'(?:[54321]|[+r]){2}', inner):
        inner = f'{inner[0]}-{inner[1]}'
    # 閉じ括弧を字と読むことがある(`III(+-15` は印字 `III(+-1)`)．
    # 中身が被度の範囲にならないときだけ，末尾の 1 字を落として試す
    if inner and not COVER_RANGE_RE.fullmatch(inner) \
            and COVER_RANGE_RE.fullmatch(inner[:-1]):
        inner = inner[:-1]
    if not inner or not COVER_RANGE_RE.fullmatch(inner):
        return None
    return f'{head}({inner})'


def correct_comp(str):
    """
    被度・群度の文字列を修正

    半角に変換
    区切りの記号('・'など)を除外
    ';'を間にいれる

    Args:
        str: 文字列
    Returns:
        修正した文字列と検証の結果
        status: 'OK' / 'Need Check' / None(空 = 非出現)
    """
    # 常在度表のセル(`IV(+-3)`)は被度とは別の形なので，先に見る
    constancy = correct_constancy(str)
    if constancy is not None:
        return {'corrected': constancy, 'status': 'OK'}
    corrected = unicodedata.normalize("NFKC", str)
    # **`;` も区切りとして落とす**(2026-09-03)．最後に 1 文字ずつ `;` で
    # 繋ぐので，読む人が約束どおり `5;5` と書くと `5;;;5` になっていた
    # (`references/reading-guide.md` は区切りを `;` と書くよう求めている)
    corrected = re.sub('[-_*･・., ;]', '', corrected)
    corrected = re.sub('十', '+', corrected)
    corrected = re.sub('^{', '', corrected)
    corrected = re.sub('}$', '', corrected)
    corrected = ';'.join(corrected)
    # 以下の修正は必要?
    # 実際のデータを見て判断する
    # corrected = text.replace(' ', '') # str_remove_all(" ")
    # corrected = re.sub(r'[|)\\]]$', '', corrected) # str_remove("[|)\\]]$")
    # corrected = re.sub(r'^[|(\[]', '', corrected) # str_remove("^[|(\\[]")
    # corrected = re.sub(r'[lj]', '1', corrected) # str_replace_all("[lj]", "1")
    if corrected == '':
        # '・'などが除かれて空になる = 非出現
        return {'corrected': corrected, 'status': None}
    return {'corrected': corrected,
            'status': 'OK' if validate_comp(corrected) else 'Need Check'}


def validate_comp(str):
    """
    組成の文字列の妥当性チェック

    「被度・群度」の順．被度は 5 4 3 2 1 + r のいずれかで，
    群度 1-5 を続けられる．
    **`+` や `r` にも群度は付く**(`+・2` は実データの最頻値)．
    省略するのは群度が1のときだけで，`+` は `+・1` を意味する
    (docs/vegetation_science.md 3.3)．

    Args:
        str: 文字列
    Returns:
        合致するとき: True
        合致しない: False
    """
    pattern = '(?:[54321]|[+r])(?:;[54321])?'
    if re.fullmatch(pattern, str):
        return True
    # 常在度表のセル(`IV(+-3)`)も値として認める(2026-09-04)
    return correct_constancy(str) == str


# 表頭の項目ごとに，取りうる形が決まっている
# 数字と紛れやすい文字(docs/vegetation_science.md 5節)を直してから検証する
_DIGIT_FIX = str.maketrans({'O': '0', 'o': '0', 'D': '0',
                            'l': '1', 'I': '1', '|': '1',
                            'S': '5', 'B': '8', 'Z': '2'})
# 方位は英字．数字と読み違えたものを戻す
_ASPECT_FIX = str.maketrans({'0': 'O', '5': 'S', '1': 'I', '巳': 'E', '己': 'E'})
ASPECTS = {'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'}
# 値が無いことを表す記号(方位が無い，傾斜が平坦など)
NO_VALUE = {'-', '—', '–', 'ー', '−', 'L'}
# 種名の候補をいくつまで並べるか．これを超えたら辞書で絞れていないとみなす
MAX_SUGGEST = 5
# 種名を辞書の名前へ置き換えてよい編集距離の上限．
# これを超える候補は採らず，元の印字を残して 'Need Check' にする
MAX_ADOPT_DIST = 1
# この長さ以下の読みでは，距離 1 の候補を採らない(下の注記を見る)
MIN_MATCH_LEN = 1


def _clean(text):
    if not isinstance(text, str):
        return ''
    return unicodedata.normalize('NFKC', text).strip()


# 文章形式の表頭では値に単位が付く(100m², 640m, 0.6m, 70%, 40°)
_UNIT = re.compile(r'\s*(m2|m²|㎡|cm|mm|km|ha|m|%|°|度)\s*$', re.I)


# 数字と読み違えられる文字．これらを含む塊を数値の候補とみなす
_NUMLIKE = '0-9OoDlI|SBZ'
_NUM_TOKEN = re.compile(rf'([{_NUMLIKE}]+(?:\s?[.,]\s?[{_NUMLIKE}]+)?)')


def correct_number(text, allow_decimal=True, limit=None):
    """数値の項目を直す

    文章形式の表頭では値の後ろにドイツ語が続く(`100m². Höhe ü. Meer`)ので，
    文字列全体を数字とみなさず，**先頭の数値だけを取り出す**．
    '1. 6' -> '1.6'，'l0' -> '10'，'640m' -> '640'，'IOOm?Hohe' -> '100'
    """
    s = _clean(text)
    if s in NO_VALUE or s == '':
        return {'corrected': None, 'status': None}
    m = _NUM_TOKEN.search(_UNIT.sub('', s))
    if m is None:
        return {'corrected': s, 'status': 'Need Check'}
    token = re.sub(r'\s+', '', m.group(1)).translate(_DIGIT_FIX)
    if not allow_decimal:
        token = token.replace('.', '').replace(',', '')
    token = token.replace(',', '.')
    token = re.sub(r'\.(?=.*\.)', '', token)  # 小数点が複数あれば最初以外を捨てる
    if not re.fullmatch(r'\d+(\.\d+)?', token):
        return {'corrected': s, 'status': 'Need Check'}
    rest = s[m.end():].lstrip()
    # 小数点のあとが読めていない形('0.Gm' -> 0 と 0.6 の区別がつかない)
    if re.match(r'[.,]\s*\D', rest):
        return {'corrected': token, 'status': 'Need Check'}
    # 先頭が 0 で小数点が無いのは，小数点を読み落とした形('0 6' -> '06')．
    # 0.6 と 6 のどちらかは決められないので，直さずに目視へ回す
    if len(token) > 1 and token[0] == '0' and '.' not in token:
        return {'corrected': token, 'status': 'Need Check'}
    # 上限を超える値は，単位を数字と読み違えた形(50% -> 509，40° -> 409)
    if limit is not None and float(token) > limit:
        return {'corrected': token, 'status': 'Need Check'}
    return {'corrected': token, 'status': 'OK'}


def correct_aspect(text):
    """方位を直す．'N巳' -> 'NE'

    文章形式では値の後ろにドイツ語が続く(`W. Neigung`)ので，
    **先頭の方位らしい文字だけ**を見る．
    """
    s = _clean(text)
    if s in NO_VALUE or s == '':
        return {'corrected': None, 'status': None}
    m = re.match(r'[^A-Za-z巳己]*([A-Za-z巳己0-9]{1,3})', s)
    head = (m.group(1) if m else s).translate(_ASPECT_FIX).upper()
    return {'corrected': head if head in ASPECTS else s,
            'status': 'OK' if head in ASPECTS else 'Need Check'}


def correct_date(text):
    """調査年月日を直す．'83 5 21' -> '1983-05-21'"""
    s = _clean(text)
    nums = [int(n) for n in re.findall(r'\d+', s.translate(_DIGIT_FIX))]
    if len(nums) < 3:
        return {'corrected': _clean(text) or None,
                'status': 'Need Check' if s else None}
    y, m, d = nums[:3]
    if y < 100:                                # '83 -> 1983
        y += 1900 if y >= 50 else 2000
    ok = 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31
    return {'corrected': f'{y:04d}-{m:02d}-{d:02d}' if ok else _clean(text),
            'status': 'OK' if ok else 'Need Check'}


# 調査番号の接頭辞は英字2文字が定型(SO, KF, MS, HK, SS, MN…)
_HEAD_FIX = str.maketrans({'0': 'O', '1': 'I', '5': 'S', '8': 'B'})


def correct_field_no(text):
    """調査番号を直す．英字2文字＋数字が定型(SO-216, KF 360, MS-82)

    `SO` を `S0` と読み違えるが，数字に先頭のゼロは付かないので，
    **先頭2文字を接頭辞**として切れば戻せる．
    番号だけ(`49`)の資料もある．
    """
    s = re.sub(r'[\s\-.]', '', _clean(text)).upper()
    if not s:
        return {'corrected': None, 'status': None}
    if s.isdigit():
        return {'corrected': s, 'status': 'OK'}
    # 文章形式では後ろにドイツ語が続くので，全体一致ではなく先頭から探す
    m = re.match(r'([A-Z0-9]{2})(\d+)', s)
    if m and any(ch.isalpha() for ch in m.group(1)):
        return {'corrected': f'{m.group(1).translate(_HEAD_FIX)}-{m.group(2)}',
                'status': 'OK'}
    return {'corrected': s, 'status': 'Need Check'}


# 項目名 -> 補正の関数
HEADER_RULES = {
    'plot_no'  : lambda t: correct_number(t, allow_decimal=False),
    'field_no' : correct_field_no,
    'date'     : correct_date,
    'area'     : correct_number,
    'altitude' : correct_number,
    'aspect'   : correct_aspect,
    'slope'    : lambda t: correct_number(t, limit=90),
    'veg_height': correct_number,
    'veg_cover': lambda t: correct_number(t, limit=100),
    'n_species': lambda t: correct_number(t, allow_decimal=False),
}

# 取りうる値の上限．`%` や `°` が `9` と読まれる誤り(50% -> 509)を弾ける
HEADER_LIMITS = {'slope': 90, 'veg_cover': 100}


def correct_header_value(key, text):
    """
    表頭の値を，項目に応じて直す

    Args:
        key: plot_table の列名(plot_no, altitude, 低木層_height など)
        text: OCRで読んだ文字列
    Returns:
        {'corrected': 直した文字列, 'status': 'OK' / 'Need Check' / None(空)}
    """
    rule = HEADER_RULES.get(key)
    if rule is None and key.endswith('_cover'):
        rule = lambda t: correct_number(t, limit=100)   # 階層ごとの植被率
    if rule is None and key.endswith('_height'):
        rule = correct_number                           # 階層ごとの高さ
    if rule is None:
        return {'corrected': _clean(text) or None, 'status': None}
    return rule(text)


def correct_cell(obj_name, text):
    """
    セルのクラスに応じて補正を振り分ける

    振り分けを1箇所にまとめる(OCRページとスキルの両方から呼ぶ)．

    文章として読む領域(header / once_species)と表頭のセルには，
    セル単位の補正が無い．そこで **text をそのまま corrected に入れる**．
    None にすると plot_table / comp_table が読む列が空になり，
    部品ごとには動くのに通しでは何も出ない，という壊れ方をする．

    Args:
        obj_name: locate.py が付けたクラス名
        text: OCRで読んだ文字列
    Returns:
        {'corrected': 直した文字列, 'status': 'OK' / 'Need Check' / ...}
    """
    # 読めなかったセルは NaN で来る．辞書との比較に渡すと落ちるので先に外す
    # (非出現のセルは comp_table 側で 'absent' として扱われる)
    if not isinstance(text, str) or not text.strip():
        return {'corrected': None, 'status': None}
    match obj_name:
        case 'species_col':
            return correct_name(text, target='j_name')
        case 'sname':
            return correct_name(text, target='s_name')
        case 'layer':
            return correct_layer(text)
        case 'comp':
            return correct_comp(text)
        case _:
            return {'corrected': text if isinstance(text, str) else None,
                    'status': None}


@lru_cache(maxsize=None)
def _load_names(dict_path):
    """
    種名の辞書ファイルを読む(1度だけ)

    correct_name() は1セルにつき1回呼ばれるので，そのたびに読むと
    ファイルの読み込みだけで数ミリ秒かかる．結果を憶えておく．

    Args:
        dict_path: 辞書ファイルのパス
    Returns:
        (名前のタプル, 名前の集合)
        タプルは並び順を保つため(候補が同点のとき辞書の順で並べる)．
        集合は完全一致を O(1) で見るため．
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        names = tuple(line.strip() for line in f)
    return names, frozenset(names)

def known_names(values, dict_path="j_name.txt"):
    """それぞれの値が，辞書にそのまま載っている名前かを返す

    種として確からしいかを，下流の検査が判じるための道具(2026-09-06)．
    種群の見出し(`亜群集区分種`)・候補が複数のまま連ねたもの(`クグ;クコ;…`)・
    1 文字の読み残りは，どれも辞書に無い．

    Args:
        values: 名前の並び(pandas の Series でも list でもよい)
    Returns:
        同じ長さの真偽のリスト
    """
    try:
        _, names = _load_names(dict_path)
    except OSError:
        return [True] * len(values)
    return [isinstance(v, str) and v in names for v in values]


def correct_name(input_str: str, target="s_name", dict_path_s="s_name.txt", dict_path_j="j_name.txt") -> dict:
    """
    指定された文字列をファイル内の名前と比較し、結果をJSON形式の辞書で返します。

    Args:
        input_str (str): 比較対象の文字列
        target: 検索する種名。"s_name"：学名，"j_name"：和名
        dict_path_j: 和名の辞書ファイルのパス
        dict_path_s: 学名の辞書ファイルのパス
    Returns:
        dict: 比較結果に応じた辞書。
    """
    if not input_str:
        return None

    sp_dict = dict_path_s if target == "s_name" else dict_path_j
    sp_name, sp_set = _load_names(sp_dict)

    # 完全一致をチェック
    if input_str in sp_set:
        return {"corrected": input_str, "status": "OK"}

    # **辞書の見出しの頭に，空白付きで一致するなら正しい名前とみなす**．
    # 辞書(維管束植物和名チェックリスト)は種内分類群まで載せているため，
    # `Actinidia arguta f. megalocarpa` はあるのに `Actinidia arguta` が無い．
    # 組成表は種名で書かれることが多く，そのままでは辞書と噛み合わない
    # (2026-09-01)．
    if _is_prefix_of_name(input_str, sp_name):
        return {"corrected": input_str, "status": "OK"}

    # 編集距離が3以内のものを探索
    # 全件に距離を計算せず，3を超えたら打ち切る(score_cutoff)．
    # process.extract は同点の候補を辞書の並び順のまま返すので，
    # 1件ずつ回していたときと候補の並びが変わらない．
    hits = process.extract(input_str, sp_name, scorer=Levenshtein.distance,
                           score_cutoff=3, limit=None)
    if not hits:
        # 編集距離が3以内のものがない場合
        return {"corrected": input_str, "status": "Need Check"}

    # 距離が最小のものだけを残す
    min_distance = min(hit[1] for hit in hits)
    # **離れた候補は採らない**．辞書に載っていない名前(異名・古い学名・
    # 品種名など)を，近いだけの別種にすり替えてしまうため
    # (`Actinidia arguta` が距離3の `Actinidia rufa` になった．2026-09-01)．
    # 元の印字を残し，'Need Check' にして段階2へ回す．
    if min_distance > MAX_ADOPT_DIST:
        return {"corrected": input_str, "status": "Need Check"}
    # **1 文字の読みは，完全に一致するときだけ採る**(2026-09-06)．
    # 1 文字どうしは必ず距離 1 になるので，辞書にある 1 文字の名前が
    # 総なめで候補になる(`口` → `イ;桂;樟`)．しかも同じ文字列が何行にも
    # 並ぶので，重複の検査が鳴る(折り込みの要確認 42 件のうち 31 件)．
    # 実物は `モミ`・`カヤ`・`クリ` で，OCR が 1 文字しか拾えていなかった．
    # 折り込みでは 115 件がこれに当たる(候補が複数 104・置き換え 11)
    if len(input_str) <= MIN_MATCH_LEN and min_distance > 0:
        return {"corrected": input_str, "status": "Need Check"}
    candidates = [hit[0] for hit in hits if hit[1] == min_distance]

    # **濁点・半濁点だけの違いは，同じ名前とみなして先に採る**．
    # 古い印刷は「クロヅル」を「クロツル」と組んでいることがあり，
    # 先頭からの一致だけで選ぶと「クロツグ」(別属の別種)に化ける．
    # どちらも編集距離1なので，距離では分けられない (2026-09-01)．
    same_kana = [n for n in candidates
                 if _strip_voiced(n) == _strip_voiced(input_str)]
    if same_kana:
        candidates = same_kana
    else:
        # 先頭からの一致が長いものを採る
        max_leading = max(_get_leading_match_length(input_str, name)
                          for name in candidates)
        candidates = [name for name in candidates
                      if _get_leading_match_length(input_str, name) == max_leading]

    # 候補が多すぎるときは，辞書で絞れていないので候補を出さない．
    # 「随伴種」のような短い語は，編集距離3以内に2-3文字の種名が何百と当たり，
    # 全部を ';' で連ねると数千文字の値になって表を壊す
    # (2026-09-01．--reader ai の出力で分かった)．
    if len(candidates) > MAX_SUGGEST:
        return {"corrected": input_str, "status": "Need Check"}

    suggested_name = ";".join(candidates)
    return {"corrected": suggested_name, "status": "suggested"}

@lru_cache(maxsize=4)
def _load_pairs(dict_path):
    """和名 -> 学名 の対応表を読む(1度だけ)

    `download_species_names.py` が作る `js_name.txt`(タブ区切り)．
    学名が複数あるときは ';' で連ねてある．
    """
    pairs = {}
    try:
        with open(dict_path, encoding='utf-8') as f:
            for line in f:
                jn, _, sn = line.rstrip('\n').partition('\t')
                if jn and sn:
                    pairs[jn] = tuple(sn.split(';'))
    except OSError:
        return {}
    return pairs


def sname_from_jname(j_name, dict_path_js="js_name.txt"):
    """和名から学名を引く．**1つに決められるときだけ**返す

    **印字されている学名を置き換えるためのものではない**(2026-09-01 決定)．
    学名が空の行を埋めるためだけに使う．引ける学名は**いまの分類の名前**なので，
    古い資料の印字とは食い違うことがある
    (イタドリ: 印字 `Polygonum cuspidatum` / 引くと `Fallopia japonica var. japonica`)．

    種内分類群まで載っているため，同じ種の別の階級が複数当たることがある．
    **属名と種小名が同じなら同じ種**とみなし，いちばん短い名前を返す．
    属や種が分かれるとき(イワガラミなど)は決められないので返さない．

    Returns:
        (学名, 理由)．理由は 'ok' / 'none'(引けない) / 'multi'(決められない)
    """
    if not j_name:
        return '', 'none'
    names = _load_pairs(dict_path_js).get(str(j_name).strip())
    if not names:
        return '', 'none'
    species = {' '.join(n.split(' ')[:2]) for n in names}
    if len(species) > 1:
        return '', 'multi'
    return min(names, key=len), 'ok'


@lru_cache(maxsize=8)
def _name_prefixes(sp_name):
    """辞書の見出しから，「種名まで」の部分を集める

    `Actinidia arguta f. megalocarpa` から `Actinidia arguta` を作る．
    種名だけで書かれたセルを，辞書に無いと切り捨てないため．
    """
    out = set()
    for name in sp_name:
        parts = name.split(' ')
        for i in range(2, len(parts)):
            out.add(' '.join(parts[:i]))
    return out


def _is_prefix_of_name(input_str, sp_name):
    """入力が，辞書の見出しの「種名まで」と一致するか"""
    return ' ' in input_str and input_str in _name_prefixes(sp_name)


def _strip_voiced(text):
    """濁点・半濁点を落とした形にする(「クロヅル」→「クロツル」)

    NFD に分けると濁点は結合文字(U+3099 / U+309A)になるので，それを外す．
    """
    return ''.join(c for c in unicodedata.normalize('NFD', str(text))
                   if c not in ('゙', '゚'))


def _get_leading_match_length(str1, str2):
    """
    2つの文字列の先頭から一致する文字の長さを返します。
    """
    length = 0
    for char1, char2 in zip(str1, str2):
        if char1 == char2:
            length += 1
        else:
            break
    return length



if __name__ == "__main__":

    # 使用例
    # 比較対象の文字列
    test_str_ok = "ショウジョウバカマ"
    test_str_suggested = "ショウジョウバカマモドキ"
    test_str_multiple = "イワナシダ"
    test_str_no_match = "アアアアアアア"

    print(f"{correct_name(test_str_ok       , target='j_name')}")
    print(f"{correct_name(test_str_suggested, target='j_name')}")
    print(f"{correct_name(test_str_multiple , target='j_name')}")
    print(f"{correct_name(test_str_no_match , target='j_name')}")

    for str in ["5,3", "+2", "+r", "62", "5", "・", ""]:
        print(f'{str!r:6} -> {correct_comp(str)}')

    for str in ["T1", "S K", "5", "S;が", "T12", "あS", ""]:
        print(f'{str!r:6} -> {correct_layer(str)}')
