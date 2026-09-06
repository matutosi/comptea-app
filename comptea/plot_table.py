"""表頭から，地点ごとの属性の表を組み立てる

組成部の縦持ち(1行 = 1地点 × 1種)とは粒度が違うので，別の表として出す．
出力は 1行 = 1地点．列は項目(調査番号・調査年月日・海抜高度…)．

入力は OCR 済みの位置決め結果で，次のクラスを使う．
    header_item     項目名(ドイツ語．header_col の範囲)
    header_item_ja  項目名(和名．header_col と値のあいだ)
    header_value    値(項目行 × 地点)

項目名は和名を優先する(カタカナ・漢字の方がOCRが安定し，
突き合わせる語彙も docs/vegetation_science.md にある)．
"""
import re
import unicodedata

import pandas as pd

import Levenshtein

import correct_text
import parse_text

# 和名の項目名 -> 出力の列名
# 資料によって書き方が揺れるので，含まれていれば一致とみなす
ITEM_ALIASES = [
    ('plot_no'    , ['通し番号', '通番']),
    ('field_no'   , ['調査番号']),
    ('date'       , ['調査年月日', '調査月日', '調査日']),
    ('locality'   , ['調査地']),
    ('area'       , ['調査面積']),
    ('altitude'   , ['海抜高度', '海抜高', '標高']),
    ('aspect'     , ['方位']),
    # 傾斜 は 順斜 と誤読されることがある．斜 を含む項目は他に無い
    ('slope'      , ['傾斜', '斜']),
    ('veg_height' , ['植生高', '植生の高さ']),
    ('veg_cover'  , ['植被率']),
    ('n_species'  , ['出現種数', '出現種']),
    ('community'  , ['群落区分', '群落区分記号']),
]

# 階層ごとの項目(低木層の高さ，草本層植被率，高木第1層の植被率…)は
# 資料ごとに階層の呼び方が違うので，語彙を並べずに形で拾う
LAYER_ITEM = re.compile(r'^(.*?層)(?:の)?(高さ|植被率)')


def normalize_item(text):
    """項目名を突き合わせやすい形にする"""
    if not isinstance(text, str):
        return ''
    s = unicodedata.normalize('NFKC', text)
    # 数字は消さない．「高木第1層」「第2層」のように階層の番号を持つ項目がある
    return re.sub(r'[\s:：.．,，()（）]', '', s)


# 1文字の誤読を拾うための編集距離の上限．実測(2026-09-03)．
# `草本厨の高さ` `出現穂数` `通し務号` `調査面称` はどれも 1 文字の誤読で，
# 部分一致では拾えなかった(`厨`→`層`，`穂`→`種`，`務`→`番`，`称`→`積`)
FUZZY_MAX_DIST = 1
FUZZY_MIN_LEN = 3       # これより短い語はあいまい一致に使わない
LAYER_MAX_DIST = 2      # 階層の呼び方はここまで許す(2位との差 1 が条件)

# 階層の呼び方(`草本厨` を `草本層` に寄せるために持つ)
LAYER_WORDS = ['高木層', '高木第1層', '高木第2層', '亜高木層', '低木層',
               '低木第1層', '低木第2層', '草本層', 'コケ層', '蘚苔層']
LAYER_TAIL = {'高さ': 'height', '植被率': 'cover'}


def _fuzzy(s, words, max_dist=FUZZY_MAX_DIST, margin=0):
    """`s` に**全体として**いちばん近い語を返す(距離 `max_dist` まで)

    **部分文字列で寄せてはいけない**(2026-09-03)．`調査面称` は
    先頭 3 文字 `調査面` が `調査日` と距離 1 なので，窓を滑らせると
    `調査年月日` に当たってしまう．正しくは `調査面積`(全体で距離 1)．

    `margin` を渡すと，**2 位との差がそれ未満なら決めない**．
    階層の呼び方は互いに近い(`低木第1層` と `低木第2層` は距離 1)ので，
    距離を緩めるときは差を見ないと取り違える．
    """
    ds = sorted((Levenshtein.distance(s, w), w) for w in words
                if len(w) >= FUZZY_MIN_LEN)
    if not ds or ds[0][0] > max_dist:
        return None
    if margin and len(ds) > 1 and ds[1][0] - ds[0][0] < margin:
        return None
    return ds[0][1]


def match_item(text):
    """項目名から出力の列名を決める．分からなければ None

    まず今までどおり**部分一致**で決め，外れたときだけ**編集距離 1** まで
    許す(1文字の誤読を拾う)．あいまい一致は全体の距離で見るので，
    語の一部がたまたま似ているだけでは寄らない．
    """
    s = normalize_item(text)
    if not s:
        return None
    m = LAYER_ITEM.match(s)
    if m:
        layer, what = m.group(1), m.group(2)
        return f'{layer}_{"height" if what == "高さ" else "cover"}'
    # **階層の項目は，語彙の部分一致より先に見る**(2026-09-03)．
    # `低木厨植被率` は `植被率` を含むので，先に語彙を見ると
    # 階層を持たない `veg_cover`(植生の植被率)に落ちる
    for tail, suffix in LAYER_TAIL.items():
        if not s.endswith(tail):
            continue
        head = s[:-len(tail)].rstrip('の')
        if not head:
            break                       # 頭が無い = 植生全体の項目．語彙へ回す
        # **階層は距離 2 まで許す**(`挙本國` は `草本層` から距離 2)．
        # ただし 2 位との差が 1 以上のときだけ採る．階層の呼び方は互いに
        # 近く(`低木第1層` と `低木第2層` は距離 1)，緩めると取り違える
        near = _fuzzy(head, LAYER_WORDS, max_dist=LAYER_MAX_DIST, margin=1)
        if near:
            return f'{near}_{suffix}'
        # **階層らしき頭が付いているのに階層を決められないときは，
        # 植生全体の項目に落とさない**．`挙本國の植被率` を `veg_cover` に
        # すると，階層が消えたまま黙って表に入る．判別できないとして返す
        return None
    for key, words in ITEM_ALIASES:
        if any(w in s for w in words):
            return key

    # --- ここから下はあいまい一致 ---
    flat = [(key, w) for key, words in ITEM_ALIASES for w in words]
    near = _fuzzy(s, [w for _, w in flat])
    if near:
        for key, w in flat:
            if w == near:
                return key
    return None


def plot_table(df: pd.DataFrame, text_col: str = 'corrected') -> pd.DataFrame:
    """
    表頭のセルから，地点ごとの表を作る

    Args:
        df: OCR済みの位置決め結果(obj_name, x1, y1, text_col を持つ)
        text_col: 使う文字列の列．無ければ 'text' に落ちる
    Returns:
        1行 = 1地点 の DataFrame．
        困ったことは df.attrs['warnings'] に入れる
    """
    warnings = []
    if text_col not in df.columns:
        text_col = 'text'
    if text_col not in df.columns:
        return _empty(warnings + ['文字列の列が無いため，表頭を組み立てられない．'])

    values = df[df['obj_name'] == 'header_value']
    if values.empty:
        # 表形式のセルが無いときは，文章形式の表頭(1地点)を試す
        return _from_text(df, text_col, warnings)

    names_ja = df[df['obj_name'] == 'header_item_ja']
    names_de = df[df['obj_name'] == 'header_item']

    # 項目行は y1 で決まる．和名を優先し，無ければドイツ語で補う
    item_of = {}
    unknown = []
    for y1 in sorted(values['y1'].unique()):
        label = None
        for src in (names_ja, names_de):
            hit = src[src['y1'] == y1]
            if not hit.empty:
                label = match_item(hit.iloc[0][text_col])
                if label:
                    break
        if label is None:
            raw = ''
            for src in (names_ja, names_de):
                hit = src[src['y1'] == y1]
                if not hit.empty and isinstance(hit.iloc[0][text_col], str):
                    raw = hit.iloc[0][text_col]
                    break
            unknown.append(raw)
        item_of[y1] = label
    if unknown:
        warnings.append(
            f'項目名を判別できない行が {len(unknown)} つある: '
            f'{[u for u in unknown if u][:5]}．出力には含めない．')

    # 地点は値の x1 の並び順(組成部の列番号と同じ順になる)
    plots = {x1: i + 1 for i, x1 in enumerate(sorted(values['x1'].unique()))}

    rows = {}
    need_check = []
    for r in values.itertuples():
        key = item_of.get(r.y1)
        if key is None:
            continue
        plot = plots[r.x1]
        raw = getattr(r, text_col, None)
        # 項目ごとに取りうる形が決まっているので，そこで直して検証する
        fixed = correct_text.correct_header_value(key, raw)
        rows.setdefault(plot, {})[key] = fixed['corrected']
        if fixed['status'] == 'Need Check':
            need_check.append(f'地点{plot} {key}={raw!r}')
    if need_check:
        warnings.append(
            f'値として読めない項目が {len(need_check)} 件ある: '
            f'{need_check[:5]}．目視で確かめる．')

    if not rows:
        return _empty(warnings + ['項目名を1つも判別できず，表頭を組み立てられない．'])

    res = pd.DataFrame([{'plot': p, **v} for p, v in sorted(rows.items())])
    if 'source_image' in df.columns and len(df):
        res.insert(0, 'source_image', df['source_image'].iloc[0])
    n_empty = int(res.drop(columns=['plot']).isna().all(axis=1).sum())
    if n_empty:
        warnings.append(f'値が1つも読めなかった地点が {n_empty} ある．')
    res.attrs['warnings'] = warnings
    return res


def _from_text(df, text_col, warnings):
    """文章形式の表頭(1地点)から作る

    1地点の表では表頭が表に組まれず，
    「調査番号：SO-216, 調査面積：100m²…」という流し込みの文章になる．
    領域を丸ごと読んだ文字列を解析する．
    """
    head = df[df['obj_name'] == 'header']
    if head.empty:
        return _empty(warnings + ['表頭が無い．'])
    text = head.iloc[0].get(text_col)
    if not isinstance(text, str) or not text.strip():
        return _empty(warnings + ['表頭を読めていない(文字列が空)．'])
    items = parse_text.parse_header_text(text, ITEM_ALIASES)
    if not items:
        return _empty(warnings + ['表頭の文章から項目を1つも取り出せなかった．'])
    row, need_check = {}, []
    for key, value in items.items():
        fixed = correct_text.correct_header_value(key, value)
        row[key] = fixed['corrected']
        if fixed['status'] == 'Need Check':
            need_check.append(f'{key}={value!r}')
    if need_check:
        warnings.append(
            f'値として読めない項目が {len(need_check)} 件ある: '
            f'{need_check[:5]}．目視で確かめる．')
    res = pd.DataFrame([{'plot': 1, **row}])
    if 'source_image' in df.columns and len(df):
        res.insert(0, 'source_image', df['source_image'].iloc[0])
    res.attrs['warnings'] = warnings
    return res


def _empty(warnings):
    res = pd.DataFrame(columns=['source_image', 'plot'])
    res.attrs['warnings'] = warnings
    return res


if __name__ == '__main__':
    for s in ['通し番号', '調査番号', '調査年月日', '海抜高度', '方 位', '植被率',
              '出現種数', '傾  斜', '順  斜', '低木層の高さ', '低木層植被率',
              '草本層の高さ', '高木第1層の植被率', 'コケ層の植被率', 'ヤブツバキ']:
        print(f'{s:16} -> {match_item(s)}')
