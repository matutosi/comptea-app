import base64
import io
import numpy as np
import pandas as pd
import easyocr
from PIL import Image, ImageOps

import ink
import parse_text

def binarize_image(img):
    """
    binarize image
    """
    img_grayscale = img.convert("L")
    img_binarized = img_grayscale.convert("1")
    return img_binarized

def trim_image(img, border=20):
    """
    Trims the whitespace from the edges of an image.
    
    add_marginによる余白：ギリギリに切り取るとOCRがうまくいかないため
    Args:
        img: A Pillow Image object.
    Returns:
        A new Pillow Image object with the whitespace trimmed.
    """
    img_binarized = binarize_image(img)
    inverted_img = ImageOps.invert(img_binarized)
    bbox = inverted_img.getbbox() # Get the bounding box of the non-white content
    if bbox:
        img = img.crop(bbox) # Crop the image to the bounding box
        img = add_margin(img)
    return img

def add_margin(img, top=20, right=20, bottom=20, left=20, color='white'):
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result

READER = easyocr.Reader(['ja', 'en'])   # 作るのに時間がかかるので使い回す


def ocr_image(img, box, reader = READER, clean=False):
    # **座標の上下・左右を整える**(2026-09-03)．格子の端で高さや幅が
    # 負になる箱ができることがあり(s01115_18_p2 の表頭の 1 行)，
    # そのまま crop すると PIL が落ちて読み取りが途中で止まる
    x1, y1, x2, y2 = box
    box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    img = img.crop(box)
    if clean:
        # 組成部のセルは，セルを横切る枠線を消してから読む(2026-09-04)
        img = ink.erase_box_lines(img)
    img = trim_image(img)
    # **中身の無いセルは読ませない**(2026-09-03)．幅か高さが 0 の画像を
    # 渡すと EasyOCR の中で OpenCV が落ち，**読み取りが途中で止まる**
    # (s01115_13_p1 は 2,134 セルの途中で終わっていた)
    if img.width < 1 or img.height < 1:
        return [], img
    text = reader.readtext(np.array(img))
    return text, img

# 組成部のセルを読み直すときの決まり(2026-09-01 に測って決めた)
COMP_ALLOW = '0123456789+r.,:;・'   # 被度・群度に出る字だけ
# **括弧付きのセルには足りなかった**(2026-09-05)．常在度の列や，被度・群度を
# `2(3-4)` と組む列では，字種に括弧もローマ数字も無いので，読み直しが
# `2(3-4)` を `36+23` のように化けさせ，1 件も拾えていなかった
# (s01115_22_p2 の Need Check 46 セルで 0 件 → 字種を足すと 20 件)．
CONSTANCY_ALLOW = COMP_ALLOW + 'IViv()-'
# 「何も無いセル」と「値のあるセル」の黒画素の間に，2つの高さを置く．
#   RETRY_AT より濃ければ読み直す / THIN_AT より薄ければ1文字の値しか認めない
RETRY_AT = 0.15
THIN_AT = 0.5


def retry_comp_cell(img, box, thin: bool):
    """組成部のセルを，**文字検出を省いて**字種を絞って読み直す

    `readtext()` は文字を**探してから**読むので，`+` が1つだけ置かれた
    小さなセルでは何も見つけられない．`recognize()` は与えた画像を
    1行の文字として読むので，このとき拾える
    (2026-09-01: 単独の `+` 13 件のうち 12 件が読めた)．

    被度・群度に出る字は `5 4 3 2 1 + r ・` だけなので，字種も絞る．
    **値として読める形になったときだけ**返す(でたらめを入れないため)．
    黒画素が薄いセルは1文字の値しか入らないはずなので，
    数字が返ってきたら採らない(非出現の `・` を数字と読む誤りを止める)．

    Returns:
        読めた文字列．採らないときは ''
    """
    import correct_text

    cell = trim_image(ink.erase_box_lines(img.crop(box)))
    # 被度・群度の字種で読み，値にならなければ**括弧付きの字種**でもう一度
    # (2026-09-05)．狭い方を先に試すので，括弧の無い表の結果は変わらない
    for allow in (COMP_ALLOW, CONSTANCY_ALLOW):
        res = READER.recognize(np.array(cell), allowlist=allow)
        text = ' '.join(t[1] for t in res) if res else ''
        if not text.strip():
            continue
        fixed = correct_text.correct_cell('comp', text)
        if fixed['status'] != 'OK':
            continue
        if thin and str(fixed['corrected']) not in ('+', 'r'):
            continue
        return text
    return ''


def ocr_images_df(df):
    # groupbyで分けたDataFrameはindexが0始まりとは限らないためilocで取る
    image = df['source_image'].iloc[0]
    img = Image.open(image)
    for i, row in df.iterrows():
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        box = (x1, y1, x2, y2) # crop range: left, top, right, bottom
        text, img_ocred = ocr_image(img, box, clean=row.get('obj_name') == 'comp')
        if text:
            # 1つのセルが複数行になることがある(表頭の調査年月日など)．
            # 先頭だけ取ると残りが落ちるので，読めたものを読み順につなぐ．
            # 空白は correct_comp / correct_layer が除くので害はない
            df.loc[i, 'text'] = parse_text.reading_order(text)
            df.loc[i, 'img_base64'] = jpg2base64(img_ocred)
    return retry_empty_comp(df, image, img)


def retry_empty_comp(df, image, img):
    """読めなかった組成部のセルのうち，**字のあるものだけ**読み直す

    非出現のセルは読めなくて当たり前なので，黒画素の量で選り分ける．
    **「何も無い」の高さはページによって違う**ので，その場で測る．
      `example.jpg`  非出現は `・` が印字され 0.013，単独の `+` は 0.029
      `kinki_047`    非出現は**空白**で 0.000，単独の `+` は 0.032
    そこで「何も無い」(読めなかったセルの下位10%)と「値がある」
    (読めたセルの中央値)の間に高さを置く．
    """
    import ink

    if 'obj_name' not in df.columns:
        return df
    comp = df[df['obj_name'] == 'comp']
    if comp.empty:
        return df

    def has_text(r):
        return isinstance(r.get('text'), str) and r['text'].strip()

    empty = [i for i, r in comp.iterrows() if not has_text(r)]
    read = [i for i, r in comp.iterrows() if has_text(r)]
    dark = ink.binarize(img)
    if not empty:
        return fix_roman_heads(df, dark)

    def ratio(i):
        return ink.ratio(dark, df.at[i, 'y1'], df.at[i, 'y2'],
                         df.at[i, 'x1'], df.at[i, 'x2'])

    ratios = {i: ratio(i) for i in empty}
    blank = float(np.percentile(list(ratios.values()), 10))
    full = float(np.median([ratio(i) for i in read])) if read else blank * 6
    if full <= blank:
        full = blank * 6 or 0.05
    limit = blank + (full - blank) * RETRY_AT
    thin_at = blank + (full - blank) * THIN_AT
    n = 0
    for i in empty:
        if ratios[i] <= limit:
            continue
        box = (df.at[i, 'x1'], df.at[i, 'y1'], df.at[i, 'x2'], df.at[i, 'y2'])
        text = retry_comp_cell(img, box, thin=ratios[i] < thin_at)
        if text:
            df.loc[i, 'text'] = text
            n += 1
    if n:
        print(f'  組成部の {n} セルは，字種を絞って読み直した')
    # **値として通らなかったセルも，字種を絞って読み直す**(2026-09-04)．
    # タイプ打ちの折り込み(s01115_08_p2)では `2.2` を `ムっム`，`+.2` を `十・ム` と
    # 読み，値のあるセルの 35 個が Need Check になっていた．破線の断片を消す案は
    # 単独の `+`・`1` の細い線まで消して改悪だったので取り下げた．
    # 読み直した結果が値として通るときだけ差し替える
    # 字種を絞ると `+` を `4` と読むことがある(example.jpg で 3 セル)ので，
    # 差し替えたセルは `note` に `retry` を付けて段階2の目視に回す
    import correct_text
    done = []
    for i in read:
        if correct_text.correct_cell('comp', df.at[i, 'text'])['status'] == 'OK':
            continue
        box = (df.at[i, 'x1'], df.at[i, 'y1'], df.at[i, 'x2'], df.at[i, 'y2'])
        text = retry_comp_cell(img, box, thin=False)
        if text:
            df.loc[i, 'text'] = text
            old = df.at[i, 'note'] if 'note' in df.columns else None
            df.loc[i, 'note'] = 'retry' if not isinstance(old, str) or not old else old + ',retry'
            done.append(int(df.at[i, 'cell_id']) if 'cell_id' in df.columns else i)
    if done:
        print(f'  組成部の {len(done)} セルは，値として通らなかったので字種を絞って読み直した'
              f'(note に retry．cell_id {done[:20]}{" …" if len(done) > 20 else ""})')
    return fix_roman_heads(df, dark)


# 常在度の頭を画数から直す決まり(2026-09-06)
ROMAN_MIN_SURE = 0.6    # 読みと画数が合う割合が，この値を下回る表では直さない
ROMAN_MIN_N = 10        # 目安にするセルがこの数に満たない表では直さない
ROMAN_MAX = 3           # `III` まで．`IV`・`V` は V の有無で読みが確かなので触らない


def fix_roman_heads(df, dark, min_sure=ROMAN_MIN_SURE, min_n=ROMAN_MIN_N,
                    max_head=ROMAN_MAX):
    """常在度の頭(ローマ数字)を，画数と合わないときだけ直す

    **`II` を `I` と読む誤りは，正しい形の値になる**ので，
    読みだけでは気づけない(s01115_22_p2 で 25 セル)．
    画数は黒画素から数えられる(`ink.head_strokes()`)ので，突き合わせる．

    直すのは `I` だけでできた頭(`I`・`II`・`III`)に限る．
    `IV`・`V` は V の形が目立ち，OCR の読みが確かなため触らない．
    **画数が読みより多いときだけ**直す(少ないときは字がくっついて
    数え落としている見込みが高い)．

    **表ごとに校正する**．同じ資料でも字の間隔が違い，`IV` の I と V が
    くっついて 1 画に見える表がある(s01115_11_p1)．その表の中で
    「読みと画数が合っている割合」が `min_sure` を下回るなら，
    画数の数え方がその表に合っていないとみなして何もしない．
    """
    import correct_text

    if 'corrected' not in df.columns and 'text' not in df.columns:
        return df
    comp = df[df['obj_name'] == 'comp']
    if comp.empty:
        return df
    heads, strokes = {}, {}
    for i, r in comp.iterrows():
        raw = r.get('corrected')
        if not isinstance(raw, str) or not raw:
            raw = r.get('text')
        if not isinstance(raw, str):
            continue
        v = correct_text.correct_constancy(raw)
        if not v:
            continue
        head = v.split('(')[0]
        if head not in ('I', 'II', 'III', 'IV', 'V'):
            continue
        n = ink.head_strokes(dark, (r['x1'], r['y1'], r['x2'], r['y2']))
        if n is None:
            continue
        heads[i] = (head, v)
        strokes[i] = n
    if len(heads) < min_n:
        return df
    # 校正: `I` だけでできた頭で，読みと画数が合っている割合
    sure = [i for i, (h, _) in heads.items() if set(h) == {'I'}]
    if len(sure) < min_n:
        return df
    agree = sum(1 for i in sure if strokes[i] == len(heads[i][0]))
    if agree / len(sure) < min_sure:
        return df
    fixed = []
    for i in sure:
        head, v = heads[i]
        n = strokes[i]
        if n <= len(head) or n > max_head:
            continue
        new = 'I' * n + v[v.index('('):]
        df.loc[i, 'text'] = new
        if 'corrected' in df.columns:
            df.loc[i, 'corrected'] = new
        old = df.at[i, 'note'] if 'note' in df.columns else None
        df.loc[i, 'note'] = ('roman' if not isinstance(old, str) or not old
                             else old + ',roman')
        fixed.append((int(df.at[i, 'cell_id']) if 'cell_id' in df.columns else i,
                      v, new))
    if fixed:
        print(f'  常在度の頭を {len(fixed)} セルで画数から直した'
              f'(note に roman．例 {[f"{a}:{b}->{c}" for a, b, c in fixed[:5]]})')
    return df


def jpg2base64(img):
    """
    PILのImage.openで読み込んだjpg画像をbase64に変換
    
    st.data_editorで画像を表示するために使用
    """
    buffer = io.BytesIO() # 画像をメモリ上でバイトデータに保存
    img.save(buffer, format="JPEG") # 画像をJPEG形式でバッファに保存
    img_bytes = buffer.getvalue() # get byte data from buffer
    base64_encoded_data = base64.b64encode(img_bytes) # encode to Base64
    base64_string = base64_encoded_data.decode('utf-8') # decode from byte Base64 to string
    data_url = f"data:image/jpeg;base64,{base64_string}" # Data URL形式に整形
    return data_url


if __name__ == "__main__":
    reader = easyocr.Reader(['ja','en'])
    path = "located.csv"
    df_all = pd.read_csv(path)
    df_image = df_all.groupby('source_image')
    for _, df in df_image:
        result_df = ocr_images_df(df)
    result_df.to_csv("ocred.csv")
