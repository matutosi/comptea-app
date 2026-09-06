from PIL import Image, ImageDraw

# 位置決めで気になった点(locate.pyのnote列)の色
# 目立つ順に見てほしいので，濃い色を先に並べる
NOTE_COLORS = {
    "on_text"     : "red"   ,  # ずらしても文字に重なったまま
    "snapped"     : "orange",  # 文字を避けてずらした
    "interpolated": "gold"  ,  # 検出されず内挿した
}

OBJ_COLORS = {
    "row"           : "purple",
    "col"           : "purple",
    "comp"          : "purple",
    "sname"         : "blue"  ,
    "species_col"   : "red"   ,
    "layer"         : "green" ,
    "summary"       : "gray"  ,   # 右端の群の要約の列(常在度など．地点ではない)
    # 表頭(項目行 × 地点)と，文章として読む領域
    "header_value"  : "orange",
    "header_item"   : "brown" ,
    "header_item_ja": "brown" ,
    "header"        : "olive" ,
    "once_species"  : "teal"  ,
}


def note_color(note, colors=None):
    """
    note(';'区切り)に対応する色．気になる点が無ければNone

    複数あるときはNOTE_COLORSの並び順で先にあるものを採る
    """
    if colors is None:
        colors = NOTE_COLORS
    if not note or not isinstance(note, str):
        return None
    parts = note.split(';')
    for key, color in colors.items():
        if key in parts:
            return color
    return None


def draw_rects_df(df, colors=None, lwd=2, note_colors=None, note_lwd=4):
    """
    dfをもとにobj_nameで異なる色の矩形を描画

    画像(img_path)ごとに分けて，描画はdraw_rects()で実行する

    Args:
        df (pd.DataFrame): source_image, x1, y1, x2, y2, obj_nameの列
            note列があれば，気になるセルはその色で太く描く
    """
    for img_path, group_df in df.groupby('source_image'):
        cols = ['x1', 'y1', 'x2', 'y2', 'obj_name']
        if 'note' in group_df.columns:
            cols.append('note')
        image = draw_rects(img_path, group_df[cols], colors=colors, lwd=lwd,
                           note_colors=note_colors, note_lwd=note_lwd)
        return image


def draw_rects(img_path, rects_df, colors=None, lwd=2, note_colors=None, note_lwd=4):
    """
    dfをもとにobj_nameで異なる色の矩形を描画

    note列があるセルは，そちらの色を優先して太く描く．
    内挿した行やずらした境界を目で確かめられるようにするため．

    Args:
        rects_df (pd.DataFrame): x1, y1, x2, y2, obj_nameの列(noteは任意)
        colors (dict, optional): obj_nameと色の対応を示す辞書。Noneの場合デフォルトの色
        note_colors (dict, optional): noteと色の対応．Noneの場合NOTE_COLORS
        note_lwd: noteが付いたセルの線の太さ
    """
    if colors is None:
        colors = OBJ_COLORS
    try:
        image = Image.open(img_path).convert("RGB")
    except (FileNotFoundError, OSError):
        print(f"Error: Image not found at {img_path}")
        return None
    draw = ImageDraw.Draw(image)
    has_note = 'note' in rects_df.columns
    for _, rect in rects_df.iterrows():
        color = colors.get(rect['obj_name'])
        width = lwd
        if has_note:
            marked = note_color(rect['note'], note_colors)
            if marked is not None:
                color, width = marked, note_lwd
        if color is None:
            continue
        # 上下・左右が入れ替わった箱でも描く．表頭の項目行が字の行から
        # 作り直されるとき，隣り合う行の境が数画素すれ違って y2 < y1 に
        # なることがあり，そのまま渡すと PIL が例外を投げて**段階1の絵ごと
        # 出なくなる**(2026-09-02．地点 37 の表で起きた)．
        # 見るための絵なので，向きを直して描き，判断は overlay に任せる
        x1, x2 = sorted((rect['x1'], rect['x2']))
        y1, y2 = sorted((rect['y1'], rect['y2']))
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)
    return image  # not draw
