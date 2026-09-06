"""段階1: 検出と格子を作り，目視できる画像と要約を出す

    python cli/run_pipeline.py IMAGE [--conf 30] [--conf-col 20] [--imgsz 1280]
    python -m comptea.pipeline.grid IMAGE ...        (同じもの)
    pipeline.run('grid', [IMAGE, ...])               (同じプロセスで呼ぶ)

出力(すべて work/<画像名>/ の下)
    detect.csv    YOLO の検出結果
    located.csv   格子(1行 = 1セル)．col / row / cell_id 付き
    overlay.png   検出と格子を描いた画像(これを見て判断する)
    summary.txt   要約と警告(標準出力と同じ内容)

**1ページに表が2つ以上あるときは，表ごとに別の置き場を作る**
(`work/<画像名>_t1/`・`work/<画像名>_t2/`)．
別々の表なので地点も表頭も違う．1つにまとめてはいけない．
以後の工程はこの置き場を1つずつ渡す(工程の側は表が2つあることを知らない)．

このスクリプトは**判断しない**．測った結果と警告を出すだけで，
妥当かどうかは overlay.png を見て決める(references/checkpoints.md)．
"""
import argparse
import sys

from . import common as _common


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='検出と格子を作る(段階1)')
    p.add_argument('image', help='組成表の画像')
    p.add_argument('--workdir', default=None, help='中間物の置き場(既定はいまいる場所の work/<画像名>)')
    # 既定はパッケージに同梱した重み(どこから呼んでも見つかる)
    p.add_argument('--weights', default=None,
                   help='重み(既定は同梱の comptea.pt)')
    # 30 は学習外を含む77枚で決めた値(.claude/CLAUDE.md 参照)
    p.add_argument('--conf', type=float, default=30, help='信頼度の閾値(%%)')
    # 20 は手元の全88枚で決めた値．30 では col が1本も取れない段が5つあり，
    # その段の組成部が丸ごと空になっていた(2026-08-24)
    p.add_argument('--conf-col', type=float, default=20, help='col だけの閾値(%%)')
    # 学習時と同じ**縮尺**で推論する．ずれると成績が大きく落ちる．
    # `auto` は画像の長辺から決める(長辺 3300 px なら 1280 で，
    # 手元の 88 枚はこれまでと同じ値になる)．大きな折り込みでは上がる
    p.add_argument('--imgsz', default='auto',
                   help="推論の大きさ．'auto' か整数(既定 auto)")
    p.add_argument('--threth-col', type=float, default=0.8, help='列の重複除去')
    p.add_argument('--threth-row', type=float, default=0.8, help='行の重複除去')
    p.add_argument('--no-snap', action='store_true', help='境界を谷へずらさない')
    p.add_argument('--no-resplit', action='store_true',
                   help='左右の切れ目で部分画像に分けてやり直さない(再帰の中で使う)')
    return p.parse_args(argv)


# 行が1件も取れなかった表を，この閾値(%)で1度だけ拾い直す．
# 数行しかない小さな表では row の信頼度が上がりきらない
# (kinki_037 の Tab.65 は本文4行で，最も高いものでも 0.26)．
# 拾い直すのは**行が0件の表だけ**なので，他のページの結果は変わらない．
ROW_RETRY_CONF = 10


def resolve_imgsz(image, value, quiet=False):
    """`--imgsz auto` を画像の大きさから決める(数で渡されていればそのまま)

    上限で頭打ちになったら知らせる．そのときは学習時より縮尺が小さく，
    行が取れないことがある(大きな折り込みの，地点数の多い表)．
    `image` は場所でも，開いた画像(PIL)でもよい．
    """
    if str(value).lower() != 'auto':
        return int(value)
    from PIL import Image
    from comptea import split_sheet
    Image.MAX_IMAGE_PIXELS = None
    w, h = image.size if isinstance(image, Image.Image) else Image.open(image).size
    size = split_sheet.auto_imgsz(w, h)
    note = ('  ← 上限で頭打ち．学習時より縮尺が小さいので行が取れないことがある'
            if split_sheet.imgsz_capped(w, h) else '')
    if not quiet:
        print(f'imgsz   : {size} (auto．画像 {w} x {h}){note}')
    return size


# 1枚から取り出す検出の上限．ultralytics の既定は 300 で，**大きな表では足りない**
# (地点 37 の表を短冊に分けると，1本の短冊で row 217 + plot_row 72 + col …
#  が出て 300 でちょうど頭打ちになり，行が途中から消えていた．2026-09-02)
MAX_DET = 5000

# 部分画像に切り出すとき，これより狭い切れ端は作らず，これより近い切れ目は
# 1つにまとめる(px)．種名の列 1 本ぶんより狭い表は無い
RESPLIT_MIN_W = 200


def detect(image, weights, conf, by_class, imgsz, source=None):
    """YOLO で検出し，クラスごとの閾値で切り直す

    Args:
        image   : 画像の場所か，開いた画像(PIL)
        by_class: クラス名 -> 閾値(0.0-1.0)．ここに無いクラスは conf を使う
        source  : `source_image` に残す名前(開いた画像を渡すときに使う)
    """
    from ultralytics import YOLO
    from comptea import detect as detect_mod

    model = YOLO(weights)
    # 低い方の閾値で推論し，クラスごとの閾値で切り直す
    floor = min([conf] + list(by_class.values()))
    results = model.predict(image, conf=floor, imgsz=imgsz, verbose=False,
                            max_det=MAX_DET)
    results = detect_mod.filter_by_conf(results, conf, by_class)
    df = detect_mod.split_box_column(results[0].to_df(decimals=2))
    df['source_image'] = str(source if source is not None else image
                             ).replace('\\', '/')
    df['model'] = weights
    return df


def label_grid(img, df):
    """格子に行番号・列番号を焼き込む

    番号があると，overlay を見て「何行目がずれている」と指し示せる．
    段階2で読み取りを頼むときの目印にもなる．
    """
    from PIL import ImageDraw, ImageFont

    comp = df[df['obj_name'] == 'comp']
    if comp.empty:
        return img
    draw = ImageDraw.Draw(img)
    size = max(12, int(comp['y2'].sub(comp['y1']).median() * 0.5))
    try:
        font = ImageFont.truetype('arial.ttf', size)
    except OSError:
        font = ImageFont.load_default()

    def put(x, y, text, anchor):
        draw.text((x, y), text, fill='blue', font=font, anchor=anchor,
                  stroke_width=max(1, size // 8), stroke_fill='white')

    for block, g in comp.groupby('block'):
        left, top = g['x1'].min(), g['y1'].min()
        for row, gr in g.groupby('row'):
            put(left - 6, (gr['y1'].iloc[0] + gr['y2'].iloc[0]) / 2, str(int(row)), 'rm')
        for col, gc in g.groupby('col'):
            put((gc['x1'].iloc[0] + gc['x2'].iloc[0]) / 2, top - 6, str(int(col)), 'md')
    return img


def summarize(df_det, df_loc, args):
    """判断の材料になる数だけを並べる"""
    out = [f'image   : {args.image}',
           f'weights : {args.weights}  imgsz={args.imgsz}  '
           f'conf={args.conf:.2f} (col {args.conf_col:.2f})']

    out.append('--- 検出 ---')
    for name, n in df_det['obj_name'].value_counts().items():
        conf = df_det.loc[df_det['obj_name'] == name, 'confidence']
        out.append(f'  {name:<14} {n:>4}   conf {conf.min():.2f}-{conf.max():.2f}')
    for name in ('layer', 'header', 'header_col', 'once_species'):
        if name not in set(df_det['obj_name']):
            out.append(f'  {name:<14}    0   ← 検出なし')

    out.append('--- 格子 ---')
    blocks = sorted(df_loc['block'].unique())
    out.append(f'  段: {len(blocks)}')
    comp = df_loc[df_loc['obj_name'] == 'comp']
    for b in blocks:
        g = comp[comp['block'] == b]
        if g.empty:
            out.append(f'  段{b}: comp なし')
            continue
        out.append(f'  段{b}: {g["row"].nunique()} 行 x {g["col"].nunique()} 列 '
                   f'= {len(g)} セル (行番号 {g["row"].min()}-{g["row"].max()})')
    for name, n in df_loc['obj_name'].value_counts().items():
        if name != 'comp':
            out.append(f'  {name:<14} {n:>4}')

    note = df_loc.get('note')
    if note is not None:
        marks = {}
        for v in note.fillna(''):
            for part in filter(None, str(v).split(';')):
                marks[part] = marks.get(part, 0) + 1
        if marks:
            out.append('--- 気になるセル(overlay で色が付く) ---')
            for k, n in sorted(marks.items()):
                out.append(f'  {k:<14} {n:>4}')
    return '\n'.join(out)


class TableFailed(Exception):
    """この表は格子にできなかった(他の表は続ける)

    理由だけでは何が起きたか分からないので，**警告も一緒に運ぶ**．
    """

    def __init__(self, reason, warnings=()):
        super().__init__(reason)
        self.warnings = list(warnings)


def build_one(image, df_det, work, args, table_no=None, n_tables=1):
    """表1つぶんの格子を作り，成果物を work へ書く

    Args:
        table_no: 1ページに表が2つ以上あるときの表番号(1つなら None)
    Returns:
        (要約の文字列, 警告のリスト)
    Raises:
        TableFailed: 格子を作れなかったとき
    """
    from comptea import draw_rect
    from comptea import locate
    from comptea import filters
    from comptea import blocks
    from PIL import Image

    work.mkdir(parents=True, exist_ok=True)
    if table_no is not None:
        # build_table.py がこれを読み，長い表に table 列を足す．
        # 画像のパスは同じで地点番号も表ごとに 1 から振り直されるため，
        # この印が無いと，後で束ねたときに別の表の地点と衝突する
        (work / 'table.txt').write_text(f'{table_no}/{n_tables}', encoding='utf-8')
    df_det.to_csv(work / 'detect.csv', index=False)

    if filters.looks_like_fragment(df_det):
        # 表頭も種名の列も無い．切り分けで残った表題・凡例だけの帯など．
        # 進めると数行 x 2 列の格子ができ，表が1つ多く数えられる
        raise TableFailed(
            'この画像には表頭も種名の列も検出されない(切れ端とみなす)．'
            '表題や凡例だけの帯なら，これで正しい．'
            '表が写っているなら --conf を下げて試す', [])
    # 表頭の外に散った項目行は，紙面全体の検出でも捨てる．残すと組成部の
    # 行が表頭の値として二重に切られ，表頭の帯が全高に広がって
    # 階層の列の判定(表頭が空)まで壊れる(s01115_18_p2．2026-09-04)
    from comptea import body_rows
    from comptea import checks
    from comptea import col_edges
    from comptea import strips
    from comptea import table_split
    df_det, stray_warn = table_split.drop_stray_plot_rows(df_det, heads=False)
    # 表頭の中に出た種名の列は捨てる(段を 1 つ増やし，本体から和名が消える)
    df_det, n_names = table_split.drop_stray_name_cols(df_det)
    if n_names:
        stray_warn.append(
            f'表頭の中にあった種名の列の検出 {n_names} 本を捨てた'
            '(凡例や表題の字を種名の列と見た誤検出．段が余分に増える)．')
    # 種名の列が 1 つも検出されなければ，組成部の左の黒画素の帯から補う
    # (s01115_08_p1・08_p2・18_p2．2026-09-04)
    from comptea import name_col
    df_det, name_warn = name_col.name_columns_from_ink(Image.open(image), df_det)
    stray_warn += name_warn
    df_det.to_csv(work / 'detect.csv', index=False)
    df_loc = locate.locate_items(
        df_det, threth_col=args.threth_col, threth_row=args.threth_row,
        image=image, snap=not args.no_snap)
    warnings = stray_warn + list(df_loc.attrs.get('warnings', []))
    if len(df_loc) == 0:
        raise TableFailed('格子を作れなかった．下の警告を読む', warnings)
    if filters.looks_like_no_table(df_loc):
        # 表のないページに枠がいくつか出ただけ．そのまま進むと
        # 空の段ができ，中身の無い表が黙って下流へ流れる
        raise TableFailed(
            'このページに組成表が見あたらない(組成セルも列も作れなかった)．'
            '本文や写真だけのページなら，これで正しい．'
            '表が写っているなら --conf-col を下げて試す', warnings)

    df_loc = locate.number_cells(df_loc)
    # 階層の列が組成部の先頭に紛れ込んでいたら，地点の列から外す
    from comptea import layer_col
    df_loc, layer_warn = layer_col.fix_columns(Image.open(image), df_loc)
    warnings += layer_warn
    # 地点が多い表では，列の境が1つずれても目では気づけない．
    # 隙間の位置は検出とは別の測り方で出せるので，突き合わせて知らせる
    # **行が大きく落ちていたら，組成部の谷から決め直す**．
    # 行の検出は地点の多い表で崩れるが，組成部は非出現でも「・」があるので，
    # どの行にも字がある(2026-09-02 ユーザ指摘)
    # **列は作り直さない**．短冊に分けない表では `col` の検出の方が確かだった
    # (2026-09-02 に測った．列の一致 38 → 37，セル 271,176 → 259,224)
    redo, warn = body_rows.rows_from_body(image, df_det, df_loc)
    if redo is not None:
        df_loc2 = locate.locate_items(
            redo, threth_col=args.threth_col, threth_row=args.threth_row,
            image=image, snap=not args.no_snap)
        if len(df_loc2):
            df_det = redo
            df_loc = locate.number_cells(df_loc2)
            df_loc, lw = layer_col.fix_columns(Image.open(image), df_loc)
            # 作り直した格子の警告を集める．捨てていたので，再 locate で行が
            # 落ちても記録に残らなかった(2026-09-03)．
            # **1 回目の格子の警告は捨てる**．行が大きく落ちた格子への警告
            # (「種名の列は格子より 141 行ぶん下まで伸びている」など)で，
            # 決め直したあとには当てはまらない(2026-09-04)
            warnings = (stray_warn + warn
                        + list(df_loc2.attrs.get('warnings', [])) + lw)
    # 行が決まったあとに，列の境だけを印字の隙間から組み直す(良いときだけ)
    df_loc, edge_warn = col_edges.fix_column_edges(Image.open(image), df_loc)
    warnings += edge_warn
    # そのうえで，境が字を割らない位置へ数 px ずらす(列の数は変えない)
    df_loc, cross_warn = col_edges.fix_edges_by_crossings(Image.open(image), df_loc)
    warnings += cross_warn
    # 表頭の列を本体に合わせる(本体で捨てた列が表頭に残ると，地点がずれる)
    df_loc, align_warn = col_edges.align_header_columns(df_loc)
    warnings += align_warn
    warnings += checks.check_grid_columns(image, df_loc)
    # **行も測る**．列だけを見ていたので，行が9割落ちても黙って通っていた
    warnings += checks.check_grid_rows(image, df_loc, df_det=df_det)
    # 行の数・列の数が合っていても壊れている形がある(2026-09-03・04)
    warnings += checks.check_row_heights(df_loc)
    warnings += checks.check_header_rows(df_loc)
    # 段階2で「このセルを読み直す」と指せるように通し番号を振る
    df_loc.insert(0, 'cell_id', range(1, len(df_loc) + 1))
    df_loc.to_csv(work / 'located.csv', index=False)

    img = draw_rect.draw_rects_df(df_loc)
    if img is None:
        img = Image.open(image).convert('RGB')
    label_grid(img, df_loc).save(work / 'overlay.png')
    return summarize(df_det, df_loc, args), warnings


def resplit_parts(image, tables, base, args):
    """検出で左右の切れ目が見つかった表を，部分画像に切り出して最初からやり直す

    左右に並んだ表の仕切りは細く，画像の空白では切れないが，検出(種名の列と
    表頭が横にいくつ並ぶか)や黒画素(組成部の中の字の密な帯)からは分かる．
    **ここで検出を振り分けて済ませてはいけない**(2026-09-03)．

    - 全高の表が隣にあると，その部分にある縦の重なりは空白でも検出でも
      見つからない．s01115_09 は左の Tab.50 の下に小さな Tab.51 があるが，
      右に全高の Tab.71 があるため空白の帯が通らず，紙面全体の検出では
      縮尺が合わずに Tab.51 の箱が1つも出ない．**左右に分けたあとの
      部分画像なら**，空白の帯で Tab.50 と 51 が分かれ，Tab.51 も自分の
      縮尺で検出できる
    - s01115_23 の Tab.141 と 148 は 50 px しか離れておらず，右の種名の列は
      検出されない．組成部の中の字の密な帯がその位置を教える

    部分画像は `split_sheet.find_tables()` で空白の帯を見てから，
    1枚ずつ自分自身(このスクリプト)にかける(`--no-resplit` 付き．再帰は1段)．

    Returns:
        (残す表の検出のリスト, 警告, 再帰で書いた置き場のリスト)
    """
    import subprocess
    from comptea import ink
    from comptea import split_sheet
    from comptea import body_rows
    from comptea import checks
    from comptea import col_edges
    from comptea import strips
    from comptea import table_split
    from PIL import Image

    im = Image.open(image)
    dark = None
    keep, warnings, done = [], [], []
    n_tables = len(tables)
    # 縦に分けた表があるときの，各表の縦の持ち分(表頭の上端 → 次の表頭の上端)．
    # 全高で切ると下の表まで入り込み，同じ表が2度組み上がる(s01115_23 で
    # 下の表が 74 行の写しとして出た．2026-09-03)
    tops, spans = [], []
    for _, df in tables:
        head = df[df['obj_name'] == 'header']
        tops.append(int(head['y1'].min()) if len(head) else int(df['y1'].min()))
        # 横の広がりは列と種名の列で測る．`row` は紙面の幅いっぱいの箱なので，
        # それを入れると隣の表と必ず重なってしまう
        narrow = df[df['obj_name'].isin(('col', 'sname', 'species_col',
                                         'header_col', 'layer'))]
        src = narrow if len(narrow) else df
        spans.append((int(src['x1'].min()), int(src['x2'].max())))

    def slot_for(k, xa, xb):
        """表 k の部分画像 (xa, xb) の縦の持ち分．
        **横に重なる隣の表がある側だけ**を表頭の上端で区切る．
        重ならなければ全高(s01115_09 の右の Tab.71 は全高で，左下の Tab.51 とは
        横に重ならない．Tab.51 の表頭で切ると Tab.71 が 117 → 106 行に減った)"""
        def overlaps(m):
            a, b = spans[m]
            return min(b, xb) - max(a, xa) > (xb - xa) * 0.3
        y0 = tops[k] if k > 0 and overlaps(k - 1) else 0
        y1 = tops[k + 1] if k < len(tops) - 1 and overlaps(k + 1) else im.height
        return y0, y1

    for k, (i, df_one) in enumerate(tables):
        cuts = list(table_split.side_by_side_cuts(df_one))
        if dark is None:
            dark = ink.binarize(im)
        band = table_split.name_band_cuts(dark, df_one)
        merged = []
        for c in sorted(set(cuts) | set(band)):
            # 近すぎる切れ目は1つにまとめる(密な帯が2つに割れて，10 px の
            # 切れ端ができた．s01115_09 で 2026-09-03)
            if merged and c - merged[-1] < RESPLIT_MIN_W:
                continue
            merged.append(c)
        cuts = merged
        if not cuts:
            keep.append((i, df_one))
            continue
        how = ('種名の列と表頭が横に並ぶ' if not band else
               '組成部の中に字の密な帯(隠れた種名の列)がある')
        x0, x1 = int(df_one['x1'].min()), int(df_one['x2'].max())
        edges = [max(0, x0 - split_sheet.MARGIN)] + cuts + [min(im.width, x1 + split_sheet.MARGIN)]
        # **縦は全高で切る**．検出の縦の範囲で切ると，紙面全体の検出では
        # 箱が1つも出なかった小さな表(s01115_09 の Tab.51)が外れてしまう．
        # 縦の重なりは，切り出したあとの空白の帯が見つける
        pieces = []
        for k2 in range(len(edges) - 1):
            if edges[k2 + 1] - edges[k2] < RESPLIT_MIN_W:
                continue
            y0, y1 = slot_for(k, edges[k2], edges[k2 + 1])
            crop = im.crop((edges[k2], y0, edges[k2 + 1], y1))
            boxes = split_sheet.find_tables(ink.binarize(crop))
            if len(boxes) < 2:
                pieces.append((f'_t{i}' if n_tables > 1 else '', k2 + 1, None, crop))
                continue
            for m, (bx1, by1, bx2, by2) in enumerate(boxes, start=1):
                pieces.append((f'_t{i}' if n_tables > 1 else '', k2 + 1, m,
                               crop.crop((bx1, by1, bx2, by2))))
        warnings.append(
            f'**表 {i} は左右に {len(cuts) + 1} つの表が並んでいる**({how})．'
            f'部分画像 {len(pieces)} 枚に切り出し，1枚ずつ最初からやり直す'
            '(空白の帯で縦の重なりも見る)．段階1で境目を目で確かめる')
        for tsuf, k, m, crop in pieces:
            name = f'{base.name}{tsuf}_s{k}' + (f'p{m}' if m else '')
            png = base.parent / f'{name}.png'
            png.parent.mkdir(parents=True, exist_ok=True)
            crop.save(png)
            cmd = [sys.executable, '-m', 'comptea.pipeline.grid',
                   str(png), '--weights', args.weights,
                   '--conf', str(args.conf), '--conf-col', str(args.conf_col),
                   '--workdir', str(base.parent / name), '--no-resplit']
            if args.no_snap:
                cmd.append('--no-snap')
            print(f'\n===== 部分画像 {name} ({crop.width} x {crop.height}) =====')
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            out = (r.stdout or '') + (r.stderr or '')
            print('\n'.join('  ' + line for line in out.rstrip().splitlines()))
            for line in out.splitlines():
                if line.startswith('書いた: '):
                    done.append(line[len('書いた: '):].strip())
    return keep, warnings, done


def deskew_page(image, df_det, args, by_class):
    """紙面の傾きを直し，直した画像で検出し直す

    折り込みは 9 表で左右の行が行の高さの 0.3-1.0 倍ずれ，水平な境では片端で
    字を割る(08_p2 は `2.2` が上下の行に割れて 28 セルが読めなかった．
    2026-09-04)．傾きは検出した `col` の範囲(組成部)で測る．直した画像は
    作業ディレクトリの隣に置き，以後はそれを元画像として扱う
    (座標はすべて直した画像のもの)．

    Returns:
        (元画像か直した画像, その画像での検出, 警告)
    """
    from comptea import deskew
    from comptea import locate
    from comptea import blocks
    from comptea import table_split

    cols0 = df_det[df_det['obj_name'] == 'col']
    rows0 = df_det[df_det['obj_name'] == 'row']
    heads0 = df_det[df_det['obj_name'] == 'header']
    items0 = df_det[df_det['obj_name'] == 'plot_row']
    # 地点の多い表は最初の検出で `row` が取れない(短冊に分けて初めて取れる)．
    # そのときは項目行(`plot_row`)の高さを行の高さの代わりにする
    pitch_src = rows0 if len(rows0) >= 3 else items0
    # **表が 1 つの紙面だけ**直す(2026-09-04)．表が縦に並ぶ・左右に並ぶ紙面では
    # 組成部の箱が複数の表にまたがって角度がでたらめになり(02_p1 の部分画像で
    # −2.3°)，親の回転で部分画像の切り出しも変わる．部分画像の子の実行
    # (`--no-resplit`)でも測らない
    single0 = (not args.no_resplit
               and len(blocks.split_tables(df_det)[0]) == 1
               and len(table_split.split_side_by_side(df_det)[0]) == 1)
    if not (single0 and len(cols0) >= 3 and len(pitch_src) >= 3):
        return image, df_det, []

    base0 = _common.workdir(args.image, args.workdir, make=False)
    out0 = base0.parent / (base0.name + '_deskew.png')
    out0.parent.mkdir(parents=True, exist_ok=True)
    # y は**表頭の下端から `col` の下端**(組成部)で測る．`col` の箱ごと測ると
    # 表頭の値の並びが混ざり，符号まで逆になる(08_p2 は col の箱で +0.36°，
    # 組成部で −0.83°)．`row` の検出の範囲は表の一部しか覆わないことがあり
    # (08_p2 は 5 本で 110 px)，そこで測ると小さく出る
    by0 = heads0['y2'].max() if len(heads0) else cols0['y1'].min()
    by1 = cols0['y2'].max()
    box0 = (cols0['x1'].min(), by0, cols0['x2'].max(), by1)
    pitch0 = float((pitch_src['y2'] - pitch_src['y1']).median())
    image2, skew_deg = deskew.deskew_to(image, out0, box0, pitch0)
    if not skew_deg:
        return image, df_det, []

    df_det2 = detect(image2, args.weights, args.conf / 100, by_class, args.imgsz)
    # **直して検出が減ったら元に戻す**(2026-09-04)．回すと種名の列や
    # 表頭が検出されなくなる紙面があり(04_p2 は 122 行 → 34 行，
    # 23_p1_t1_s2 は 231 → 116，18_p1 は表ごと消えた)，傾きを直す
    # 利益より大きい．目印の検出数と `col` の数が減らないときだけ採る
    keys = ('sname', 'species_col', 'header', 'header_col', 'once_species')
    n1 = int(df_det['obj_name'].isin(keys).sum())
    n2 = int(df_det2['obj_name'].isin(keys).sum())
    c1 = int((df_det['obj_name'] == 'col').sum())
    c2 = int((df_det2['obj_name'] == 'col').sum())
    # 目印は 1 つの増減なら検出のゆらぎ(08_p2 は 3 → 2 で結果は正しかった)
    if not df_det2.empty and n2 >= n1 - 1 and c2 >= c1 * 0.8:
        return image2, df_det2, [
            f'**紙面の傾き {skew_deg:+.2f}° を直して検出し直した**．'
            f'座標は直した画像 {out0.name} のもの．'
            '段階1で行の境が左右で字を割っていないか見る']
    if out0.is_file():
        out0.unlink()                      # 使わない回転画像は残さない
    return image, df_det, [
        f'紙面の傾き {skew_deg:+.2f}° を測ったが，直すと検出が減る'
        f'(目印 {n1} → {n2}，col {c1} → {c2})ので直さなかった．'
        '行の境が左右で字を割っていたら，段階1で知らせる']


def split_page(image, df_det, args, base):
    """1 枚の紙面を，表ごとの検出に分ける

    縦に重なった表・左右に並んだ表・部分画像に切り出してやり直すものを，
    ここですべて片づける．**左右に並んだ表は検出の手掛かりで分ける**
    (表と表のあいだが 1 つの表の内部の空白より狭いことがあり，
    s01115_23 は 50 px．画像の空白だけでは分けられない)．

    Returns:
        ([(表の番号, 検出), ...], 元の表の数, 別に片づいた作業ディレクトリ, 警告)
    """
    from comptea import locate
    from comptea import blocks
    from comptea import filters
    from comptea import table_split

    # 種群を覆った偽の表頭は，表を分ける前に捨てる(locate_items も同じ選別をする)
    df_det, n_heads = filters.drop_unsupported_headers(df_det)
    tables, warnings = blocks.split_tables(df_det)
    warnings = list(warnings or [])
    if n_heads:
        warnings.append(
            f"項目行にも項目名の列にも重ならない 'header' の検出 {n_heads} 本を捨てた"
            '(組成部の最初の種群を表頭として覆っていることがある)．')

    sub_done = []
    n_orig = len(tables)          # 元の表の数．番号はこれで振る
    tables = list(enumerate(tables, start=1))
    if not args.no_resplit:
        # 切れ目があれば部分画像に切り出して最初からやり直す(再帰は1段)．
        # 抜けた表があっても，残った表は元の番号(_t2 など)を保つ
        tables, resplit_warn, sub_done = resplit_parts(image, tables, base, args)
        warnings += resplit_warn

    sided = []
    for idx, df_one in tables:
        parts, side_warn = table_split.split_side_by_side(df_one)
        warnings += side_warn
        sided.extend((idx, part) for part in parts)
    if len(sided) != n_orig or sub_done:
        # 左右に分かれて増えたときは，番号を振り直す(再帰で抜けたときは保つ)
        if not sub_done:
            sided = list(enumerate((df for _, df in sided), start=1))
            n_orig = len(sided)
    return sided, n_orig, sub_done, warnings


def widen_tables(image, tables, n_orig, sub_done, args, by_class):
    """地点が多すぎて行が取れない表を，短冊に分けて検出し直す

    手元の 88 枚はこの経路に入らない．

    Returns:
        ([(表の番号, 検出), ...], 警告)
    """
    from comptea import strips

    def detect_strip(strip, label):
        return detect(strip, args.weights, args.conf / 100, by_class,
                      resolve_imgsz(strip, 'auto', quiet=True), source=label)

    out, warnings = [], []
    for idx, df_one in tables:
        if not strips.needs_strips(df_one):
            out.append((idx, df_one))
            continue
        # 表が縦に重なっているときは，短冊をこの表の高さだけで切る．
        # 全高で切ると別の表の箱まで拾い直し，先に分けた表がまた1つに戻る
        # (s01115_19_p2 の右側は縦に 2 つ重なり，下の表の行が 1回出現種の
        # 帯で切れて丸ごと捨てられていた．2026-09-03)
        # **元の表の数で判定する**(2026-09-04)．`len(tables)` だと，左右に
        # 分かれた表を部分画像に切り出して残りが 1 つになったとき偽になり，
        # 短冊が全高で切られて上の表を拾い直す(s01115_23_p1_t2 は Tab.149
        # だけのはずが，Tab.148 の 454-6000 px も含む 134 行の格子になった)
        y_range = ((float(df_one['y1'].min()), float(df_one['y2'].max()))
                   if (n_orig > 1 or sub_done) else None)
        wide, warn = strips.detect_wide(image, df_one, detect_strip,
                                        y_range=y_range)
        warnings += warn
        out.append((idx, df_one if wide is None else wide))
    return out, warnings


def build_tables(image, tables, base, args, n_orig, by_class, table_warnings):
    """表ごとに格子を組み，まとめを書く

    Returns:
        (できた作業ディレクトリ, できなかった (番号, 理由))
    """
    from comptea import locate
    from comptea import blocks

    retry = None   # 行の閾値を下げた検出(要るときだけ1度作る)
    done, failed = [], []
    for i, df_one in tables:
        single = n_orig == 1
        work = base if single else base.with_name(f'{base.name}_t{i}')
        table_no = None if single else i
        label = '' if single else f'\n===== 表 {i} / {n_orig} =====\n'
        extra = []
        try:
            text, warnings = build_one(image, df_one, work, args, table_no, n_orig)
        except TableFailed as e:
            if (df_one['obj_name'] == 'row').sum() == 0:
                if retry is None:
                    retry = detect(image, args.weights, args.conf / 100,
                                   {**by_class, 'row': ROW_RETRY_CONF / 100},
                                   args.imgsz)
                    retry = blocks.split_tables(retry)[0]
                df_one = retry[i - 1] if len(retry) == n_orig else None
            else:
                df_one = None
            if df_one is None or (df_one['obj_name'] == 'row').sum() == 0:
                # 表が1つなら，最後の SystemExit が同じ理由を出す
                if not single:
                    print(f'{label}{e}')
                _common.show_warnings(table_warnings + e.warnings)
                failed.append((i, str(e)))
                continue
            extra = [f'行が1件も検出されなかったので，行だけ閾値を '
                     f'{ROW_RETRY_CONF}% に下げて拾い直した'
                     f'({(df_one["obj_name"] == "row").sum()} 本)．'
                     '**行の位置は確かでない**．段階1で overlay を目で確かめる']
            try:
                text, warnings = build_one(image, df_one, work, args, table_no, n_orig)
            except TableFailed as e2:
                if not single:
                    print(f'{label}{e2}')
                _common.show_warnings(table_warnings + extra + e2.warnings)
                failed.append((i, str(e2)))
                continue
        warnings = extra + warnings
        print(label + text)
        _common.show_warnings(table_warnings + warnings)
        (work / 'summary.txt').write_text(
            text + '\n' + '\n'.join(f'! {w}' for w in table_warnings + warnings),
            encoding='utf-8')
        print(f'\n書いた: {work}')
        done.append(work)
    return done, failed


def main(argv=None):
    args = parse_args(argv)
    image, = _common.setup([args.image])
    if not args.weights:
        import comptea
        args.weights = comptea.WEIGHTS
    args.imgsz = resolve_imgsz(image, args.imgsz)

    by_class = {'col': args.conf_col / 100}
    df_det = detect(image, args.weights, args.conf / 100, by_class, args.imgsz)
    if df_det.empty:
        raise SystemExit(
            '検出が0件．**このページに組成表が無い**ことが多い'
            '(本文・写真・隣のページから続く流し込みだけのページ)．'
            '画像を Read して確かめ，表があるなら conf を下げるか weights を疑う')

    image, df_det, skew_warn = deskew_page(image, df_det, args, by_class)

    base = _common.workdir(args.image, args.workdir, make=False)
    tables, n_orig, sub_done, warn = split_page(image, df_det, args, base)
    table_warnings = skew_warn + warn

    tables, warn = widen_tables(image, tables, n_orig, sub_done, args, by_class)
    table_warnings += warn
    _common.show_warnings(table_warnings, head='--- このページの成り立ち ---')

    done, failed = build_tables(image, tables, base, args, n_orig, by_class,
                                table_warnings)

    from pathlib import Path
    done += [Path(d) for d in sub_done]
    if not done:
        raise SystemExit(failed[0][1] if failed else '格子を作れなかった')
    if failed:
        print(f'\n表 {", ".join(str(i) for i, _ in failed)} は格子にできなかった'
              '(上の理由を読む)')
    print('\n次: 下の overlay.png を見て判断する(段階1)')
    for work in done:
        print(f'  {work / "overlay.png"}')
    if len(done) > 1:
        print('以後の工程は，この置き場を**1つずつ**渡す'
              '(別々の表なので，まとめない)')




if __name__ == '__main__':      # python -m comptea.pipeline.<段>
    sys.exit(main())
