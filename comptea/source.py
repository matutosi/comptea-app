"""格子を作ったのと同じ画像を開く (source.py)

**格子の座標は，工程が実際に見た画像のもの**です．工程は紙面の傾きを直したり
(`pipeline.grid.deskew_page` → `<名前>_deskew.png`)，横倒しを起こしたり
(`rotate_page` → `<名前>_ccw.png` / `_cw.png`) してから検出するので，
元画像とは数 px から 90 度ずれます．

元画像と突き合わせる間違いは**何度も起きています** (2026-09-11: 20_p3 の
表頭の境が「12 本字を割る」と出たが，正しい画像で測ると 0 本．
kinki_014 の切り出しが回転を忘れていた件も同じ型)．

`located.csv` と `detect.csv` には `source_image` 列があり，そこに使った画像の
場所が入っています．**格子を画像と突き合わせるときは，必ずこの関数を通す**．

    from comptea import source
    im = source.open_image(df_loc, workdir)

置き場が移っていることもある (一時ディレクトリで作った格子を別の場所から見る)
ので，記録された場所に無ければ同じ名前を `workdir` とその親から探します．
"""
import os
import re

import pandas as pd


COLUMN = 'source_image'


def recorded(df):
    """格子に記録された画像の場所 (無ければ None)"""
    if df is None or COLUMN not in getattr(df, 'columns', []):
        return None
    vals = [v for v in df[COLUMN].dropna().unique().tolist() if str(v).strip()]
    return str(vals[0]) if vals else None


def find(df, workdir=None):
    """格子を作ったのと同じ画像の場所を返す (見つからなければ None)

    記録された場所をまず見る．そこに無ければ，**同じ名前**を `workdir` と
    その親から探す (一時ディレクトリで作った格子を移したとき用)．
    """
    p = recorded(df)
    if p and os.path.exists(p):
        return p
    if not p:
        return None
    name = basename(p)
    for d in _dirs(workdir):
        q = os.path.join(d, name)
        if os.path.exists(q):
            return q
    return None


def basename(path):
    """`\\` と `/` のどちらで区切られていても名前を返す (2026-09-12)

    格子は Windows で作り，CI や別の PC (Linux) から読むことがあります．
    `os.path.basename` は Linux では `\\` を区切りと見ないので，
    `C:\\もう無い場所\\a_deskew.png` がまるごと名前になり，探せませんでした．
    """
    return re.split(r'[\\/]', str(path))[-1]


def _dirs(workdir):
    if not workdir:
        return []
    workdir = os.path.abspath(workdir)
    out = [workdir, os.path.dirname(workdir)]
    parent = os.path.dirname(os.path.dirname(workdir))
    if parent:
        out.append(parent)
    return [d for d in out if d and os.path.isdir(d)]


def open_image(df, workdir=None, fallback=None):
    """格子を作ったのと同じ画像を開く

    Args:
        df: `located.csv` か `detect.csv` を読んだもの
        workdir: 格子の置き場 (画像が移っているときの探し先)
        fallback: 見つからないときに開く画像の場所
    Returns:
        PIL の画像
    Raises:
        FileNotFoundError: どこにも見つからないとき
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    p = find(df, workdir) or fallback
    if not p or not os.path.exists(p):
        raise FileNotFoundError(
            f'格子を作った画像が見つからない (記録: {recorded(df)!r})．'
            '元画像で代用してはいけない (座標がずれる)')
    return Image.open(p)


def fits(df, size):
    """格子が，その大きさの画像に収まるか

    収まらなければ**違う画像と突き合わせている**印 (回転や傾き補正の取り違え)．
    """
    if df is None or not len(df) or not {'x2', 'y2'} <= set(df.columns):
        return True
    w, h = float(size[0]), float(size[1])
    return float(df['x2'].max()) <= w + 1 and float(df['y2'].max()) <= h + 1


def check(df, im, workdir=None):
    """突き合わせる前の点検．合わなければ理由を添えて知らせる

    Returns:
        警告のリスト (問題なければ空)
    """
    if fits(df, im.size):
        return []
    return [f'**格子が画像に収まっていない** (格子の右下 '
            f'{float(df["x2"].max()):.0f},{float(df["y2"].max()):.0f} / '
            f'画像 {im.size[0]},{im.size[1]})．'
            f'工程が見た画像は {recorded(df)!r}．'
            '傾きを直した画像や回した画像と突き合わせること '
            '(`comptea.source.open_image`)']


def load(workdir, which='located'):
    """格子と，それを作った画像をまとめて読む

    Returns:
        (格子, 画像)
    """
    f = os.path.join(workdir, f'{which}.csv')
    df = pd.read_csv(f)
    return df, open_image(df, workdir)
