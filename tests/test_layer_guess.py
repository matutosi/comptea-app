"""検出されなかった階層の列を，隙間の中の**字のかたまり**から補う

いままでは種名の列と組成部の隙間を**丸ごと**測っていた．そのため
  - 隙間が広く記号が一部にしかない表で，比が薄まって「階層なし」になる
    (07_p3: 隙間 222 px に記号は 75 px だけ．比 0.25 で閾値 0.45 に届かない)
  - 隙間に検出されなかった地点の列が入ると，階層の列が組成部まで広がる
    (03_p1: 階層の枠が 561 px になり，202 行 x 3 列が階層と誤判定)
"""
import pandas as pd
from PIL import Image

from comptea import locate


Y_EDGES = [float(v) for v in range(100, 100 + 34 * 12 + 1, 34)]
NAME = [pd.DataFrame({'x1': [10.0], 'x2': [300.0]})]
X_EDGES = [700.0, 760.0, 820.0, 880.0]      # 組成部は 700 から


def _fill(px, x1, x2, y1, y2):
    for x in range(int(x1), int(x2)):
        for y in range(int(y1), int(y2)):
            px[x, y] = 0


def _sheet(sym=(500, 560), extra=(), rule_at=None, name_tail=None, size=(950, 600),
           tail_rows=(2, 5, 9)):
    """種名 (10-300) と組成部 (700-) のあいだに，記号を `sym` に置いた紙面

    extra: 隙間の中の余分なかたまり (検出されなかった地点の列など) の x 範囲．
    rule_at: 縦罫線．name_tail: 種名が隙間へはみ出す範囲 (**長い和名の行だけ**．
    実データでも全行にはみ出すことはない)．
    """
    img = Image.new('L', size, 255)
    px = img.load()
    for i, (ya, yb) in enumerate(zip(Y_EDGES[:-1], Y_EDGES[1:]), 1):
        c = int((ya + yb) // 2)
        _fill(px, 20, 280, c - 8, c + 8)                    # 種名
        if sym:
            _fill(px, sym[0], sym[1], c - 8, c + 8)         # 階層の記号
        for a, b in extra:
            _fill(px, a, b, c - 3, c + 4)                   # 「・」のような小さい塊
        if name_tail and i in tail_rows:
            _fill(px, name_tail[0], name_tail[1], c - 6, c + 6)
        for xa in (720, 780, 840):
            _fill(px, xa, xa + 5, c - 3, c + 4)             # 組成の「・」
    if rule_at:
        _fill(px, rule_at, rule_at + 3, Y_EDGES[0], Y_EDGES[-1])
    return img


def _guess(img):
    return locate._guess_layer_column(img, NAME, X_EDGES, Y_EDGES)


def test_隙間が広くても記号のかたまりで測る():
    # 隙間 300-700 (400 px) に記号は 60 px だけ．丸ごと測ると薄まって落ちる
    df, guessed = _guess(_sheet(sym=(500, 560)))
    assert guessed and df is not None
    assert 480 <= float(df['x1'].iloc[0]) <= 500
    assert 560 <= float(df['x2'].iloc[0]) <= 580


def test_隙間の中の余分な列は取り込まない():
    # 検出されなかった地点の列が隙間に 3 本ある (03_p1 型)
    df, guessed = _guess(_sheet(sym=(400, 460), extra=((550, 555), (610, 615), (670, 675))))
    assert guessed and df is not None
    assert float(df['x2'].iloc[0]) <= 500          # 記号のかたまりだけ
    assert float(df['x2'].iloc[0]) - float(df['x1'].iloc[0]) <= 2.5 * 34 + 5


def test_縦罫線だけなら階層は無いとする():
    df, guessed = _guess(_sheet(sym=None, rule_at=500))
    assert df is None and not guessed


def test_隙間が空なら階層は無いとする():
    df, guessed = _guess(_sheet(sym=None))
    assert df is None and not guessed


def test_種名のはみ出しは記号と間違えない():
    # 種名の末尾が隙間へ垂れているだけ (12 行中 3 行．記号はほとんどの行にある)
    df, guessed = _guess(_sheet(sym=None, name_tail=(300, 380)))
    assert df is None and not guessed


def test_記号が和名の枠に接していても拾う():
    # 隙間の先頭から記号が始まる紙面 (079-1・06_p2・17_p1・20_p3 型)．
    # 距離で除くと落ちるので，行の割合で見分ける
    df, guessed = _guess(_sheet(sym=(300, 360)))
    assert guessed and df is not None
    assert float(df['x1'].iloc[0]) <= 305


def test_記号と種名のはみ出しが両方あれば記号を採る():
    df, guessed = _guess(_sheet(sym=(500, 560), name_tail=(300, 360)))
    assert guessed and df is not None
    assert float(df['x1'].iloc[0]) >= 400
