"""傾きの推定は，行の周期にロックしない (deskew.estimate)

2026-09-16 に足した．組成部は行が周期的に並ぶので，ずれを広く探すと
相互相関が**行の高さの整数倍**を選ぶ．19_p2 は帯の高さの 5% (302 px) まで
探して 136 px (行の高さ 44.5 px の 3 倍) を選び，**−7.9 度**という推定に
なっていた．`DESKEW_MAX_DEG` に引っかかって捨てられるので，
**傾きがあるのに直さない**うえ，警告も出ない．

120 表で測ると，**1.5 度を超える推定が 6 表 → 0 表**になり，
発動する表は 4 → 4 で変わらなかった (下流は動かない)．
"""
import numpy as np
from PIL import Image

from comptea import deskew


def _sheet(h=600, w=900, pitch=30, shift=6):
    """行が `pitch` ごとに並び，右へ行くほど `shift` px 下がる紙面"""
    a = np.full((h, w), 255, dtype=np.uint8)
    for y0 in range(60, h - pitch, pitch):
        for x in range(w):
            y = int(y0 + shift * x / w)
            if 0 <= y < h - 4:
                a[y:y + 4, x] = 0
    return Image.fromarray(a).convert('RGB')


def test_行の高さを渡すと周期にロックしない():
    img = _sheet(pitch=30, shift=6)
    wide, dy_wide = deskew.estimate(img)
    tight, dy_tight = deskew.estimate(img, pitch=30)
    assert abs(dy_tight) <= 15                  # 半行 (15 px) の中
    assert abs(tight) < 1.5                     # 信じられる角度
    assert abs(dy_wide) >= abs(dy_tight)        # 広く探すほど大きく出る


def test_傾きの向きと大きさが合う():
    """右へ行くほど下がる紙面．`estimate` は右が下がると**負**を返す

    (`deskew_to` はこの符号を反転して回す．08_p2 で確かめた向き)
    """
    deg, dy = deskew.estimate(_sheet(shift=6), pitch=30)
    assert dy < 0
    assert 3 <= abs(dy) <= 9                    # 置いたずれ 6 px のあたり
    up, _dy = deskew.estimate(_sheet(shift=-6), pitch=30)
    assert up > 0 > deg                         # 向きが逆なら符号も逆


def test_傾きが無ければ0():
    img = _sheet(shift=0)
    deg, dy = deskew.estimate(img, pitch=30)
    assert abs(dy) <= 1 and abs(deg) < 0.2


def test_行の高さを渡さなければこれまでどおり():
    img = _sheet(shift=6)
    a = deskew.estimate(img)
    b = deskew.estimate(img, pitch=None)
    assert a == b


def test_探す範囲は5pxを下回らない():
    """行がとても低い表でも，最低限は探す"""
    img = _sheet(pitch=30, shift=6)
    deg, dy = deskew.estimate(img, pitch=1)
    assert abs(dy) <= 5


def test_小さすぎる紙面は測らない():
    small = Image.new('RGB', (100, 50), 'white')
    assert deskew.estimate(small) == (0.0, 0)
