"""紙面の傾きを測って直す(2026-09-04)

折り込み(s01115)の 68 表のうち 9 表は，組成部の左端と右端で行が
行の高さの 0.3-1.0 倍ずれていた(08_p2 は幅 2,375 px で −23 px = −0.83°)．
水平な行の境ではどうやっても片端で字を割る(08_p2 は `2.2` が上下の行に
割れて 28 セルが読めなかった)．既存 88 枚(本のページ)は中央値 0.03 倍で無関係．

測り方は**左右の帯の黒画素の並びの相互相関**．字の行は紙面を横切って
同じ高さに並ぶので，左 1/3 と右 1/3 の行ごとの黒画素の並びをずらして
いちばん重なる所が，傾きぶんのずれになる．Hough 直線の平均(旧 `preprocess_image.py`)
は罫線の無い表では当てにならない．
"""
import numpy as np
from PIL import Image

import ink

DESKEW_MIN_ROWS = 0.3     # 左右のずれが行の高さのこの倍未満なら直さない
DESKEW_MAX_DEG = 1.5      # これを超える推定は信じない(測り損ね．実在する傾きは最大 1.06°．
                          # 04_p2 は −2.27° を採用して 122 行 → 34 行・種名 0 になった)
MAX_LAG_RATIO = 0.05      # ずれを探す範囲(帯の高さに対する比)


def _shift(a, b, max_lag):
    a = a - a.mean()
    b = b - b.mean()
    best, best_r = 0, None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = float(np.dot(a[lag:], b[:len(b) - lag]))
        else:
            r = float(np.dot(a[:lag], b[-lag:]))
        if best_r is None or r > best_r:
            best, best_r = lag, r
    return best


def estimate(img, box=None):
    """(傾きの度, 左右のずれ px) を返す．`box` は測る範囲 (x1, y1, x2, y2)"""
    dark = ink.binarize(img)
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        dark = dark[y1:y2, x1:x2]
    h, w = dark.shape
    if w < 300 or h < 100:
        return 0.0, 0
    w3 = w // 3
    left = dark[:, :w3].sum(axis=1).astype(float)
    right = dark[:, -w3:].sum(axis=1).astype(float)
    if not left.any() or not right.any():
        return 0.0, 0
    dy = _shift(left, right, max(5, int(h * MAX_LAG_RATIO)))
    return float(np.degrees(np.arctan2(dy, w - w3))), int(dy)


def deskew_to(image, out, box, pitch, min_rows=DESKEW_MIN_ROWS, max_deg=DESKEW_MAX_DEG):
    """傾きを測り，直した画像を `out` に書いてその場所を返す

    **測るのは組成部の範囲 `box`**(検出した `col` の箱の和)．紙面全体で
    測ると種名の列や表題が混ざって当てにならない(08_p2 は全体で +0.54°，
    組成部で −0.83°．+0.83° 回すと −0.04° になる)．

    **直すかどうかは角度でなく，左右のずれが行の高さ `pitch` の何倍か**で
    決める(2026-09-04)．角度 0.15° で切ると本のページ 17 枚にも発動し，
    検出し直しで列や表頭の行が減った(kinki_029 は 5 列 → 3 列)．
    本のページは 0.03 倍，直すべき折り込み 9 表は 0.3-1.0 倍．

    Returns:
        (使う画像の場所, 直した角度)．直さなければ (元の場所, 0.0)
    """
    img = Image.open(image)
    deg, dy = estimate(img, box)
    if pitch <= 0 or abs(dy) < pitch * min_rows or abs(deg) > max_deg:
        return image, 0.0
    # `estimate` は右が上がっていると負．PIL の rotate は反時計回りが正なので，
    # 符号を反転して回す(08_p2 で確かめた)
    fixed = img.convert('RGB').rotate(-deg, resample=Image.BICUBIC, expand=False,
                                      fillcolor=(255, 255, 255))
    fixed.save(out)
    return str(out), deg
