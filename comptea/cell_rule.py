"""組成のセルの箱から，縦罫線を外す (段階 2 の切り出しだけ)

2026-09-17 に足した．基準の通し (147 表) で `1` を含む読みの 13,031 セルのうち
**931 セルの箱の中に縦罫線があった** (60 表)．罫線と `・` の組を `1` と読む．
左端の列が 505・内側の列が 426．

左端の列の押し出し (`left_rule`) は，罫線に**直線 + 一律の余白**を当てる．
ところが罫線から値までの隙間は表ごと・行ごとに違い，二重罫線のすぐ右から
値が始まる表 (23_p3・06_p2) では余白 6〜8 px が値の頭を切った．
押し出しの歯止め (境にインクが乗るセルが減るときだけ採る) がそれを拒むので，
10_p2・02_p2・23_p3 などでは押し出しが発動せず，罫線が箱に残っていた．

そこで**セルごとに**罫線を見つけ，罫線と値のあいだの白い隙間に境を置く．

- **罫線と字の `1` は，箱の上下の外まで続くかで見分ける**．罫線は行をまたいで
  続き，字は行の中で切れる (`RULE_EXT`)．
- 探すのは箱の左右の端に近い範囲だけ (`SIDE`)．値の中の縦線は触らない．
- 境は**罫線の右 `GAP` px** に置くが，**右の字の手前 1 px を越えない**
  (値の頭を切らない)．読み直しは箱の周りに 3 px 足すので，隙間があれば
  その分を空ける．
- 幅を元の `KEEP_W` より細くしない．

格子 (`located.csv`) は変えない (x1 で列を束ねる段があるため．`left_rule` と同じ)．
"""
import numpy as np

RULE_EXT = 0.5      # 箱の上下に，箱の高さのこの割合だけ伸ばして罫線が続くかを見る
RULE_FILL = 0.9     # 伸ばした縦の範囲でこの割合以上黒い x を罫線とみなす
EDGE_FILL = 0.3     # 罫線に続く列でこの割合以上黒ければ，罫線の縁 (にじみ) とみなす
SIDE = 0.35         # 箱の左右，幅のこの割合の中だけ探す
OUTSIDE = 2         # 箱の外へこの px まで探す (境のすぐ外に乗る罫線)
GAP = 5             # 罫線の端から境までの距離の目安 (読み直しの余白 3 + 2)
TICK = 3            # 罫線の縁のこの px までは字を探さない (縁のぎざぎざを字と取り違える)
KEEP_W = 0.5        # 幅を元のこの割合より細くしない


def rule_columns(dark, y1, y2, xa, xb, ext=RULE_EXT, fill=RULE_FILL):
    """[xa, xb) のうち，[y1, y2) を上下に伸ばした範囲で罫線とみなせる x の配列"""
    h, w = dark.shape
    e = int(round((y2 - y1) * ext))
    ya, yb = max(0, int(y1) - e), min(h, int(y2) + e)
    xa, xb = max(0, int(xa)), min(w, int(xb))
    if yb <= ya or xb <= xa:
        return np.array([], dtype=int)
    frac = dark[ya:yb, xa:xb].mean(axis=0)
    return xa + np.flatnonzero(frac >= fill)


def _has_ink(dark, y1, y2, x):
    return bool(dark[int(y1):int(y2), int(x)].any())


def _rule_body(dark, y1, y2, x, step, ext=RULE_EXT, fill=EDGE_FILL, limit=6):
    """罫線の芯 x から `step` の向きに，にじみ (上下に伸ばした範囲で `fill` 以上黒い列)
    が続く最後の x．罫線の縁は芯ほど黒くないので，字と取り違えないために飛ばす"""
    h, w = dark.shape
    e = int(round((y2 - y1) * ext))
    ya, yb = max(0, int(y1) - e), min(h, int(y2) + e)
    end = x
    for k in range(1, limit + 1):
        xx = x + step * k
        if not 0 <= xx < w or dark[ya:yb, xx].mean() < fill:
            break
        end = xx
    return end


def trim_box(dark, x1, y1, x2, y2, side=SIDE, gap=GAP, keep_w=KEEP_W):
    """箱 (x1, y1, x2, y2) から左右の縦罫線を外した (新しい x1, 新しい x2)"""
    w = x2 - x1
    if w <= 0 or y2 <= y1:
        return x1, x2
    nx1, nx2 = float(x1), float(x2)
    h, W = dark.shape
    y1i, y2i = max(0, int(y1)), min(h, int(y2))

    left = rule_columns(dark, y1, y2, x1 - OUTSIDE, x1 + w * side)
    if len(left):
        r = _rule_body(dark, y1, y2, int(left.max()), +1)
        stop = int(min(W - 1, x2))
        first_ink = next((x for x in range(r + TICK + 1, stop)
                          if _has_ink(dark, y1i, y2i, x)), stop)
        nx1 = max(nx1, float(min(r + gap, max(r + TICK, first_ink - 1))))

    right = rule_columns(dark, y1, y2, x2 - w * side, x2 + OUTSIDE)
    if len(right):
        l_ = _rule_body(dark, y1, y2, int(right.min()), -1)
        stop = int(max(0, x1))
        last_ink = next((x for x in range(l_ - TICK - 1, stop, -1)
                         if _has_ink(dark, y1i, y2i, x)), stop)
        nx2 = min(nx2, float(max(l_ - gap, min(l_ - TICK, last_ink + 1))))

    if nx2 - nx1 < w * keep_w:
        return x1, x2
    return nx1, nx2


def trim_rules(df, dark):
    """組成のセルの箱から縦罫線を外した写しと，動かしたセルの数"""
    out = df.copy()
    if 'obj_name' not in out.columns:
        return out, 0
    idx = out.index[out['obj_name'] == 'comp']
    n = 0
    for i in idx:
        x1, y1, x2, y2 = (float(out.at[i, k]) for k in ('x1', 'y1', 'x2', 'y2'))
        a, b = trim_box(dark, x1, y1, x2, y2)
        if a != x1 or b != x2:
            out.at[i, 'x1'] = a
            out.at[i, 'x2'] = b
            n += 1
    return out, n
