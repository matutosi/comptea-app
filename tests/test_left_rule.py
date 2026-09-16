"""組成部の左端の列だけを，左の縦罫線より右へ押し出す (left_rule)

2026-09-16 に足した．19_p2 は縦罫線が下へ行くほど右へずれ，組成の左端が
罫線の左に入り込んで，階層の記号 `K` と罫線が地点 1 のセルに入っていた．
全列を一律に傾ける案は取り下げた (紙面が剛体のように回っていない) ので，
**罫線に接する左端の列の左の境だけ**を，**右へ押し出す向きだけ**動かす．
"""
import numpy as np
import pandas as pd

from comptea import left_rule

H, W = 1200, 600
PITCH = 30


def _dark(slope=0.02, rule_x0=100, gaps=()):
    d = np.zeros((H, W), dtype=bool)
    for y in range(H):
        if any(a <= y < b for a, b in gaps):
            continue
        x = int(round(rule_x0 + slope * y))
        if 0 <= x < W - 3:
            d[y, x:x + 3] = True
    return d


def _grid(x_left=120, n_cols=4, col_w=80):
    rows = []
    for c in range(n_cols):
        for k, y in enumerate(range(0, H - PITCH + 1, PITCH)):
            rows.append({'obj_name': 'comp', 'col': c + 1, 'row': k + 1,
                         'x1': float(x_left + c * col_w),
                         'x2': float(x_left + (c + 1) * col_w),
                         'y1': float(y), 'y2': float(y + PITCH)})
    return pd.DataFrame(rows)


def test_罫線に直線を当てる():
    a, b = left_rule.fit_rule(_dark(slope=0.02), _grid())
    assert abs(a - 0.02) < 0.003
    assert abs(b - 100) < 5


def test_罫線が越えている行だけ右へ押し出す():
    """上の方は罫線が左の境より左にあるので動かさない"""
    df = _grid()
    out, moved = left_rule.push_first_column(df, (0.02, 100.0), margin=10)
    first = out[(out['col'] == 1)]
    top = first[first['row'] == 1]['x1'].iloc[0]
    bot = first[first['row'] == first['row'].max()]['x1'].iloc[0]
    assert top == 120.0                            # 上は罫線+余白 (110) より右
    assert bot > 120.0 + 10                        # 下の方は押し出す (100+24+10)
    assert 0 < moved < len(first)


def test_左へは動かさない():
    """右上がりの罫線は左端の列から離れていくので，記号を引き込まない"""
    df = _grid()
    out, moved = left_rule.push_first_column(df, (-0.02, 100.0), margin=10)
    assert moved == 0
    assert out['x1'].equals(df['x1'])


def test_他の列には触らない():
    df = _grid()
    out, _m = left_rule.push_first_column(df, (0.02, 100.0))
    assert out[out['col'] != 1][['x1', 'x2']].equals(df[df['col'] != 1][['x1', 'x2']])


def test_左端の列の右の境には触らない():
    df = _grid()
    out, _m = left_rule.push_first_column(df, (0.02, 100.0))
    assert out[out['col'] == 1]['x2'].equals(df[df['col'] == 1]['x2'])


def test_幅を半分より細くしない():
    """罫線が大きく入り込んでも，値の入る幅を残す"""
    df = _grid()
    out, _m = left_rule.push_first_column(df, (0.2, 100.0), margin=10)
    first = out[out['col'] == 1]
    assert ((first['x2'] - first['x1']) >= 80 * 0.5 - 1e-6).all()


def test_罫線が無ければ当てない():
    assert left_rule.fit_rule(np.zeros((H, W), bool), _grid()) is None


def test_罫線が途切れがちなら当てない():
    gaps = [(y, y + 92) for y in range(0, H, 100)]
    assert left_rule.fit_rule(_dark(slope=0.02, gaps=gaps), _grid()) is None


def test_元の格子は変えない():
    df = _grid()
    before = df.copy()
    left_rule.push_first_column(df, (0.02, 100.0))
    assert df.equals(before)


def test_境のインクが減らなければ採らない():
    """列の内側をまっすぐ走る線を罫線と誤認すると，押し出しても良くならない

    全 147 表で測ると 56 表で発動し，減ったのは 25 表だけだった．
    kinki_004-1 は全セルを 53 px 押し出し，06_p2 は 5 → 14 と悪化していた．
    """
    d = np.zeros((H, W), dtype=bool)
    d[:, 150:153] = True                       # 列の内側に，まっすぐな線
    df = _grid()                               # 左端 120
    out, moved = left_rule.push_first_column(df, (0.0, 150.0), margin=10, dark=d)
    assert moved == 0
    assert out['x1'].equals(df['x1'])


def test_境のインクが減るなら採る():
    """19_p2: 罫線が下へ行くほど列に入り込み，左の境に乗っていた"""
    d = _dark(slope=0.02, rule_x0=110)         # 下で左端 120 を越える
    df = _grid()
    before = left_rule.edge_ink(d, df)
    out, moved = left_rule.push_first_column(df, (0.02, 110.0), margin=10, dark=d)
    assert moved > 0
    assert left_rule.edge_ink(d, out) < before
