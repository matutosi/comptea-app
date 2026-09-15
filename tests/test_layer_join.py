"""階層と組成の境を，印字の谷で 1 本にする (layer_col.join_layer_comp_edge)

2026-09-15 に足した．`refit_layer_width` にも同じ規則があるが，そちらは
**階層の枠が記号を切っている段**でしか動かない．記号が枠に収まったまま
組成の 1 列目が階層へ食い込む紙面がある (kinki_076 の段 2 は 28 px 重なり，
要確認 14 件．kinki_045-1 は 105 px で 7 件)．
"""
import pandas as pd
from PIL import Image, ImageDraw

from comptea import layer_col

PITCH = 20
N_ROWS = 20


def _sheet(sym_x=(30, 58), val_x=(66, 102), width=140):
    """階層の記号と値を，谷をはさんで並べた紙面"""
    img = Image.new('L', (width, PITCH * (N_ROWS + 2)), 'white')
    d = ImageDraw.Draw(img)
    for k in range(N_ROWS):
        y = PITCH * (k + 1)
        d.rectangle((sym_x[0], y + 4, sym_x[1], y + 15), fill=0)
        d.rectangle((val_x[0], y + 4, val_x[1], y + 15), fill=0)
    return img.convert('RGB')


def _grid(lay=(20, 60), comp=(28, 130)):
    """階層の枠と組成の 1 列が重なった格子"""
    rows = []
    for k in range(N_ROWS):
        y = PITCH * (k + 1)
        rows.append({'obj_name': 'layer', 'block': 1, 'row': k + 1, 'col': 1,
                     'x1': lay[0], 'x2': lay[1], 'y1': y, 'y2': y + PITCH})
        rows.append({'obj_name': 'comp', 'block': 1, 'row': k + 1, 'col': 2,
                     'x1': comp[0], 'x2': comp[1], 'y1': y, 'y2': y + PITCH})
    return pd.DataFrame(rows)


def test_重なっていれば谷へ動かす():
    """kinki_076: 境が記号の塊の中にあり，セルが `K;+` を拾っていた"""
    out, warns = layer_col.join_layer_comp_edge(_sheet(), _grid())
    got = float(out[out['obj_name'] == 'comp']['x1'].min())
    assert 58 <= got <= 66                      # 記号と値のあいだ
    assert any('谷に合わせた' in w for w in warns)


def test_階層の右端も同じ値にする():
    """階層と組成の境は 1 本 (和名と階層の境と同じ規則)"""
    out, _w = layer_col.join_layer_comp_edge(_sheet(), _grid())
    lay = float(out[out['obj_name'] == 'layer']['x2'].max())
    comp = float(out[out['obj_name'] == 'comp']['x1'].min())
    assert lay == comp


def test_重なっていなければ触らない():
    df = _grid(lay=(20, 60), comp=(62, 130))
    out, warns = layer_col.join_layer_comp_edge(_sheet(), df)
    assert out is df and warns == []


def test_1列目が細くなるなら動かさない():
    """値が入らなくなる幅までは寄せない"""
    df = _grid(lay=(20, 60), comp=(28, 70))
    out, warns = layer_col.join_layer_comp_edge(_sheet(), df)
    assert out is df
    assert any('1 列目' in w for w in warns)


def test_谷が無ければ動かさない():
    """記号と値が同じ x に重なる紙面 (19_p2 型) は，切っても分けられない"""
    img = _sheet(sym_x=(30, 58), val_x=(59, 102))   # 谷が 1 px
    df = _grid()
    out, warns = layer_col.join_layer_comp_edge(img, df)
    assert out is df or float(out[out['obj_name'] == 'comp']['x1'].min()) >= 58


def test_谷がほぼ空でなければ動かさない():
    """05_p2: 谷に見えた所に 237 行中 23 行の値があり，**本物の値 17 件が消えた**

    折込の密な表では，記号と値のあいだにも字がある．「いちばん空いた x」でも
    行数の 3% (または 1 行) を超えて割るなら，そこは谷ではない．
    """
    img = Image.new('L', (140, PITCH * (N_ROWS + 2)), 'white')
    d = ImageDraw.Draw(img)
    for k in range(N_ROWS):
        y = PITCH * (k + 1)
        d.rectangle((30, y + 4, 58, y + 15), fill=0)        # 記号
        d.rectangle((66, y + 4, 102, y + 15), fill=0)       # 値
        if k % 7 == 0:                      # 谷にも字がある (3/20 行．票では
                                            # 谷に見えるが，割れば 3 行を切る)
            d.rectangle((59, y + 4, 65, y + 15), fill=0)
    df = _grid()
    out, warns = layer_col.join_layer_comp_edge(img.convert('RGB'), df)
    assert float(out[out['obj_name'] == 'comp']['x1'].min()) == 28   # 動かさない
    assert any('割る' in w for w in warns)


def test_1列目に縦罫線が残るなら動かさない():
    """kinki_045-1: 罫線はセルの中で `1` と読まれ，`5`・`4`・`3` が全部 `1` になった

    **要確認は 7 → 0 になるので，数だけでは見抜けない**．中身を突き合わせて分かった．
    """
    img = Image.new('L', (140, PITCH * (N_ROWS + 2)), 'white')
    d = ImageDraw.Draw(img)
    for k in range(N_ROWS):
        y = PITCH * (k + 1)
        d.rectangle((30, y + 4, 58, y + 15), fill=0)        # 記号
        d.rectangle((90, y + 4, 120, y + 15), fill=0)       # 値 (右寄り)
    d.rectangle((70, PITCH, 72, PITCH * (N_ROWS + 1)), fill=0)   # 縦罫線
    df = _grid(lay=(20, 60), comp=(28, 130))
    out, warns = layer_col.join_layer_comp_edge(img.convert('RGB'), df)
    assert float(out[out['obj_name'] == 'comp']['x1'].min()) == 28
    assert any('縦罫線' in w for w in warns)


def test_階層が無ければ何もしない():
    df = _grid()
    df = df[df['obj_name'] == 'comp']
    out, warns = layer_col.join_layer_comp_edge(_sheet(), df)
    assert out is df and warns == []


def test_段ごとに見る():
    """1 調査区を左右 2 段に組んだ紙面 (kinki_076) は，段 2 だけ重なっていた"""
    a = _grid(lay=(20, 60), comp=(62, 130))         # 段1: 重なりなし
    b = _grid(lay=(20, 60), comp=(28, 130))         # 段2: 重なり
    b['block'] = 2
    out, warns = layer_col.join_layer_comp_edge(_sheet(), pd.concat([a, b]))
    left = out[(out['block'] == 2) & (out['obj_name'] == 'comp')]['x1'].min()
    kept = out[(out['block'] == 1) & (out['obj_name'] == 'comp')]['x1'].min()
    assert float(kept) == 62                        # 段1 は動かさない
    assert 58 <= float(left) <= 66
