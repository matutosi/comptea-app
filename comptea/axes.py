"""境を組み立て，字を避けてずらす

検出した箱の並びから行と列の境を作り(`locate_edges`)，落ちた所だけ内挿する．
境が字に重なっていれば，黒画素の谷へ寄せる(`snap_edges`)．表頭は本体と
数十 px ずれて組まれていることがあるので，**数を変えずに**ずらす
(`_shift_edges_for_header`．地の文に引っぱられないよう，地点の値が並ぶ行
だけで測る)．
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import ink
from .filters import filter_results


def locate_x_range(df: pd.DataFrame,
    source_image: str = "",
    obj_name: str = "col"):
    """
    obj_nameのx1とx2の範囲

    Returns:
        検出があるとき: x1, x2を1行持つDataFrame
        検出が無いとき: None(0,0の範囲を作らない．幅ゼロの領域をでっちあげないため)
    """
    results = filter_results(df, source_image, obj_name)
    if results.empty:
        return None
    return pd.DataFrame({'x1': [results['x1'].min()], 'x2': [results['x2'].max()]})


def locate_y_range(df: pd.DataFrame,
    source_image: str = "",
    obj_name: str = "row"):
    """
    obj_nameのy1とy2の範囲

    Returns:
        検出があるとき: y1, y2を1行持つDataFrame
        検出が無いとき: None(0,0の範囲を作らない)
    """
    results = filter_results(df, source_image, obj_name)
    if results.empty:
        return None
    return pd.DataFrame({'y1': [results['y1'].min()], 'y2': [results['y2'].max()]})


def _merge_close(centers: np.ndarray, min_gap: float) -> np.ndarray:
    """
    近すぎる中心をまとめる

    同じ行(列)を2回検出したものを1つにする．
    重なりが小さい重複はremove_dup_range()では残るため，ここで拾う．
    """
    merged = []
    group = [centers[0]]
    for c in centers[1:]:
        if c - group[-1] < min_gap:
            group.append(c)
        else:
            merged.append(np.mean(group))
            group = [c]
    merged.append(np.mean(group))
    return np.array(merged)


def _drop_isolated(starts: np.ndarray, ends: np.ndarray, pitch: float,
                   max_gap_pitches: float = 5):
    """
    他から大きく離れた検出を捨てる

    ページの隅に1本でも誤検出があると，そこまでの隙間を内挿で埋めてしまう．
    いちばん大きな塊(隣との隙間が max_gap_pitches 以内で繋がる並び)だけを残す．

    **離れ具合は箱と箱の隙間(前の下端 → 次の上端)で測る**．箱の中心の間隔で
    測ってはいけない(2026-09-03)．組成部の谷から作った行の箱は，字の詰まった
    数行(谷が取れない区間)をまたいで 150-500 px の高さになることがある．
    中心で測るとその箱の前後が「5行分より離れている」になり，塊が切れて
    最大の塊以外が丸ごと捨てられていた(s01115_10_p1 は 335 行のうち 113 行しか
    残らず，残り 6653 px を1つの巨大な行が飲み込んでいた)．
    隙間で測れば，連続した箱は隙間 0 なので切れず，本来の用途(kinki_007 の
    2584 px 離れた1本)はそのまま捨てられる．

    Returns:
        (残す箱の bool マスク, 捨てた本数)
    """
    n = len(starts)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep, 0
    limit = pitch * max_gap_pitches
    reach = np.maximum.accumulate(ends)     # ここまでの箱の下端の最大
    groups, current = [], [0]
    for i in range(1, n):
        if starts[i] - reach[i - 1] > limit:
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)
    if len(groups) == 1:
        return keep, 0
    best = max(groups, key=len)
    keep[:] = False
    keep[best] = True
    return keep, n - len(best)


def locate_edges(df: pd.DataFrame, source_image: str = "",
                 obj_name: str = "row", axis: str = "y"):
    """
    検出された1本ずつの位置から，分割の境界を組み立てる

    「表全体を等分する」のをやめ，検出そのものを境界に使う．
    隣り合う検出の間隔を代表値(中央値)で割った値を丸めて，
    その区間に何行(何列)あるかを数える．
      1 -> 隣接．そのまま
      2以上 -> 検出漏れ．その区間だけ内挿する
      0 -> 同じ行の重複検出．まとめる(_merge_close)
    行数を「表の高さ ÷ 平均行高」で決めるより，
    検出の増減に強く，ドリフト(途中からのずれ)にも追従する．

    Args:
        df: 検出結果．重複除去済みであること
        source_image: 対象の画像
        obj_name: 'row' または 'col'
        axis: 'y'(行) または 'x'(列)
    間隔の代表値は2通りで見積もる．
      - 検出どうしの間隔の中央値: 検出漏れが多いと「2行分」を1行と見誤る
      - 検出の箱の大きさの中央値: 漏れの影響を受けないが，
        ラベルが行より小さく付けられていると過小になる
    小さい方を採る(漏れによる過大を避ける)．
    2つが食い違うときは，どちらが正しいか決められないので警告する．

    Returns:
        (edges, interpolated, warnings)
        edges       : 長さ n+1 の境界．n が行数(列数)
        interpolated: 長さ n の bool．内挿で作ったセルが True
        warnings    : 気づいたことの文字列のリスト
        検出が2本未満で間隔を出せないときは (None, None, warnings)
    """
    a, b = ('y1', 'y2') if axis == 'y' else ('x1', 'x2')
    warnings = []
    res = filter_results(df, source_image, obj_name)
    if res.empty:
        return None, None, warnings
    if len(res) == 1:
        # 1地点だけの組成表はcolが1本しかない．間隔は要らないのでそのまま使う
        warnings.append(
            f"'{obj_name}' の検出が1本しかないため，分割せず1つとして扱う．"
            '本当に1つかどうかは目で確かめる．')
        edges = np.array([res[a].iloc[0], res[b].iloc[0]], dtype=float)
        return edges, np.array([False]), warnings
    res = res.sort_values(by=a)
    starts = res[a].to_numpy(dtype=float)
    ends = res[b].to_numpy(dtype=float)
    centers = (starts + ends) / 2

    pitch_size = np.median(ends - starts)
    pitch_diff = np.median(np.diff(centers))
    if not np.isfinite(pitch_diff) or pitch_diff <= 0:
        return None, None, warnings
    if not np.isfinite(pitch_size) or pitch_size <= 0:
        pitch_size = pitch_diff
    pitch = min(pitch_size, pitch_diff)

    # 遠く離れた検出を捨てる．
    # ページの隅の誤検出が1本混じるだけで，そこまでの隙間を内挿で埋めてしまう
    # (kinki_007 で本来の行から 2584px = 46行分 離れた1本があり，
    #  6行のはずが52行になった)．
    # 箱の隙間で測るので，中心をまとめる前に箱のまま行う
    keep, dropped = _drop_isolated(starts, ends, pitch_size, max_gap_pitches=5)
    if dropped:
        warnings.append(
            f"'{obj_name}' の検出 {dropped} 本が他から大きく離れていたため捨てた．"
            '誤検出とみなしたが，表が離れて2つある資料なら見直す．')
        centers = centers[keep]
        # **両端の境にも使わない**(2026-09-04)．捨てたのは中心の並びだけで，
        # 最後の境は `ends.max()` から作っていたので，流し込みの中の 1 本の
        # 下端が最後の行の下端になり，最後の行が 1,487 px になっていた
        # (s01115_02_p1_s1．14_p1 は 923 px)
        starts, ends = starts[keep], ends[keep]
    if len(centers) < 2:
        return None, None, warnings

    # 代表値で重複をまとめ，間隔を取り直す．
    # **列は箱の大きさでまとめる**(2026-09-03)．`pitch`(2つの見積もりの
    # 小さい方)で決めると，同じ地点を何本も検出しているときに
    # 中心の間隔の中央値がその重なりの幅まで潰れ(実測 88 px の列に対し 6 px)，
    # `pitch/2` では重複がまとまらない．まとまらないまま間隔を取るので
    # 「1つの列に何本もある」を「列がたくさんある」と数えてしまう
    # (s01115_19_p2_t1 は 17 地点の表が 42 列，05_p2 は 17 地点が 53 列)．
    # **行には同じことをしない**．行の箱は隣どうしが重なって並ぶのが普通で，
    # 幅を広げると `_merge_close` が数珠つなぎに繋いで行を失う
    # (全 66 表で測ると 19 表・569 行が減った)．列は地点ごとに離れて並ぶ
    merge_gap = (pitch_size if axis == 'x' else pitch) / 2
    centers = _merge_close(centers, min_gap=merge_gap)
    if len(centers) < 2:
        return None, None, warnings

    diffs = np.diff(centers)
    pitch_diff = np.median(diffs)
    pitch = min(pitch_size, pitch_diff)

    # 区間ごとに何行あるかを数え，足りない分を内挿する
    counts = np.maximum(1, np.round(diffs / pitch)).astype(int)
    # 2通りの見積もりで行数が変わるなら，どちらが正しいか決められない．
    # 見積もりの値そのものは検出漏れが多いと当然ずれるので，
    # 「結果が変わるか」で判断する
    other = pitch_diff if pitch == pitch_size else pitch_size
    n_other = int(np.maximum(1, np.round(diffs / other)).sum()) + 1
    n_used = int(counts.sum()) + 1
    if n_other != n_used:
        warnings.append(
            f"'{obj_name}' の数を決められない．"
            f'検出の間隔から見ると {n_used if pitch == pitch_diff else n_other}，'
            f'検出の箱の大きさから見ると {n_used if pitch == pitch_size else n_other}．'
            f'間隔の狭い方をとって {n_used} とした．検出結果を目で確かめる．')
    full = [centers[0]]
    interpolated = [False]
    for i, k in enumerate(counts):
        step = (centers[i + 1] - centers[i]) / k
        for j in range(1, k + 1):
            full.append(centers[i] + step * j)
            interpolated.append(j != k)  # 区間の最後だけが実測
    full = np.array(full)

    # 境界は中心どうしの中点．両端だけは検出の外側をそのまま使う
    mid = (full[:-1] + full[1:]) / 2
    edges = np.concatenate([[starts.min()], mid, [ends.max()]])
    return edges, np.array(interpolated), warnings


def ink_profile(image, box, axis: str = "y"):
    """
    領域を二値化して，軸に沿ったインクの量を数える

    行の境界(axis='y')なら，y の位置ごとに横一列のインク画素数を数える．
    文字のある高さは山，行と行の隙間は谷になる．

    Args:
        image: PILのImage
        box: (x1, y1, x2, y2) の集計範囲
        axis: 'y'(行の境界を見る) または 'x'(列の境界を見る)
    Returns:
        (profile, origin, cross_len)
        profile  : 位置ごとのインク画素数
        origin   : profile[0] が画像のどの座標にあたるか
        cross_len: 足し合わせた方向の長さ(谷の判定に使う)
        領域が取れないときは (None, 0, 0)
    """
    w, h = image.size
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, 0, 0
    gray = np.asarray(image.crop((x1, y1, x2, y2)).convert('L'), dtype=np.uint8)
    # 大津の方法で二値化(古い資料は明るさが一定でないため固定閾値にしない)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if axis == 'y':
        return bw.sum(axis=1) / 255.0, y1, x2 - x1
    return bw.sum(axis=0) / 255.0, x1, y2 - y1


def _peak_run(profile: np.ndarray, pos: int, tol: float):
    """
    posを含む「インクがtolを超えて続く区間」を (始まり, 終わり) で返す
    """
    start = pos
    while start > 0 and profile[start - 1] > tol:
        start -= 1
    end = pos
    while end < len(profile) - 1 and profile[end + 1] > tol:
        end += 1
    return start, end


def snap_edges(edges: np.ndarray, profile: np.ndarray, origin: int,
               cross_len: int, max_shift: int, ink_ratio: float = 0.02,
               max_rule_width: int = 6):
    """
    文字に重なっている境界を，近くの谷へずらす

    等分でも検出どおりでも，境界が文字の上に乗ると，
    1文字が2つのセルに割れてOCRが失敗する．
    インクの少ない位置へ寄せることでこれを避ける．
    両端は表の外枠なので動かさない．

    **罫線は例外**．組成表は実線や点線で一部を囲むことがあり，
    罫線は表を横切るのでプロファイルでは山になる．
    しかし罫線のある位置は「文字」ではなく，むしろ正しい境界そのものなので，
    谷へ逃がしてはいけない．山の幅が細いもの(max_rule_width以下)を罫線とみなし，
    その中央へ合わせる．文字の帯は太い山になるので区別できる．

    Args:
        edges: 境界の並び(長さ n+1)
        profile, origin, cross_len: ink_profile()の戻り値
        max_shift: ずらせる最大の画素数
        ink_ratio: これ以下のインク量を「谷」とみなす割合(cross_lenに対する)
        max_rule_width: この幅以下の山は罫線とみなす(画素)
    Returns:
        (new_edges, snapped, unresolved, on_rule)
        snapped   : 境界ごとのbool．動かしたものがTrue
        unresolved: 境界ごとのbool．ずらしても谷に入らなかったものがTrue
        on_rule   : 境界ごとのbool．罫線に合わせたものがTrue
    """
    new_edges = np.array(edges, dtype=float)
    snapped = np.zeros(len(edges), dtype=bool)
    unresolved = np.zeros(len(edges), dtype=bool)
    on_rule = np.zeros(len(edges), dtype=bool)
    tol = max(1.0, ink_ratio * cross_len)
    # 谷はプロファイル全体から取る．窓の中だけで見ると，
    # 窓に切られた谷の「端」を中央と勘違いする
    runs = _valley_runs(profile, tol)
    run_centers = np.array([(s + e) / 2 for s, e in runs]) if runs else np.array([])
    for i in range(1, len(edges) - 1):
        pos = int(round(edges[i])) - origin
        if pos < 0 or pos >= len(profile):
            continue
        if profile[pos] <= tol:
            continue  # すでに谷にある
        run_start, run_end = _peak_run(profile, pos, tol)
        if run_end - run_start + 1 <= max_rule_width:
            # 罫線に乗っている．逃がさず，罫線の中央に合わせる
            pick = (run_start + run_end) / 2
            on_rule[i] = True
            if pick != pos:
                new_edges[i] = pick + origin
                snapped[i] = True
            continue
        lo = max(0, pos - max_shift)
        hi = min(len(profile), pos + max_shift + 1)
        window = profile[lo:hi]
        if not len(window):
            continue
        # 窓に掛かる谷を探す
        reachable = [j for j, (s, e) in enumerate(runs) if s <= hi - 1 and e >= lo]
        if reachable:
            # 谷の端では文字に近すぎるので，いちばん近い谷の中央へ寄せる
            centers = run_centers[reachable]
            target = centers[np.argmin(np.abs(centers - pos))]
            pick = float(np.clip(target, lo, hi - 1))  # ずらせる幅は超えない
        else:
            # 谷が無い．いちばんインクの少ない位置まで寄せて，直せなかったと伝える
            pick = float(np.argmin(window) + lo)
            unresolved[i] = True
        if pick != pos:
            new_edges[i] = pick + origin
            snapped[i] = True
    return new_edges, snapped, unresolved, on_rule


def _valley_runs(window: np.ndarray, tol: float):
    """
    インクがtol以下で続く区間(谷)を (始まり, 終わり) の並びで返す
    """
    is_valley = window <= tol
    if not is_valley.any():
        return []
    edges = np.diff(np.concatenate([[0], is_valley.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts, ends))


def axis_notes(interpolated=None, snapped=None, unresolved=None, n: int = 0):
    """
    1つの軸について，セルごとの気になる点を並べる

    境界のフラグ(長さ n+1)は，その境界に接する両側のセルに配る．

    Args:
        interpolated: セルごとのbool(長さ n)．検出漏れを内挿したもの
        snapped: 境界ごとのbool(長さ n+1)．文字を避けてずらしたもの
        unresolved: 境界ごとのbool(長さ n+1)．ずらしても文字に重なるもの
        n: セルの数
    Returns:
        長さ n のリスト．各要素は文字列のリスト
    """
    notes = [[] for _ in range(n)]
    for i in range(n):
        if interpolated is not None and interpolated[i]:
            notes[i].append('interpolated')
        if snapped is not None and (snapped[i] or snapped[i + 1]):
            notes[i].append('snapped')
        if unresolved is not None and (unresolved[i] or unresolved[i + 1]):
            notes[i].append('on_text')
    return notes


def coord_item(x_edges: np.ndarray, y_edges: np.ndarray, obj_name: str = "",
               x_notes=None, y_notes=None) -> pd.DataFrame:
    """
    境界の並びからセルの座標を作る

    分割しない軸は，端から端までの2要素を渡す．

    Args:
        x_edges: 長さ n_cols+1 の境界
        y_edges: 長さ n_rows+1 の境界
        obj_name: 付けるクラス名
        x_notes, y_notes: axis_notes()の戻り値．セルのnote列に合わせて入れる
    Returns:
        x1, x2, y1, y2, obj_name, note を持つDataFrame(列ごとに行が並ぶ)
        note は ';' 区切り．気になる点が無ければ空文字
    """
    x1, x2 = x_edges[:-1], x_edges[1:]
    y1, y2 = y_edges[:-1], y_edges[1:]
    n_cols, n_rows = len(x1), len(y1)
    if x_notes is None:
        x_notes = [[] for _ in range(n_cols)]
    if y_notes is None:
        y_notes = [[] for _ in range(n_rows)]
    # np.repeat / np.tile と同じ並び(列ごとに行が並ぶ)にする
    notes = [';'.join(dict.fromkeys(x_notes[c] + y_notes[r]))
             for c in range(n_cols) for r in range(n_rows)]
    return pd.DataFrame({
        'x1': np.repeat(x1, n_rows),
        'x2': np.repeat(x2, n_rows),
        'y1': np.tile(y1, n_cols),
        'y2': np.tile(y2, n_cols),
        'obj_name': obj_name,
        'note': notes,
    })


def _edge_ink(edges, dark, y1, y2):
    """境の線上の黒画素(帯の中央値との比)．小さいほど字を割っていない"""
    x1, x2 = int(edges[0]), int(edges[-1])
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    prof = dark[int(y1):int(y2), x1:x2].sum(axis=0).astype(float)
    if not (prof > 0).any():
        return None
    med = float(np.median(prof[prof > 0]))
    # **境が画像の外にあることがある**．`x2` が画像の幅を超えると `dark[..., x1:x2]` は
    # そこで切られるので，`x2 - x1` でなく**並びの長さ**で判定しないと落ちる
    # (s01115_23_p1 で `IndexError: index 2315 is out of bounds`．2026-09-10)
    n = len(prof)
    vals = [prof[i] for i in (int(e) - x1 for e in edges[1:-1]) if 0 <= i < n]
    return (float(np.mean(vals)) / med) if vals and med else None


HEADER_SHIFT_MAX = 0.4      # 表頭をずらす範囲(列の間隔に対する比)


HEADER_SHIFT_GAIN = 0.5     # 線上の黒画素がこの倍以下になるときだけずらす


HEADER_SHIFT_MIN = 5        # これ未満のずれは動かさない(誤差の範囲)


HEADER_SHIFT_TOL = 5        # 行ごとのずれがこの差以内なら，同じずれとみなす


def _shift_edges_for_header(x_edges, img, bands, max_ratio=HEADER_SHIFT_MAX,
                            gain=HEADER_SHIFT_GAIN, min_shift=HEADER_SHIFT_MIN,
                            tol=HEADER_SHIFT_TOL):
    """表頭の列の境を，**地点の値の行の合意**で左右にずらす

    帯ぜんたいの黒画素で決めてはいけない(2026-09-05 に測って分かった)．
    表頭には群落名や凡例の**地の文**が混ざり，そちらの字数が多いので
    ずらし量がそれに引っぱられる(s01115_22_p2 は帯ぜんたいで +41 px だが，
    通し番号の行で測ると 0 px = ずらす必要なし．読んだ通し番号の一致は
    0.56 → 0.04 と落ちていた)．

    **地点ごとの値が並ぶ行**(塊が列の数の半分以上)だけを対象にし，
    行ごとの最良のずれを求めて，`tol` px 以内で最も多くの行が賛成する値を採る．

    Args:
        bands: 表頭の行の (y1, y2) のリスト
    Returns:
        (ずらした境, ずらした量 px．ずらさなければ (元の境, 0))
    """
    from . import count_plots

    if img is None or x_edges is None or len(x_edges) < 3 or not bands:
        return x_edges, 0
    dark = ink.binarize(img)
    n_col = len(x_edges) - 1
    pitch = float(np.median(np.diff(x_edges)))
    votes = []
    for y1, y2 in bands:
        if y2 - y1 < 5:
            continue
        bl = count_plots.blobs(dark, x_edges[0], x_edges[-1], y1, y2,
                               max(6, int(pitch * 0.25)))
        if len(bl) < n_col * 0.5:
            continue                      # 地の文の行(値が並んでいない)
        _, s = _shift_edges_to_band(x_edges, img, y1, y2, max_ratio,
                                    gain=1.0, min_shift=1)
        votes.append(s)
    if not votes:
        return x_edges, 0
    # いちばん多くの行が賛成するずれ(±tol px を同じ票とみなす)
    best, best_n = 0, 0
    for v in votes:
        n = sum(1 for w in votes if abs(w - v) <= tol)
        if n > best_n or (n == best_n and abs(v) < abs(best)):
            best, best_n = v, n
    agree = [w for w in votes if abs(w - best) <= tol]
    shift = int(round(float(np.median(agree))))
    if abs(shift) < min_shift or best_n < max(2, len(votes) * 0.5):
        return x_edges, 0
    e = np.asarray(x_edges, dtype=float).copy()
    e[1:-1] += shift
    return np.maximum.accumulate(e), shift


def _shift_edges_to_band(x_edges, img, y1, y2, max_ratio=HEADER_SHIFT_MAX,
                         gain=HEADER_SHIFT_GAIN, min_shift=HEADER_SHIFT_MIN):
    """列の境を，`y1`-`y2` の帯の黒画素に合わせて左右にずらす

    列の数は変えない．ずらして線上の黒画素が `gain` 倍以下になるときだけ採る．

    Returns:
        (ずらした境, ずらした量 px．ずらさなければ (元の境, 0))
    """
    if img is None or x_edges is None or len(x_edges) < 3 or y2 - y1 < 5:
        return x_edges, 0
    dark = ink.binarize(img)
    base = _edge_ink(x_edges, dark, y1, y2)
    if base is None:
        return x_edges, 0
    pitch = float(np.median(np.diff(x_edges)))
    reach = max(1, int(pitch * max_ratio))

    def shifted(s):
        # **外側の境は動かさない**．端ごとずらすと，端の値が枠の外に出る
        # (22_p2 は +41 px で通し番号の「1」が左端の外に落ちた．2026-09-04)
        e = np.asarray(x_edges, dtype=float).copy()
        e[1:-1] += s
        return np.maximum.accumulate(e)

    best, best_v = 0, base
    for s in range(-reach, reach + 1):
        v = _edge_ink(shifted(s), dark, y1, y2)
        if v is not None and v < best_v:
            best, best_v = s, v
    # **しきい値は 0.5 倍・5 px 以上**(2026-09-05 に 3 案を測って決めた)．
    # 0.8 倍だと 6 表 420 セルのうち動きが 1 表増えるだけで，読みは同じだった．
    # 発動する表が少ないほど，意図しない副作用が減る
    if abs(best) < min_shift or best_v > base * gain:
        return x_edges, 0
    return shifted(best), best


def _snap_axis(edges, img, box, axis: str, label: str, max_shift_ratio: float):
    """
    1つの軸の境界をまとめてずらす(locate_itemsから使う)

    Returns:
        (new_edges, warnings)
    """
    warnings = []
    none_flags = np.zeros(len(edges), dtype=bool)
    profile, origin, cross_len = ink_profile(img, box, axis=axis)
    if profile is None:
        warnings.append(f'{label}の境界を調べる領域が画像の外にあり，調整を行わなかった．')
        return edges, warnings, none_flags, none_flags
    pitch = np.median(np.diff(edges))
    max_shift = max(1, int(round(pitch * max_shift_ratio)))
    # 文字の帯より細い山を罫線とみなす
    max_rule_width = max(3, int(round(pitch * 0.15)))
    new_edges, snapped, unresolved, on_rule = snap_edges(
        edges, profile, origin, cross_len, max_shift,
        max_rule_width=max_rule_width)
    n_snapped = int(snapped.sum())
    n_unresolved = int(unresolved.sum())
    n_on_rule = int(on_rule.sum())
    if n_snapped - n_on_rule > 0:
        warnings.append(
            f'{label}の境界 {n_snapped - n_on_rule} 本が文字に重なっていたため，'
            f'最大 {max_shift} px の範囲で隙間へずらした．')
    if n_on_rule:
        warnings.append(
            f'{label}の境界 {n_on_rule} 本は罫線に乗っていたため，'
            'その中央に合わせた(罫線は正しい区切りなので，動かさない)．')
    if n_unresolved:
        warnings.append(
            f'{label}の境界 {n_unresolved} 本は，ずらしても文字に重なったまま．'
            'そのセルは文字が割れて読めていないおそれがある．'
            f'{label}数が実際と違う可能性を疑う．')
    # ずらした先が隣の隙間だと，セルの大きさが偏る．格子そのものを疑う手がかりになる
    sizes = np.diff(new_edges)
    n_odd = int((np.abs(sizes - pitch) > pitch * 0.25).sum())
    if n_odd:
        warnings.append(
            f'{label}のうち {n_odd} 件は，大きさが代表値({pitch:.0f} px)から'
            f'25%以上ずれている．{label}の区切りが1つずれているおそれがある．')
    # 罫線に合わせたものは問題ではないので，セルの印には含めない
    return new_edges, warnings, snapped & ~on_rule, unresolved
