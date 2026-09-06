"""格子の切り出し精度を，正解ラベルと突き合わせて数える

    py -3.12 eval_grid.py [--split val] [--imgsz 1280] [--conf 30]

やっていること
    1. 正解(labelme から作った YOLO のラベル)から，row / col の帯を取り出す
    2. 同じ画像に検出と位置決めをかけ，comp のセルから帯を作り直す
    3. 帯どうしを1次元の IoU で突き合わせ，一致した数を数える

「帯」で見るのは，行の切り出しが縦位置だけで決まり，横位置は関係ないため
(col はその逆)．箱の IoU で見ると，表の幅の推定誤差が行の成績に混ざる．

出力は画像ごとの recall / precision と，取りこぼした帯の位置．
**判断はしない**．改善案を試したときに，前後で比べるための物差しとして使う．
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# yolo/ から実行する前提だが，どこから呼ばれても通るようにしておく
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASET = 'labelme_data/YOLODataset'
# 帯として評価するクラス．row は縦(y)，col は横(x)で切り出しが決まる
BANDS = {'row': 'y', 'col': 'x'}
# 段の横幅を測るのに使うクラス(ページ全幅にかかる 'header' 'table' は入れない)
BLOCK_WIDTH_CLASSES = ('comp', 'col', 'layer', 'sname', 'species_col')


def parse_args():
    p = argparse.ArgumentParser(description='格子の切り出し精度を数える')
    p.add_argument('--split', default='val', help='images/<split> を対象にする')
    p.add_argument('--dataset', default=DATASET)
    p.add_argument('--weights', default='weights/comptea.pt')
    p.add_argument('--conf', type=float, default=30, help='信頼度の閾値(%%)')
    p.add_argument('--conf-col', type=float, default=20,
                   help='col だけの閾値(%%)．run_pipeline.py と同じ 20 が既定')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--iou', type=float, default=0.5, help='一致とみなす1次元 IoU')
    p.add_argument('--csv', default=None, help='画像ごとの結果を書き出す')
    return p.parse_args()


def class_names(dataset):
    """dataset.yaml の names を読む(yaml に依存しないよう素朴に解く)"""
    names, in_names = {}, False
    for line in (dataset / 'dataset.yaml').read_text(encoding='utf-8').splitlines():
        if line.startswith('names:'):
            in_names = True
            continue
        if in_names:
            if not line.startswith((' ', '\t')):
                break
            k, _, v = line.strip().partition(':')
            names[int(k)] = v.strip()
    return names


def merge_bands(bands):
    """重なりの大きい帯はまとめる

    1地点の表を左右2段に折り返して組んだページでは，同じ行が段ごとにラベル付け
    されている．帯として見るときは同じものなので，畳んでから突き合わせる．
    """
    out = []
    for lo, hi in sorted(bands):
        if out and lo <= out[-1][1] - (out[-1][1] - out[-1][0]) * 0.5:
            out[-1] = (min(out[-1][0], lo), max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def truth_items(label_path, names, w, h):
    """正解ラベルから，クラスごとの帯を取り出す

    帯は (始点, 終点, 横位置の中心)．3つめは段の振り分けに使う．
    1地点の表を左右2段に折り返したページでは，同じ y にある左右の行は
    **別の行**なので，段を分けずに畳むと数が合わなくなる．
    """
    out = {k: [] for k in BANDS}
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding='utf-8').splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        name = names.get(int(f[0]))
        if name not in BANDS:
            continue
        cx, cy, bw, bh = (float(v) for v in f[1:5])
        if BANDS[name] == 'y':
            out[name].append(((cy - bh / 2) * h, (cy + bh / 2) * h, cx * w))
        else:
            out[name].append(((cx - bw / 2) * w, (cx + bw / 2) * w, cx * w))
    return out


def block_ranges(df_loc):
    """段ごとの横の範囲を返す．正解の帯をどの段のものと見るかに使う

    組成セルが1つも作れなかった段(その段の 'col' が検出されなかったとき)も
    範囲を返す．返さないと，その段の正解が隣の段へ流れ込んで
    「取りこぼしが無い」ように見えてしまう．
    """
    if df_loc is None or len(df_loc) == 0 or 'block' not in df_loc:
        return {}, set()
    ranges, empty = {}, set()
    for b, g in df_loc.groupby('block'):
        comp = g[g['obj_name'] == 'comp']
        if comp.empty:
            empty.add(b)
            # 種名の列など，その段に属する縦長のクラスの範囲で代用する．
            # 'header' はページの全幅にかかるので入れない(段の境が消える)
            side = g[g['obj_name'].isin(BLOCK_WIDTH_CLASSES)]
            if side.empty:
                continue
            ranges[b] = (side['x1'].min(), side['x2'].max())
        else:
            ranges[b] = (comp['x1'].min(), comp['x2'].max())
    return ranges, empty


def assign_block(x, ranges):
    """横位置 x を，範囲に入る段へ入れる．どこにも入らなければ一番近い段へ"""
    if not ranges:
        return None
    inside = [b for b, (lo, hi) in ranges.items() if lo <= x <= hi]
    if inside:
        return min(inside, key=lambda b: ranges[b][1] - ranges[b][0])
    return min(ranges, key=lambda b: min(abs(x - ranges[b][0]), abs(x - ranges[b][1])))


def truth_by_block(items, ranges):
    """正解の帯を段ごとに振り分け，段の中だけで畳む"""
    out = {}
    for lo, hi, ctr in items:
        out.setdefault(assign_block(ctr, ranges), []).append((lo, hi))
    return {b: merge_bands(v) for b, v in out.items()}


def located_bands(df_loc):
    """位置決めの結果(comp のセル)から，段ごとの行・列の帯を作り直す"""
    comp = df_loc[df_loc['obj_name'] == 'comp'] if df_loc is not None else None
    out = {k: {} for k in BANDS}
    if comp is None or comp.empty:
        return out
    for (b, _), g in comp.groupby(['block', 'row']):
        out['row'].setdefault(b, []).append((g['y1'].min(), g['y2'].max()))
    for (b, _), g in comp.groupby(['block', 'col']):
        out['col'].setdefault(b, []).append((g['x1'].min(), g['x2'].max()))
    return {k: {b: merge_bands(v) for b, v in d.items()} for k, d in out.items()}


def iou_1d(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match(truth, found, thresh):
    """欲張りに1対1で対応づける(IoU の大きい組から確定させる)"""
    pairs = sorted((iou_1d(t, f), i, j)
                   for i, t in enumerate(truth) for j, f in enumerate(found))
    pairs.reverse()
    used_t, used_f, hits = set(), set(), []
    for v, i, j in pairs:
        if v < thresh or i in used_t or j in used_f:
            continue
        used_t.add(i)
        used_f.add(j)
        hits.append((i, j, v))
    missed = [i for i in range(len(truth)) if i not in used_t]
    extra = [j for j in range(len(found)) if j not in used_f]
    return hits, missed, extra


def run_one(image, args):
    from comptea import locate
    from comptea import detect as detect_mod
    from ultralytics import YOLO

    model = YOLO(args.weights)
    conf, conf_col = args.conf / 100, args.conf_col / 100
    results = model.predict(str(image), conf=min(conf, conf_col),
                            imgsz=args.imgsz, verbose=False)
    results = detect_mod.filter_by_conf(results, conf, {'col': conf_col})
    df_det = detect_mod.split_box_column(results[0].to_df(decimals=2))
    if df_det.empty:
        return None, ['検出が0件']
    df_det['source_image'] = str(image).replace('\\', '/')
    df_det['model'] = args.weights
    # 1ページに表が2つ以上あるなら，表ごとに格子を作る(run_pipeline.py と同じ)．
    # 正解のラベルはページ単位なので，段の番号をずらして繋げてから比べる
    tables, warns = locate.split_tables(df_det)
    parts, shift = [], 0
    from comptea import table_split
    for df_one in tables:
        # 表頭の外の項目行は捨てる(run_pipeline.py と同じ)
        df_one, stray = table_split.drop_stray_plot_rows(df_one, heads=False)
        warns += stray
        df_loc = locate.locate_items(df_one, image=str(image), snap=True)
        warns += list(df_loc.attrs.get('warnings', []))
        if not len(df_loc):
            continue
        # row / col の通し番号はここで付く(run_pipeline.py と同じ手順)
        df_loc = locate.number_cells(df_loc)
        df_loc['block'] = df_loc['block'] + shift
        shift = int(df_loc['block'].max())
        parts.append(df_loc)
    if not parts:
        return df_loc, warns
    return pd.concat(parts, ignore_index=True), warns


def offsets(hits):
    """一致した帯について，ずれと太さを測る

    IoU が 1 に届かない理由は2つに分かれる．どちらかで直し方が変わる．
      ずれ: (格子の中心 - 正解の中心) / 正解の高さ．0 なら中心は合っている
      太さ: 格子の高さ / 正解の高さ．1 より大きければ帯が太い

    格子の境界は検出の中心どうしの中点なので，帯の高さは行ピッチになる．
    正解ラベルの高さは行ピッチの 0.95 倍(全33枚の中央値)なので，
    **太さは 1.05 前後が下限**で，それ以上は取りすぎ．
    """
    if not hits:
        return float('nan'), float('nan')
    shift, size = [], []
    for (tlo, thi), (flo, fhi), _ in hits:
        th = thi - tlo
        if th <= 0:
            continue
        shift.append((((flo + fhi) - (tlo + thi)) / 2) / th)
        size.append((fhi - flo) / th)
    if not shift:
        return float('nan'), float('nan')
    return float(np.median(shift)), float(np.median(size))


def extra_kinds(image, df_loc, extra):
    """余分な帯を，中身で振り分けて数える(値あり / 見出し / まっさら)

    見出しの行と，折り返した学名の行は**ラベルしないのが元からの決まり**なので，
    余分に出ても格子の欠陥ではない．振り分けは黒画素で機械的にやるので目安．
    足すかどうかを決めるときは `label_gaps.py` で切り出して目で見る．
    """
    from comptea import ink
    if not extra:
        return {}
    dark = ink.binarize(image)
    counts = {}
    for b, band in extra:
        g = df_loc[df_loc['block'] == b]
        comp = g[g['obj_name'] == 'comp']
        name = g[g['obj_name'].isin(('species_col', 'sname'))]
        comp_x = (comp['x1'].min(), comp['x2'].max()) if len(comp) else None
        name_x = (name['x1'].min(), name['x2'].max()) if len(name) else None
        base = 0.02
        if comp_x is not None:
            rows = [ink.ratio(dark, y1, y2, *comp_x)
                    for y1, y2 in zip(comp['y1'], comp['y2'])]
            if rows:
                base = float(np.median(rows))
        kind, _, _ = ink.classify(dark, band, comp_x, name_x, base)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def note_counts(df_loc):
    if df_loc is None or 'note' not in df_loc:
        return {}
    marks = {}
    for v in df_loc['note'].fillna(''):
        for part in filter(None, str(v).split(';')):
            marks[part] = marks.get(part, 0) + 1
    return marks


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    names = class_names(dataset)
    img_dir = dataset / 'images' / args.split
    lbl_dir = dataset / 'labels' / args.split
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not images:
        raise SystemExit('画像がない: ' + str(img_dir))

    rows = []
    for image in images:
        w, h = Image.open(image).size
        items = truth_items(lbl_dir / (image.stem + '.txt'), names, w, h)
        df_loc, warns = run_one(image, args)
        found = located_bands(df_loc)
        ranges, empty_blocks = block_ranges(df_loc)

        rec = {'image': image.name, 'warnings': len(warns), 'blocks': len(ranges),
               'blocks_no_comp': len(empty_blocks)}
        print('=== {} ({}x{}) 段 {} ==='.format(image.name, w, h, len(ranges)))
        if empty_blocks:
            print('       ** 組成セルを作れなかった段: {} **'
                  .format(', '.join(str(b) for b in sorted(empty_blocks))))
        for cls in BANDS:
            truth_b = truth_by_block(items[cls], ranges)
            # 段ごとに突き合わせてから足し上げる．取りこぼしの位置も段ごとに出す
            hits, missed, extra, n_t, n_f = [], [], [], 0, 0
            for b in sorted(set(truth_b) | set(found[cls]), key=lambda v: (v is None, v)):
                t, f = truth_b.get(b, []), found[cls].get(b, [])
                h_, m_, e_ = match(t, f, args.iou)
                # 帯そのものを持ち回る．段ごとの添字のままだと後で引けない
                hits += [(t[i], f[j], v) for i, j, v in h_]
                missed += [(b, t[i]) for i in m_]
                extra += [(b, f[j]) for j in e_]
                n_t += len(t)
                n_f += len(f)
            # 余分な帯の中身を見る(値のある行 / 見出しの行 / まっさら)．
            # 見出しの行と，折り返した学名の行はラベルしないのが決まりなので，
            # precision の分母から外した値も併せて出す(2026-09-01 決定)
            kinds = extra_kinds(image, df_loc, extra) if cls == 'row' else {}
            n_skip = kinds.get('header', 0) + kinds.get('blank', 0)
            recall = len(hits) / n_t if n_t else float('nan')
            prec = len(hits) / n_f if n_f else float('nan')
            prec_v = (len(hits) / (n_f - n_skip)
                      if n_f - n_skip > 0 else float('nan'))
            miou = float(np.mean([v for _, _, v in hits])) if hits else float('nan')
            shift, size = offsets(hits)
            rec.update({cls + '_truth': n_t, cls + '_found': n_f,
                        cls + '_hit': len(hits), cls + '_recall': recall,
                        cls + '_prec': prec, cls + '_iou': miou,
                        cls + '_shift': shift, cls + '_size': size})
            if cls == 'row':
                rec.update({'row_extra_value': kinds.get('value', 0),
                            'row_extra_header': kinds.get('header', 0),
                            'row_extra_blank': kinds.get('blank', 0),
                            'row_prec_value': prec_v})
            print('  {:<4} 正解 {:>3} / 格子 {:>3} / 一致 {:>3}'
                  '  recall {:.2f}  prec {:.2f}  平均IoU {:.2f}'
                  '  ずれ {:+.2f}  太さ {:.2f}'
                  .format(cls, n_t, n_f, len(hits), recall, prec, miou, shift, size))
            rec[cls + '_missed'] = len(missed)
            rec[cls + '_extra'] = len(extra)
            if missed:
                pos = ', '.join('段{} {:.0f}-{:.0f}'.format(b, lo, hi)
                                for b, (lo, hi) in missed[:6])
                tail = ' ...' if len(missed) > 6 else ''
                print('       取りこぼし {}: {}{}'.format(len(missed), pos, tail))
            if extra:
                # 正解に対応の無い帯．ラベルの付け漏れか，本当に余分かは
                # 位置を切り出して目で見ないと分からない(label_gaps.py)
                pos = ', '.join('段{} {:.0f}-{:.0f}'.format(b, lo, hi)
                                for b, (lo, hi) in extra[:6])
                tail = ' ...' if len(extra) > 6 else ''
                print('       余分 {}: {}{}'.format(len(extra), pos, tail))
                if kinds:
                    print('       余分の中身: 値あり {} / 見出し {} / まっさら {}'
                          '   prec(見出しを除く) {:.2f}'
                          .format(kinds.get('value', 0), kinds.get('header', 0),
                                  kinds.get('blank', 0), prec_v))
        marks = note_counts(df_loc)
        if marks:
            print('       note: '
                  + ', '.join('{} {}'.format(k, n) for k, n in sorted(marks.items())))
            rec.update({'note_' + k: n for k, n in marks.items()})
        for msg in warns:
            print('       ! ' + str(msg))
        rows.append(rec)

    df = pd.DataFrame(rows)
    print('')
    print('=== まとめ ===')
    for cls in BANDS:
        t = df[cls + '_truth'].sum()
        f = df[cls + '_found'].sum()
        hit = df[cls + '_hit'].sum()
        print('  {:<4} 正解 {} / 格子 {} / 一致 {}  recall {:.3f}  prec {:.3f}'
              .format(cls, t, f, hit,
                      hit / t if t else float('nan'),
                      hit / f if f else float('nan')))
        print('       すべて取れた画像: {}/{}'
              .format(int((df[cls + '_recall'] >= 0.99).sum()), len(df)))
        print('       取りこぼし {} / 余分 {}'
              .format(int(df[cls + '_missed'].sum()), int(df[cls + '_extra'].sum())))
        if cls == 'row' and 'row_extra_value' in df:
            nv = int(df['row_extra_value'].sum())
            nh = int(df['row_extra_header'].sum())
            nb = int(df['row_extra_blank'].sum())
            print('       余分の中身: 値あり {} / 見出し {} / まっさら {}'
                  .format(nv, nh, nb))
            print('       prec(見出しとまっさらを分母から外す) {:.3f}'
                  .format(hit / (f - nh - nb) if f - nh - nb else float('nan')))
        print('       ずれ(中央値) {:+.3f}   太さ(中央値) {:.3f}   ※太さは 1.05 が下限'
              .format(df[cls + '_shift'].median(), df[cls + '_size'].median()))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print('書いた: ' + args.csv)


if __name__ == '__main__':
    sys.exit(main())
