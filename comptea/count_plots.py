"""表頭の帯から地点数を数え，格子の列数と突き合わせる(物差しの改訂 2026-09-03)

前の版は表頭の各行で字の塊を数え「いちばん多い行」を採っていた．
値を縦に積んだ行('83 / 5 / 3)や小数点(1.5)で塊が割れて多く数え，
格子が正しい表を誤りと数えていた(06_p1 は 12 列で正しいのに 15)．
**通し番号の行は等間隔**なので，塊の間隔がいちばん揃った行を採る．

    py -3.12 count_plots.py <作業ディレクトリの親> <切り分けた画像の親>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from . import ink                                      # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def blobs(dark, x1, x2, y1, y2, gap_px):
    """帯の中で横に並ぶ字の塊の中心 x を返す"""
    sub = dark[int(y1):int(y2), int(x1):int(x2)]
    if sub.size == 0:
        return []
    prof = sub.sum(axis=0)
    tol = max(1, int(np.median(prof[prof > 0]) * 0.05)) if (prof > 0).any() else 1
    out, gap, start, last = [], 0, None, 0
    for i, v in enumerate(prof):
        if v > tol:
            if start is None:
                start = i
            last = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_px:
                out.append((start + last) / 2)
                start = None
    if start is not None:
        out.append((start + last) / 2)
    return out


MIN_BLOBS = 5           # これ未満の行は候補にしない
MIN_BLOB_RATIO = 0.5    # 塊の数が最大の行の何割以上を候補にするか


def plots_from_header(dark, bands, x1, x2, gap_px):
    """行ごとに塊を数え，**間隔がいちばん揃った行**の塊の数を返す

    **塊の多い行に絞ってから**揃い具合を比べる(2026-09-03)．
    塊が 2-3 個しかない行は間隔の CV がほぼ 0 になり，通し番号の行
    (塊 24，CV 0.15)に勝ってしまう(20_p2 は塊 2 の行が勝って 3 と数え，
    19_p1 も 52 → 3 に壊れた)．
    """
    counted = []
    for _, b in bands.iterrows():
        c = blobs(dark, x1, x2, b["y1"], b["y2"], gap_px)
        if len(c) >= 3:
            counted.append(c)
    if not counted:
        return 0
    floor = max(MIN_BLOBS, int(max(len(c) for c in counted) * MIN_BLOB_RATIO))
    best = (np.inf, 0)
    for c in counted:
        if len(c) < floor:
            continue
        d = np.diff(c)
        cv = float(np.std(d) / np.mean(d))
        if (cv, -len(c)) < (best[0], -best[1]):
            best = (cv, len(c))
    return best[1]


def main(work_root, parts):
    rows = []
    for w in sorted(glob.glob(os.path.join(work_root, "*"))):
        p = os.path.join(w, "located.csv")
        if not os.path.isfile(p):
            continue
        d = pd.read_csv(p)
        comp = d[d.obj_name == "comp"]
        head = d[d.obj_name == "header_value"]
        if comp.empty or head.empty:
            continue
        stem = os.path.basename(w).split("_t")[0]
        img = os.path.join(parts, stem + ".png")
        # 部分画像に切り出した置き場(_s1 など)は，切り出した PNG が隣にある
        beside = os.path.join(work_root, os.path.basename(w) + ".png")
        if os.path.isfile(beside):
            img = beside
        if not os.path.isfile(img):
            continue
        dark = ink.binarize(Image.open(img))
        cols = comp.groupby("col").agg(x1=("x1", "min"), x2=("x2", "max"))
        pitch = float(np.median(cols["x2"] - cols["x1"]))
        bands = head.groupby("row").agg(y1=("y1", "min"), y2=("y2", "max"))
        x1, x2 = float(comp["x1"].min()), float(comp["x2"].max())
        n_head = plots_from_header(dark, bands, x1, x2, max(6, int(pitch * 0.25)))
        n_plot = int(comp["col"].nunique())
        rows.append(dict(work=os.path.basename(w), cols=n_plot, head=n_head,
                         diff=n_plot - n_head))
    d = pd.DataFrame(rows)
    # **地点数を数えられなかった表は，分母から外す**(2026-09-03)．
    # 塊が 5 未満の行は候補にしないので，地点が 4 つ以下の小さな表では
    # 0 が返る(14_p5 は地点 3・格子 3 列で正しいのに「ずれ 3」と数えていた)．
    known = d["head"] > 0
    d = d[known].copy()
    d["ok"] = d["diff"].abs() <= 1
    print(d.to_string(index=False))
    print(f"\n列数が表頭の地点数と一致(±1): {int(d['ok'].sum())} / {len(d)}")
    print(f"ずれの中央値(絶対値): {d['diff'].abs().median():.0f}   "
          f"ずれの合計: {int(d['diff'].abs().sum())}")
    return d


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
