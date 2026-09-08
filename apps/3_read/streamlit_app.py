"""3 読み取り: セルを読む

入力  表の画像 と，2 の zip(または `located.csv`)
出力  `ocred.csv`・`plot_table.csv`(表頭)・`review.tsv` を zip で

中身は CUI と同じ `cli/run_ocr.py` を呼ぶ．
EasyOCR は**初回にモデルを 100 MB ほど落とす**ので，最初の 1 回は待ち時間が長い．
"""
import importlib
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

# **読み込み済みのモジュールは走らせ直されない**(2026-09-06)．
# Streamlit は書き換えたファイルを走らせ直すが，`import` した先はそのままなので，
# 更新した直後に古い `_shared` が残り，足したばかりのものが無いと言われる
importlib.reload(_shared)
# 中核も，古い写しが残っていれば捨てる(次に使うときに読み直される)
_shared.reload_core()

from comptea import pipeline            # noqa: E402

_shared.start("3_read", "読み取り",
    "格子のセルを 1 つずつ読み，被度・種名・階層を補正します．"
    "読めなかったセルは字種を絞って読み直し，値にならないものは "
    "`Need Check` として残します．**表頭の項目**も表にして出します．")

use_sample = st.checkbox("見本の画像・データを使う", value=True,
                         help="2 を通さずに，この場で試せます")
col1, col2 = st.columns(2)
if use_sample:
    img_up = _shared.LocalFile(_shared.SAMPLE)
    loc_up = _shared.LocalFile(_shared.SAMPLE_GRID)
    col1.info(f"見本の画像: {img_up.name}")
    col2.info(f"見本の格子: {loc_up.name}")
else:
    img_up = col1.file_uploader("表の画像", type=["png", "jpg", "jpeg"])
    loc_up = col2.file_uploader("2 の結果 (grid.zip か located.csv)",
                                type=["zip", "csv"])
if img_up is None or loc_up is None:
    _shared.stop_with("2 で受け取った zip と，同じ画像を選んでください．")

from PIL import Image                            # noqa: E402
import pandas as pd                              # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work, wd = _shared.workdir()
src = os.path.join(work, img_up.name)
with open(src, "wb") as f:
    f.write(img_up.getbuffer())
msg = _shared.too_big(Image.open(src))
if msg:
    _shared.fail_with(msg)

loc_path = _shared.take_zip(loc_up, wd, "located.csv",
                            "2 の zip を渡してください．")
# **zip に画像が入っていれば，そちらを使う**(2026-09-07)．
# 1 枚に表が 2 つある紙面は，2 が部分画像に切り出して格子を作る．
# 格子の座標はその部分画像のものなので，元の紙面に当てるとずれる
part = os.path.join(wd, "image.png")
if os.path.isfile(part):
    src = part
    st.info("2 が格子を作った画像(傾きを直したもの，または切り出した部分画像)を"
            "使います．格子の座標はこの画像のものです．")

# 画像の場所は，この場で置いた写しに読み替える
loc = pd.read_csv(loc_path)
if "source_image" in loc.columns:
    loc["source_image"] = src
loc.to_csv(loc_path, index=False, encoding="utf-8-sig")
det_path = os.path.join(wd, "detect.csv")
if os.path.isfile(det_path):
    det = pd.read_csv(det_path)
    if "source_image" in det.columns:
        det["source_image"] = src
        det.to_csv(det_path, index=False, encoding="utf-8-sig")
else:
    pd.DataFrame([{"obj_name": "table", "source_image": src, "x1": 0, "y1": 0,
                   "x2": 0, "y2": 0, "confidence": 1.0}]).to_csv(
        det_path, index=False, encoding="utf-8-sig")
st.caption(f"格子: {len(loc)} セル")

sig = _shared.upload_sig(img_up, loc_up)
res = _shared.cached("3_read", sig)

if st.button("読む", type="primary"):
    with st.spinner("読んでいます(初回はモデルの取得で数分かかります)"):
        # 同じプロセスで呼ぶ．easyocr の読み込みは 7 秒かかるので，
        # 一度読めばこのアプリが生きているあいだは使い回せる
        _, log = pipeline.run("read", [wd, "--reader", "easyocr"])
    p_ocr = os.path.join(wd, "ocred.csv")
    if not os.path.isfile(p_ocr):
        _shared.fail_with("読み取りに失敗しました", log)
    d = pd.read_csv(p_ocr)

    # **表頭も表にして出す**．組み上げを待たずに，調査地・面積・海抜高を見たい
    from comptea import plot_table                            # noqa: E402

    warn = None
    try:
        df_plot = plot_table.plot_table(d)
    except Exception as e:                       # noqa: BLE001
        df_plot = pd.DataFrame()
        warn = f"表頭を表にできませんでした: {e}"
    plot_warnings = list(df_plot.attrs.get("warnings", [])) if len(df_plot) else []
    if len(df_plot):
        df_plot.to_csv(os.path.join(wd, "plot_table.csv"), index=False,
                       encoding="utf-8-sig")
    # **中身を憶える**．読み取りは数分かかるので，再実行でやり直させない．
    # 場所をファイル名だけにしてから読み直す(画面で直した値を書き戻すときも，
    # 一時ディレクトリの長い場所が混じらない)
    _shared.strip_paths(wd)
    d = pd.read_csv(p_ocr)
    res = {
        "ocred": d,
        "plot": df_plot if len(df_plot) else None,
        "plot_warnings": plot_warnings,
        "warn": warn,
        "zip": _shared.zip_files(wd, ["ocred.csv", "plot_table.csv", "review.tsv",
                                      "located.csv", "detect.csv"]),
    }
    _shared.remember("3_read", sig, res)

if res:
    d = res["ocred"]
    ng = int((d["status"] == "Need Check").sum()) if "status" in d else 0
    st.success(f"{len(d)} セルを読みました(Need Check {ng} 件)")
    if res["warn"]:
        st.warning(res["warn"])
    if res["plot"] is not None:
        st.subheader(f"表頭 ({len(res['plot'])} 地点)")
        st.dataframe(res["plot"], use_container_width=True)
        for w in res["plot_warnings"]:
            st.caption(f"! {w}")

    st.subheader("読んだセル")
    c1, c2 = st.columns([1, 2])
    n_show = _shared.how_many("出す行数", "n_read")
    only_ng = c2.checkbox("Need Check だけ出す", value=False,
                          help="直すところだけを見たいとき")
    view = d[d["status"] == "Need Check"] if only_ng and "status" in d else d
    view = view if n_show is None else view.head(n_show)
    st.caption(f"{len(view)} 行を出しています(全 {len(d)} 行)．"
               "**`corrected` は直せます**．直したら下のボタンで反映してください．")

    shown = _shared.cell_images(view, src)
    # **`suggest` は「採らなかった候補」**(2026-09-08)．読みが短いと，
    # 濁点だけの違いでないかぎり辞書の候補を採らず，印字を残して目視に回す．
    # 候補は**誤りうる**(`クツル` に `ハクツル` が出て，正解は `クロヅル` だった)．
    # 見るのは画像で，候補は参考にとどめる
    cols = [c for c in ["画像", "cell_id", "obj_name", "row", "col",
                        "text", "corrected", "suggest", "status", "note"]
            if c in shown.columns]
    edited = st.data_editor(
        shown[cols], use_container_width=True, hide_index=True,
        disabled=[c for c in cols if c != "corrected"],
        column_config={
            "画像": st.column_config.ImageColumn("実物", width="medium"),
            "text": st.column_config.TextColumn("読み", width="small"),
            "corrected": st.column_config.TextColumn("直した値", width="small"),
            "suggest": st.column_config.TextColumn(
                "採らなかった候補", width="small",
                help="読みが短いので採らなかった候補．**誤りうる**ので，"
                     "実物の画像を見て決める"),
        },
        key="read_editor")

    if st.button("直した値を反映する"):
        from comptea import correct_text                  # noqa: E402

        n = 0
        for _, r in edited.iterrows():
            i = d.index[d["cell_id"] == r["cell_id"]]
            if not len(i) or str(d.loc[i[0], "corrected"]) == str(r["corrected"]):
                continue
            i = i[0]
            # **人の入力は「読み」として扱う**(CUI の apply_text と同じ．2026-09-07)．
            # 同じ規則で整えて corrected と status を作る．入力をそのまま
            # corrected に入れると，`5・5` のように整っていない値が
            # 「OK」のまま次の工程へ行き，値として通らない
            again = correct_text.correct_cell(d.loc[i, "obj_name"],
                                              str(r["corrected"]))
            d.loc[i, "text"] = r["corrected"]
            d.loc[i, "corrected"] = again["corrected"]
            d.loc[i, "status"] = again["status"]
            note = str(d.loc[i, "note"] or "").strip(";") if "note" in d else ""
            d.loc[i, "note"] = (note + ";" if note else "") + "hand"
            n += 1
        if n:
            res["ocred"] = d
            res["zip"] = _shared.replace_in_zip(
                res["zip"], "ocred.csv", d.to_csv(index=False))
            _shared.remember("3_read", sig, res)
            st.success(f"{n} 件を直しました(`note` に `hand`)．"
                       "下の zip に反映してあります．")
            st.rerun()
        else:
            st.info("直したところはありませんでした．")

    _shared.offer(res["zip"], "read.zip",
                  "この zip をそのまま「4. 縦持ちに組んで検査する」に渡してください．")
_shared.footer()
