"""4 組み上げ: 縦持ちに組んで検査する

入力  3 の zip(または `ocred.csv`)
出力  `comp_table_long.csv`・`comp_table_wide.csv`・`plot_table.csv`・`checks.txt` を zip で

中身は CUI と同じ `cli/build_table.py` を呼ぶ．
画像を触らないので依存は pandas だけ．いちばん軽い工程．
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

from comptea import pipeline            # noqa: E402

_shared.start("4_table", "組み上げ",
    "読んだセルを縦持ちの表に組みます(1 行 = 1 地点 × 1 種)．"
    "地点数・重複・被度の形を検査し，引っかかったものを知らせます．"
    "**表頭の項目**も 1 行 = 1 地点の表にします．")

use_sample = st.checkbox("見本のデータを使う", value=True,
                         help="2・3 を通さずに，この場で試せます")
if use_sample:
    up = _shared.LocalFile(_shared.SAMPLE_READ)
    st.info(f"見本の読み取り: {up.name}")
else:
    up = st.file_uploader("3 の結果 (read.zip か ocred.csv)", type=["zip", "csv"])
if up is None:
    _shared.stop_with("3 で受け取った zip を選んでください．")

import pandas as pd                              # noqa: E402

work, wd = _shared.workdir()
p_ocr = _shared.take_zip(up, wd, "ocred.csv", "3 の zip を渡してください．")
st.caption(f"読み取り: {len(pd.read_csv(p_ocr))} セル")

sig = _shared.upload_sig(up)
res = _shared.cached("4_table", sig)

if st.button("組み上げる", type="primary"):
    with st.spinner("組み上げています"):
        _, log = pipeline.run("table", [wd])
    p_long = os.path.join(wd, "comp_table_long.csv")
    if not os.path.isfile(p_long):
        _shared.fail_with("組み上げに失敗しました", log)
    p_plot = os.path.join(wd, "plot_table.csv")
    p_chk = os.path.join(wd, "checks.txt")
    # **中身を憶える**．作業ディレクトリは再実行のたびに作り直されるので，
    # 場所ではなく中身を預ける
    res = {
        "long": pd.read_csv(p_long),
        "plot": pd.read_csv(p_plot) if os.path.isfile(p_plot) else None,
        "checks": (open(p_chk, encoding="utf-8").read()
                   if os.path.isfile(p_chk) else ""),
        "zip": _shared.zip_files(wd, ["comp_table_long.csv", "comp_table_wide.csv",
                                      "plot_table.csv", "checks.txt", "ocred.csv"]),
    }
    _shared.remember("4_table", sig, res)

if res:
    lg = res["long"]
    c1, c2, c3 = st.columns(3)
    c1.metric("縦持ちの行", len(lg))
    c2.metric("OK", int((lg["status"] == "OK").sum()))
    c3.metric("Need Check", int((lg["status"] == "Need Check").sum()))
    st.subheader("縦持ちの表")
    n_show = _shared.how_many("出す行数", "n_long", default=100)
    only_ng = st.checkbox("Need Check だけ出す", value=False)
    view = lg[lg["status"] == "Need Check"] if only_ng and "status" in lg else lg
    st.caption(f"{len(view if n_show is None else view.head(n_show))} 行を"
               f"出しています(全 {len(lg)} 行)")
    st.dataframe(view if n_show is None else view.head(n_show),
                 use_container_width=True)

    if res["plot"] is not None:
        st.subheader(f"表頭 ({len(res['plot'])} 地点)")
        st.dataframe(res["plot"], use_container_width=True)

    _shared.offer(res["zip"], "table.zip")
    if res["checks"]:
        with st.expander("検査の結果", expanded=True):
            st.text(res["checks"])
_shared.footer()
