"""4 組み上げ: 縦持ちに組んで検査する

入力  3 の zip(または `ocred.csv`)
出力  `comp_table_long.csv`・`comp_table_wide.csv`・`plot_table.csv`・`checks.txt` を zip で

中身は CUI と同じ `cli/build_table.py` を呼ぶ．
画像を触らないので依存は pandas だけ．いちばん軽い工程．
"""
import os
import subprocess
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

st.set_page_config(page_title="comptea 4 組み上げ", layout="wide")
st.title("4. 縦持ちに組んで検査する")
st.write(
    "読んだセルを縦持ちの表に組みます(1 行 = 1 地点 × 1 種)．"
    "地点数・重複・被度の形を検査し，引っかかったものを知らせます．"
    "**表頭の項目**も 1 行 = 1 地点の表にします．"
)

use_sample = st.checkbox("見本のデータを使う", value=True,
                         help="2・3 を通さずに，この場で試せます")
if use_sample:
    up = _shared.LocalFile(_shared.SAMPLE_READ)
    st.info(f"見本の読み取り: {up.name}")
else:
    up = st.file_uploader("3 の結果 (read.zip か ocred.csv)", type=["zip", "csv"])
if up is None:
    st.info("3 で受け取った zip を選んでください．")
    _shared.footer()
    st.stop()

import pandas as pd                              # noqa: E402

work = tempfile.mkdtemp(prefix="comptea_")
wd = os.path.join(work, "out")
os.makedirs(wd, exist_ok=True)
put = _shared.unzip_into(up, wd)
p_ocr = os.path.join(wd, "ocred.csv")
if not os.path.isfile(p_ocr):
    csvs = [n for n in put if n.lower().endswith(".csv")]
    if not csvs:
        st.error("`ocred.csv` が見つかりません．3 の zip を渡してください．")
        _shared.footer()
        st.stop()
    os.replace(os.path.join(wd, csvs[0]), p_ocr)
st.caption(f"読み取り: {len(pd.read_csv(p_ocr))} セル")

if st.button("組み上げる", type="primary"):
    cli = os.path.join(_shared.ROOT, "cli", "build_table.py")
    with st.spinner("組み上げています"):
        r = subprocess.run([sys.executable, cli, wd], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           cwd=_shared.CORE)
    p = os.path.join(wd, "comp_table_long.csv")
    if not os.path.isfile(p):
        st.error("組み上げに失敗しました")
        st.code(((r.stdout or "") + (r.stderr or ""))[-3000:])
        _shared.footer()
        st.stop()
    lg = pd.read_csv(p)
    c1, c2, c3 = st.columns(3)
    c1.metric("縦持ちの行", len(lg))
    c2.metric("OK", int((lg["status"] == "OK").sum()))
    c3.metric("Need Check", int((lg["status"] == "Need Check").sum()))
    st.subheader("縦持ちの表 (先頭 50)")
    st.dataframe(lg.head(50), use_container_width=True)

    q = os.path.join(wd, "plot_table.csv")
    if os.path.isfile(q):
        df_plot = pd.read_csv(q)
        st.subheader(f"表頭 ({len(df_plot)} 地点)")
        st.dataframe(df_plot, use_container_width=True)

    z = _shared.zip_files(wd, ["comp_table_long.csv", "comp_table_wide.csv",
                               "plot_table.csv", "checks.txt", "ocred.csv"])
    st.download_button("結果をまとめて受け取る (zip)", z,
                       file_name="table.zip", mime="application/zip",
                       type="primary")
    c = os.path.join(wd, "checks.txt")
    if os.path.isfile(c):
        with st.expander("検査の結果", expanded=True):
            st.text(open(c, encoding="utf-8").read())
_shared.footer()
