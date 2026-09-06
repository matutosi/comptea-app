"""4 組み上げ: 縦持ちに組んで検査する

入力  `ocred.csv`(3 の出力)
出力  `comp_table_long.csv`(1 行 = 1 地点 × 1 種)と検査の結果

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
)

up = st.file_uploader("ocred.csv (3 の出力)", type=["csv"])
if up is None:
    st.info("3 で作った `ocred.csv` を選んでください．")
    _shared.footer()
    st.stop()

import pandas as pd                              # noqa: E402

work = tempfile.mkdtemp(prefix="comptea_")
wd = os.path.join(work, "out")
os.makedirs(wd, exist_ok=True)
d = pd.read_csv(up)
d.to_csv(os.path.join(wd, "ocred.csv"), index=False, encoding="utf-8-sig")
st.caption(f"読み取り: {len(d)} セル")

if st.button("組み上げる", type="primary"):
    cli = os.path.join(_shared.ROOT, "cli", "build_table.py")
    with st.spinner("組み上げています"):
        r = subprocess.run([sys.executable, cli, wd], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           cwd=_shared.CORE)
    p = os.path.join(wd, "comp_table_long.csv")
    if os.path.isfile(p):
        lg = pd.read_csv(p)
        ok = int((lg["status"] == "OK").sum())
        ng = int((lg["status"] == "Need Check").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("縦持ちの行", len(lg))
        c2.metric("OK", ok)
        c3.metric("Need Check", ng)
        st.dataframe(lg.head(50), use_container_width=True)
        st.download_button("comp_table_long.csv を受け取る", open(p, "rb").read(),
                           file_name="comp_table_long.csv", mime="text/csv")
        q = os.path.join(wd, "checks.txt")
        if os.path.isfile(q):
            with st.expander("検査の結果", expanded=True):
                st.text(open(q, encoding="utf-8").read())
    else:
        st.error("組み上げに失敗しました")
        st.code(((r.stdout or "") + (r.stderr or ""))[-3000:])
_shared.footer()
