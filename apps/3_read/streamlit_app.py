"""3 読み取り: セルを読む

入力  表の画像 と `located.csv`(2 の出力)
出力  `ocred.csv`

中身は CUI と同じ `cli/run_ocr.py` を呼ぶ．
EasyOCR は**初回にモデルを 100 MB ほど落とす**ので，最初の 1 回は待ち時間が長い．
"""
import os
import subprocess
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

st.set_page_config(page_title="comptea 3 読み取り", layout="wide")
st.title("3. セルを読む")
st.write(
    "格子のセルを 1 つずつ読み，被度・種名・階層を補正します．"
    "読めなかったセルは字種を絞って読み直し，値にならないものは "
    "`Need Check` として残します．"
)

col1, col2 = st.columns(2)
img_up = col1.file_uploader("表の画像", type=["png", "jpg", "jpeg"])
loc_up = col2.file_uploader("located.csv (2 の出力)", type=["csv"])
if img_up is None or loc_up is None:
    st.info("2 で作った `located.csv` と，同じ画像を選んでください．")
    _shared.footer()
    st.stop()

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work = tempfile.mkdtemp(prefix="comptea_")
wd = os.path.join(work, "out")
os.makedirs(wd, exist_ok=True)
src = os.path.join(work, img_up.name)
with open(src, "wb") as f:
    f.write(img_up.getbuffer())
msg = _shared.too_big(Image.open(src))
if msg:
    st.error(msg)
    _shared.footer()
    st.stop()

import pandas as pd                              # noqa: E402

loc = pd.read_csv(loc_up)
# 画像の場所は，この場で置いた写しに読み替える
if "source_image" in loc.columns:
    loc["source_image"] = src
loc.to_csv(os.path.join(wd, "located.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([{"obj_name": "table", "source_image": src, "x1": 0, "y1": 0,
               "x2": 0, "y2": 0, "confidence": 1.0}]).to_csv(
    os.path.join(wd, "detect.csv"), index=False, encoding="utf-8-sig")
st.caption(f"格子: {len(loc)} セル")

if st.button("読む", type="primary"):
    cli = os.path.join(_shared.ROOT, "cli", "run_ocr.py")
    with st.spinner("読んでいます(初回はモデルの取得で数分かかります)"):
        r = subprocess.run([sys.executable, cli, wd, "--reader", "easyocr"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=_shared.CORE)
    p = os.path.join(wd, "ocred.csv")
    if os.path.isfile(p):
        d = pd.read_csv(p)
        ng = int((d["status"] == "Need Check").sum()) if "status" in d else 0
        st.success(f"{len(d)} セルを読みました(Need Check {ng} 件)")
        st.dataframe(d.head(30), use_container_width=True)
        st.download_button("ocred.csv を受け取る", open(p, "rb").read(),
                           file_name="ocred.csv", mime="text/csv")
    else:
        st.error("読み取りに失敗しました")
        st.code(((r.stdout or "") + (r.stderr or ""))[-3000:])
_shared.footer()
