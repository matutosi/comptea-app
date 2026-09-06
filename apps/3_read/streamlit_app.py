"""3 読み取り: セルを読む

入力  表の画像 と，2 の zip(または `located.csv`)
出力  `ocred.csv`・`plot_table.csv`(表頭)・`review.tsv` を zip で

中身は CUI と同じ `cli/run_ocr.py` を呼ぶ．
EasyOCR は**初回にモデルを 100 MB ほど落とす**ので，最初の 1 回は待ち時間が長い．
"""
import importlib
import os
import subprocess
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

# **読み込み済みのモジュールは走らせ直されない**(2026-09-06)．
# Streamlit は書き換えたファイルを走らせ直すが，`import` した先はそのままなので，
# 更新した直後に古い `_shared` が残り，足したばかりのものが無いと言われる
importlib.reload(_shared)

st.set_page_config(page_title="comptea 3 読み取り", layout="wide")
st.title("3. セルを読む")
st.write(
    "格子のセルを 1 つずつ読み，被度・種名・階層を補正します．"
    "読めなかったセルは字種を絞って読み直し，値にならないものは "
    "`Need Check` として残します．**表頭の項目**も表にして出します．"
)

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
    st.info("2 で受け取った zip と，同じ画像を選んでください．")
    _shared.footer()
    st.stop()

from PIL import Image                            # noqa: E402
import pandas as pd                              # noqa: E402

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

put = _shared.unzip_into(loc_up, wd)
loc_path = os.path.join(wd, "located.csv")
if not os.path.isfile(loc_path):
    # CSV を 1 枚だけ渡された場合は，その名前を located.csv にする
    csvs = [n for n in put if n.lower().endswith(".csv")]
    if not csvs:
        st.error("`located.csv` が見つかりません．2 の zip を渡してください．")
        _shared.footer()
        st.stop()
    os.replace(os.path.join(wd, csvs[0]), loc_path)

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

if st.button("読む", type="primary"):
    cli = os.path.join(_shared.ROOT, "cli", "run_ocr.py")
    with st.spinner("読んでいます(初回はモデルの取得で数分かかります)"):
        r = subprocess.run([sys.executable, cli, wd, "--reader", "easyocr"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=_shared.CORE)
    p = os.path.join(wd, "ocred.csv")
    if not os.path.isfile(p):
        st.error("読み取りに失敗しました")
        st.code(((r.stdout or "") + (r.stderr or ""))[-3000:])
        _shared.footer()
        st.stop()
    d = pd.read_csv(p)
    ng = int((d["status"] == "Need Check").sum()) if "status" in d else 0
    st.success(f"{len(d)} セルを読みました(Need Check {ng} 件)")

    # **表頭も表にして出す**．組み上げを待たずに，調査地・面積・海抜高を見たい
    sys.path.insert(0, _shared.CORE)
    import plot_table                            # noqa: E402

    try:
        df_plot = plot_table.plot_table(d)
    except Exception as e:                       # noqa: BLE001
        df_plot = pd.DataFrame()
        st.warning(f"表頭を表にできませんでした: {e}")
    if len(df_plot):
        df_plot.to_csv(os.path.join(wd, "plot_table.csv"), index=False,
                       encoding="utf-8-sig")
        st.subheader(f"表頭 ({len(df_plot)} 地点)")
        st.dataframe(df_plot, use_container_width=True)
        for w in df_plot.attrs.get("warnings", []):
            st.caption(f"! {w}")

    st.subheader("読んだセル (先頭 30)")
    st.dataframe(d.head(30), use_container_width=True)
    z = _shared.zip_files(wd, ["ocred.csv", "plot_table.csv", "review.tsv",
                               "located.csv", "detect.csv"])
    st.download_button("結果をまとめて受け取る (zip)", z,
                       file_name="read.zip", mime="application/zip",
                       type="primary")
    st.caption("この zip をそのまま「4. 縦持ちに組んで検査する」に渡してください．")
_shared.footer()
