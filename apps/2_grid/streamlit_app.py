"""2 格子: 検出して格子を作る(関門1)

入力  表の画像
出力  `located.csv`(セルの座標)と重ね描き

中身は CUI と同じ `cli/run_pipeline.py` を呼ぶ．
関門の作り(格子を見てから先へ進む)をそのまま画面にする．
"""
import os
import subprocess
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

st.set_page_config(page_title="comptea 2 格子", layout="wide")
st.title("2. 検出して格子を作る (関門1)")
st.write(
    "表の領域(表頭・種名の列・組成部)を検出し，**行と列はアルゴリズムで**決めます．"
    "できた格子を重ね描きで確かめてください．"
)

up = st.file_uploader("表の画像", type=["png", "jpg", "jpeg"])
use_sample = st.checkbox("見本の画像を使う", value=up is None)
if up is None and not use_sample:
    st.info("画像を選ぶか，見本を使ってください．")
    _shared.footer()
    st.stop()

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work = tempfile.mkdtemp(prefix="comptea_")
if up is not None:
    src = os.path.join(work, up.name)
    with open(src, "wb") as f:
        f.write(up.getbuffer())
else:
    src = _shared.SAMPLE
img = Image.open(src)
msg = _shared.too_big(img)
if msg:
    st.error(msg)
    _shared.footer()
    st.stop()
st.caption(f"画像: {img.size[0]} x {img.size[1]} px")

if st.button("格子を作る", type="primary"):
    cli = os.path.join(_shared.ROOT, "cli", "run_pipeline.py")
    wd = os.path.join(work, "out")
    with st.spinner("検出しています(初回はモデルの読み込みに時間がかかります)"):
        r = subprocess.run(
            [sys.executable, cli, src, "--weights", _shared.WEIGHTS, "--workdir", wd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=_shared.CORE)
    log = (r.stdout or "") + (r.stderr or "")
    overlay = os.path.join(wd, "overlay.png")
    if os.path.isfile(overlay):
        st.image(overlay, caption="格子の重ね描き", use_container_width=True)
    for name, label in (("summary.txt", "まとめ"), ("located.csv", "格子")):
        p = os.path.join(wd, name)
        if not os.path.isfile(p):
            continue
        if name.endswith(".txt"):
            with st.expander(label, expanded=True):
                st.text(open(p, encoding="utf-8").read())
        else:
            st.download_button(f"{label} ({name}) を受け取る",
                               open(p, "rb").read(), file_name=name, mime="text/csv")
    if not os.path.isfile(overlay):
        st.error("格子が作れませんでした")
        st.code(log[-3000:])
_shared.footer()
