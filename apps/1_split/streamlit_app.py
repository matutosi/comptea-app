"""1 切り分け: 大きな折り込みを，表ごとの画像に切る

入力  折り込みの画像(または PDF)
出力  表ごとの画像(zip)

この工程は画像しか触らないので，依存は Pillow と numpy だけ．
"""
import io
import os
import sys
import zipfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

st.set_page_config(page_title="comptea 1 切り分け", layout="wide")
st.title("1. 折り込みを表ごとに切る")
st.write(
    "A0 級の折り込みには，表が 2-5 個並んでいます．"
    "そのまま検出にかけると，別々の表の地点が 1 つの番号体系に混ざります．"
    "**空白の帯**で切り分けて，表ごとの画像にします．"
)

up = st.file_uploader("折り込みの画像", type=["png", "jpg", "jpeg"])
if up is None:
    st.info("画像を選んでください．")
    _shared.footer()
    st.stop()

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
img = Image.open(up)
st.caption(f"読み込んだ画像: {img.size[0]} x {img.size[1]} px")

import ink                                       # noqa: E402
import split_sheet                               # noqa: E402

with st.spinner("空白の帯を探しています"):
    boxes = split_sheet.find_tables(ink.binarize(img))
st.success(f"{len(boxes)} 個の表に分かれました")

cols = st.columns(min(3, max(1, len(boxes))))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for i, b in enumerate(boxes, 1):
        part = img.crop(tuple(int(v) for v in b))
        with cols[(i - 1) % len(cols)]:
            st.image(part, caption=f"{i}: {part.size[0]} x {part.size[1]} px",
                     use_container_width=True)
        p = io.BytesIO()
        part.save(p, format="PNG")
        z.writestr(f"{os.path.splitext(up.name)[0]}_p{i}.png", p.getvalue())
st.download_button("表ごとの画像をまとめて受け取る", buf.getvalue(),
                   file_name="tables.zip", mime="application/zip")
_shared.footer()
