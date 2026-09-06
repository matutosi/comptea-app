"""1 切り分け: 大きな折り込みを，表ごとの画像に切る

入力  折り込みの画像，または PDF(ページごとに PNG へ直してから切る)
出力  表ごとの画像(zip)

この工程は画像しか触らないので，依存は Pillow・numpy・PyMuPDF だけ．
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
    "**PDF も受け取れます**(ページごとに PNG へ直してから切ります)．"
)

up = st.file_uploader("折り込みの画像か PDF", type=["png", "jpg", "jpeg", "pdf"])
if up is None:
    st.info("画像か PDF を選んでください．")
    _shared.footer()
    st.stop()

import tempfile                                  # noqa: E402

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work = tempfile.mkdtemp(prefix="comptea_")
src = os.path.join(work, up.name)
with open(src, "wb") as f:
    f.write(up.getbuffer())

import ink                                       # noqa: E402
import split_sheet                               # noqa: E402

# PDF は**貼ってある画像をそのまま取り出す**(描き直すと字が甘くなる)．
# 取り出したものを PNG にして，以降はその PNG を使う
pages = []
stem = os.path.splitext(up.name)[0]
if up.name.lower().endswith(".pdf"):
    try:
        import fitz                              # noqa: E402  PyMuPDF
        n = fitz.open(src).page_count
    except Exception as e:                       # noqa: BLE001
        st.error(f"PDF を開けません: {e}")
        _shared.footer()
        st.stop()
    st.caption(f"PDF: {n} ページ")
    with st.spinner("ページを PNG に直しています"):
        for i in range(n):
            img = split_sheet.load_page(src, page=i)
            p = os.path.join(work, f"{stem}_page{i + 1}.png")
            img.save(p)
            pages.append((f"{stem}_page{i + 1}", img, p))
else:
    img = Image.open(src)
    pages.append((stem, img, src))

for name, img, _ in pages:
    st.caption(f"{name}: {img.size[0]} x {img.size[1]} px")
big = [n for n, im, _ in pages if max(im.size) > _shared.MAX_SPLIT_SIDE]
if big:
    st.error(
        f"長辺が {_shared.MAX_SPLIT_SIDE} px を超えるページがあります({', '.join(big)})．"
        "ここでは扱えないので，手元で CUI を使ってください．"
    )
    _shared.footer()
    st.stop()

buf = io.BytesIO()
total = 0
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for name, img, path in pages:
        with st.spinner(f"{name}: 空白の帯を探しています"):
            boxes = split_sheet.find_tables(ink.binarize(img))
        st.write(f"**{name}** — {len(boxes)} 個の表に分かれました")
        cols = st.columns(min(3, max(1, len(boxes))))
        for i, b in enumerate(boxes, 1):
            part = img.crop(tuple(int(v) for v in b))
            with cols[(i - 1) % len(cols)]:
                st.image(part, caption=f"{i}: {part.size[0]} x {part.size[1]} px",
                         use_container_width=True)
            p = io.BytesIO()
            part.convert("L").save(p, format="PNG")
            z.writestr(f"{name}_p{i}.png", p.getvalue())
            total += 1
st.download_button(f"表ごとの画像を受け取る (zip・{total} 枚)", buf.getvalue(),
                   file_name=f"{stem}_tables.zip", mime="application/zip",
                   type="primary")
st.caption("この中の 1 枚を「2. 検出して格子を作る」に渡してください．")
_shared.footer()
