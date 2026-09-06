"""1 切り分け: 大きな折り込みを，表ごとの画像に切る

入力  折り込みの画像，または PDF(ページごとに PNG へ直してから切る)
出力  表ごとの画像(zip)

この工程は画像しか触らないので，依存は Pillow・numpy・PyMuPDF だけ．
"""
import io
import importlib
import os
import sys
import zipfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _shared                                  # noqa: E402

# **読み込み済みのモジュールは走らせ直されない**(2026-09-06)．
# Streamlit は書き換えたファイルを走らせ直すが，`import` した先はそのままなので，
# 更新した直後に古い `_shared` が残り，足したばかりのものが無いと言われる
importlib.reload(_shared)

_shared.start("1_split", "切り分け",
    "A0 級の折り込みには，表が 2-5 個並んでいます．"
    "そのまま検出にかけると，別々の表の地点が 1 つの番号体系に混ざります．"
    "**空白の帯**で切り分けて，表ごとの画像にします．"
    "**PDF も受け取れます**(ページごとに PNG へ直してから切ります)．")

up = st.file_uploader("折り込みの画像か PDF", type=["png", "jpg", "jpeg", "pdf"])
if up is None:
    _shared.stop_with("画像か PDF を選んでください．")

import tempfile                                  # noqa: E402

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work, _wd = _shared.workdir()
src = os.path.join(work, up.name)
with open(src, "wb") as f:
    f.write(up.getbuffer())

from comptea import ink                                       # noqa: E402
from comptea import split_sheet                               # noqa: E402

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

# **切り分けは 1 度だけ**．ここはボタンの中に無いので，前は再実行のたびに
# やり直していた(A0 の紙面では毎回数十秒．受け取りのボタンを押しても起きる)
sig = _shared.upload_sig(up)
res = _shared.cached("1_split", sig)

if res is None:
    buf = io.BytesIO()
    cuts = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, img, path in pages:
            with st.spinner(f"{name}: 空白の帯を探しています"):
                boxes = split_sheet.find_tables(ink.binarize(img))
            names = []
            for i, b in enumerate(boxes, 1):
                part = img.crop(tuple(int(v) for v in b)).convert("L")
                q = io.BytesIO()
                part.save(q, format="PNG")
                fn = f"{name}_p{i}.png"
                z.writestr(fn, q.getvalue())
                names.append((fn, part.size))
            cuts.append((name, names))
    res = {"zip": buf.getvalue(), "cuts": cuts,
           "total": sum(len(n) for _, n in cuts)}
    _shared.remember("1_split", sig, res)

# 表示は憶えた zip から読み出す(切り出した画像を二重に持たない)
with zipfile.ZipFile(io.BytesIO(res["zip"])) as z:
    for name, names in res["cuts"]:
        st.write(f"**{name}** — {len(names)} 個の表に分かれました")
        cols = st.columns(min(3, max(1, len(names))))
        for i, (fn, size) in enumerate(names, 1):
            with cols[(i - 1) % len(cols)]:
                st.image(z.read(fn), caption=f"{i}: {size[0]} x {size[1]} px",
                         use_container_width=True)

st.download_button(f"表ごとの画像を受け取る (zip・{res['total']} 枚)", res["zip"],
                   file_name=f"{stem}_tables.zip", mime="application/zip",
                   type="primary")
st.caption("この中の 1 枚を「2. 検出して格子を作る」に渡してください．")
_shared.footer()
