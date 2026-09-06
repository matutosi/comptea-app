"""2 格子: 検出して格子を作る

入力  表の画像
出力  `located.csv`(セルの座標)と重ね描き

中身は CUI と同じ `cli/run_pipeline.py` を呼ぶ．
格子を見てから先へ進む，という工程の分け方をそのまま画面にする．
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

_shared.start("2_grid", "格子",
    "表の領域(表頭・種名の列・組成部)を検出し，**行と列はアルゴリズムで**決めます．"
    "できた格子を重ね描きで確かめてください．")

up = st.file_uploader("表の画像か PDF", type=["png", "jpg", "jpeg", "pdf"])
use_sample = st.checkbox("見本の画像を使う", value=up is None)
if up is None and not use_sample:
    _shared.stop_with("画像か PDF を選ぶか，見本を使ってください．")

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work, wd = _shared.workdir()
if up is not None:
    src = os.path.join(work, up.name)
    with open(src, "wb") as f:
        f.write(up.getbuffer())
    # PDF は**貼ってある画像をそのまま取り出して** PNG にし，以降はそれを使う
    # (描き直すと再標本化で字が甘くなる)
    if up.name.lower().endswith(".pdf"):
        from comptea import split_sheet                   # noqa: E402

        try:
            import fitz                          # noqa: E402  PyMuPDF
            n = fitz.open(src).page_count
        except Exception as e:                   # noqa: BLE001
            st.error(f"PDF を開けません: {e}")
            _shared.footer()
            st.stop()
        page = 1
        if n > 1:
            page = st.number_input(f"ページ (1-{n})", 1, n, 1,
                                   help="1 ページに 1 つの表があるものを選びます")
        with st.spinner("PDF のページを PNG に直しています"):
            stem = os.path.splitext(up.name)[0]
            png = os.path.join(work, f"{stem}_page{int(page)}.png")
            split_sheet.load_page(src, page=int(page) - 1).save(png)
        src = png
        st.caption(f"PDF {n} ページのうち {int(page)} ページ目を使います")
else:
    src = _shared.SAMPLE
img = Image.open(src)
msg = _shared.too_big(img)
if msg:
    _shared.fail_with(msg)
st.caption(f"画像: {img.size[0]} x {img.size[1]} px")

sig = _shared.upload_sig(up) + (os.path.basename(src),)
res = _shared.cached("2_grid", sig)

if st.button("格子を作る", type="primary"):
    with st.spinner("検出しています(初回はモデルの読み込みに時間がかかります)"):
        # **同じプロセスで呼ぶ**(2026-09-07)．前は Python を起こし直しており，
        # そのたびに torch を読み込んでいた(1 回 5 秒)
        _, log = pipeline.run("grid", [src, "--weights", _shared.WEIGHTS,
                                       "--workdir", wd])
    overlay = os.path.join(wd, "overlay.png")
    if not os.path.isfile(overlay):
        _shared.fail_with("格子が作れませんでした", log)
    p_sum = os.path.join(wd, "summary.txt")
    # **中身を憶える**．作業ディレクトリは再実行のたびに作り直される
    res = {
        "overlay": open(overlay, "rb").read(),
        "summary": (open(p_sum, encoding="utf-8").read()
                    if os.path.isfile(p_sum) else ""),
        # 次の工程へは zip で渡す．読み取りには格子と検出の両方が要る
        "zip": _shared.zip_files(wd, ["located.csv", "detect.csv", "summary.txt",
                                      "overlay.png"]),
    }
    _shared.remember("2_grid", sig, res)

if res:
    st.image(res["overlay"], caption="格子の重ね描き", use_container_width=True)
    if res["summary"]:
        with st.expander("まとめ", expanded=True):
            st.text(res["summary"])
    _shared.offer(res["zip"], "grid.zip",
                  "この zip をそのまま「3. セルを読む」に渡してください．")
_shared.footer()
