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
# 中核も，古い写しが残っていれば捨てる(次に使うときに読み直される)
_shared.reload_core()

from comptea import pipeline            # noqa: E402

_shared.start("2_grid", "格子",
    "表の領域(表頭・種名の列・組成部)を検出し，**行と列はアルゴリズムで**決めます．"
    "できた格子を重ね描きで確かめてください．")

# **組(組成表と，その続きのページ)をまとめて受け取れる**(2026-09-08)．
# 流し込み(出現1回の種・調査地)は紙面に収まらないと次のページへあふれる．
# どのページが続きかは**ファイル名の枝番**で示す(`xxx-1.jpg` 表 / `xxx-2.jpg` 続き)．
ups = st.file_uploader("表の画像か PDF(組なら 2 枚まとめて)",
                       type=["png", "jpg", "jpeg", "pdf"],
                       accept_multiple_files=True)
ups = ups or []
st.caption("続きのページがあるときは，`xxx-1.jpg`(組成表)と `xxx-2.jpg`(続き)を"
           "まとめて選びます．枝番が付いていれば，続きのページで表が無くても"
           "異常ではありません．")
use_sample = st.checkbox("見本の画像を使う", value=not ups)
if not ups and not use_sample:
    _shared.stop_with("画像か PDF を選ぶか，見本を使ってください．")
up = ups[0] if ups else None

from PIL import Image                            # noqa: E402

Image.MAX_IMAGE_PIXELS = None
work, wd = _shared.workdir()


def take(up_one, work_dir):
    """受け取った 1 枚を置き場に書き，画像の場所を返す(PDF は PNG に直す)"""
    path = os.path.join(work_dir, up_one.name)
    with open(path, "wb") as f:
        f.write(up_one.getbuffer())
    if not up_one.name.lower().endswith(".pdf"):
        return path
    # PDF は**貼ってある画像をそのまま取り出して** PNG にし，以降はそれを使う
    # (描き直すと再標本化で字が甘くなる)
    from comptea import split_sheet                       # noqa: E402

    try:
        import fitz                              # noqa: E402  PyMuPDF
        n = fitz.open(path).page_count
    except Exception as e:                       # noqa: BLE001
        _shared.fail_with(f"PDF を開けません: {e}")
    page = 1
    if n > 1:
        page = st.number_input(f"{up_one.name} のページ (1-{n})", 1, n, 1,
                               key=f"page_{up_one.name}",
                               help="1 ページに 1 つの表があるものを選びます")
    with st.spinner("PDF のページを PNG に直しています"):
        stem = os.path.splitext(up_one.name)[0]
        png = os.path.join(work_dir, f"{stem}_page{int(page)}.png")
        split_sheet.load_page(path, page=int(page) - 1).save(png)
    st.caption(f"{up_one.name}: {n} ページのうち {int(page)} ページ目を使います")
    return png


if ups:
    srcs = [take(u, work) for u in ups]
else:
    srcs = [_shared.SAMPLE]
for s in srcs:
    im = Image.open(s)
    msg = _shared.too_big(im)
    if msg:
        _shared.fail_with(msg)
    st.caption(f"{os.path.basename(s)}: {im.size[0]} x {im.size[1]} px")

sig = _shared.upload_sig(*ups) + tuple(os.path.basename(s) for s in srcs)
res = _shared.cached("2_grid", sig)

if st.button("格子を作る", type="primary"):
    tables, conts, logs = [], [], []
    for i, src in enumerate(srcs):
        # **枝番は画像の名前で決まる**ので，置き場も画像の名前にする
        one_wd = os.path.join(wd, os.path.splitext(os.path.basename(src))[0])
        with st.spinner(f"検出しています({os.path.basename(src)})"):
            # **同じプロセスで呼ぶ**(2026-09-07)．前は Python を起こし直しており，
            # そのたびに torch を読み込んでいた(1 回 5 秒)
            _, log = pipeline.run("grid", [src, "--weights", _shared.WEIGHTS,
                                           "--workdir", one_wd])
        logs.append(log)
        # **1 枚に表が 2 つ以上あると，別々の置き場ができる**(2026-09-07)．
        # `wd` だけを見ていたので，そういう紙面は「作れなかった」ことになっていた
        found = _shared.grid_tables(one_wd, src)
        for name, d, part in found:
            p_sum = os.path.join(d, "summary.txt")
            p_over = os.path.join(d, "overlay.png")
            # 格子が別の画像(傾きを直したもの・部分画像)を指していれば，
            # **その画像も一緒に渡す**(格子の座標はその画像のもの)
            if part:
                import shutil

                shutil.copy(part, os.path.join(d, "image.png"))
            # 渡す表では，場所をファイル名だけにする
            _shared.strip_paths(d)
            tables.append({
                "name": name,
                "overlay": (open(p_over, "rb").read()
                            if os.path.isfile(p_over) else None),
                "summary": (open(p_sum, encoding="utf-8").read()
                            if os.path.isfile(p_sum) else ""),
                "zip": _shared.zip_files(d, ["located.csv", "detect.csv",
                                             "summary.txt", "overlay.png",
                                             "image.png"]),
            })
        # **続きのページは，表が無くても異常ではない**(2026-09-08)．
        # 枝番が付いていれば，流し込みと注記を切り出して次の工程へ渡す
        got = _shared.continuation(one_wd)
        if got:
            conts.append(got)
    if not tables and not conts:
        _shared.fail_with("格子が作れませんでした", logs[-1] if logs else None)
    # **中身を憶える**．作業ディレクトリは再実行のたびに作り直される
    res = {"tables": tables, "conts": conts}
    _shared.remember("2_grid", sig, res)

if res:
    tables = res["tables"]
    for c in res.get("conts", []):
        st.info(f'**{c["page"]} は続きのページです**'
                f'(枝番 {c["part"]}．表は `{c["table"]}` にまとまります)．'
                "格子はできません．流し込みと注記を読んで，"
                "`link_pages.py` で表につなぎます．")
        if c["once"]:
            st.image(c["once"], caption="流し込み(出現1回の種)",
                     use_container_width=True)
        if c["note"]:
            st.image(c["note"], caption="注記(調査地・調査年月日・出典)",
                     use_container_width=True)
        _shared.offer(c["zip"], f'{c["page"]}_continuation.zip')
    if not tables:
        _shared.footer()
        st.stop()
    if len(tables) > 1:
        st.warning(f"**この紙面には表が {len(tables)} 個あります**．"
                   "別々の表なので，**1 つずつ**次の工程へ渡してください"
                   "(地点も表頭も別のものです)．")
        names = [t["name"] for t in tables]
        pick = st.selectbox("どの表を見ますか", names, key="which_table")
        one = tables[names.index(pick)]
    else:
        one = tables[0]

    if one["overlay"]:
        st.image(one["overlay"], caption=f'{one["name"]} の格子',
                 use_container_width=True)
    if one["summary"]:
        with st.expander("まとめ", expanded=True):
            st.text(one["summary"])
    _shared.offer(one["zip"], f'{one["name"]}_grid.zip',
                  "この zip をそのまま「3. セルを読む」に渡してください．")
_shared.footer()
