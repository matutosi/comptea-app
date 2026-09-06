"""アプリから中核のモジュールを読めるようにする

中核は `comptea/` に平らに置いてあり，互いを `import locate` の形で読む．
Streamlit Cloud は入口のファイルを走らせるので，その場で `sys.path` を通す．
"""
import os
import sys
import tempfile

# ultralytics は設定を `~/.config/Ultralytics` に書こうとする．
# Streamlit Cloud では書けないので，一時ディレクトリに向ける
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(),
                                                      "Ultralytics"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "comptea")
WEIGHTS = os.path.join(CORE, "weights", "comptea.pt")
SAMPLE = os.path.join(ROOT, "examples", "sample.jpg")

for p in (CORE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# 無料枠のメモリに収まる大きさ．超える画像は手元の CUI へ誘導する
MAX_SIDE = 4000
# 切り分けは検出を通さないぶん軽いので，少し大きくても扱える
MAX_SPLIT_SIDE = 8000


def too_big(img):
    """大きすぎる画像なら，その旨の文言を返す(問題なければ None)"""
    if max(img.size) > MAX_SIDE:
        return (f"この画像は長辺 {max(img.size)} px あります．"
                f"ここでは長辺 {MAX_SIDE} px までしか扱えません"
                "(無料枠のメモリに収まらないため)．"
                "大きな折り込みは，手元で CUI を使ってください．")
    return None


def zip_files(work, names):
    """作業ディレクトリの中の何枚かを zip にまとめて返す(無ければ None)

    工程のあいだの受け渡しは，**ファイルが 2 つ以上なら zip** にする．
    1 つずつ受け取ると，次の工程で入れ忘れが起きる．
    """
    import io
    import zipfile

    have = [n for n in names if os.path.isfile(os.path.join(work, n))]
    if not have:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in have:
            z.write(os.path.join(work, n), n)
    return buf.getvalue()


def unzip_into(upload, work):
    """受け取った zip(または CSV 1 枚)を作業ディレクトリへ展開する

    Returns:
        置いたファイル名のリスト
    """
    import zipfile

    name = getattr(upload, "name", "")
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(upload) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            for n in names:
                with open(os.path.join(work, os.path.basename(n)), "wb") as f:
                    f.write(z.read(n))
        return [os.path.basename(n) for n in names]
    with open(os.path.join(work, name), "wb") as f:
        f.write(upload.getbuffer())
    return [name]


def footer():
    """どのアプリにも出す但し書き"""
    import streamlit as st

    st.divider()
    st.caption(
        "comptea — 植生学の組成表を構造化データにする道具．"
        "見本の画像は，宮脇昭 (編) 1984.『日本植生誌 近畿』至文堂 からの引用です．"
        "大きな折り込みや，まとめて処理する場合は CUI 版を使ってください．"
    )
