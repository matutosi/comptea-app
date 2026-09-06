"""アプリから中核のモジュールを読めるようにする

中核は `comptea/` に平らに置いてあり，互いを `import locate` の形で読む．
Streamlit Cloud は入口のファイルを走らせるので，その場で `sys.path` を通す．
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "comptea")
WEIGHTS = os.path.join(CORE, "weights", "comptea.pt")
SAMPLE = os.path.join(ROOT, "examples", "sample.jpg")

for p in (CORE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# 無料枠のメモリに収まる大きさ．超える画像は手元の CUI へ誘導する
MAX_SIDE = 4000


def too_big(img):
    """大きすぎる画像なら，その旨の文言を返す(問題なければ None)"""
    if max(img.size) > MAX_SIDE:
        return (f"この画像は長辺 {max(img.size)} px あります．"
                f"ここでは長辺 {MAX_SIDE} px までしか扱えません"
                "(無料枠のメモリに収まらないため)．"
                "大きな折り込みは，手元で CUI を使ってください．")
    return None


def footer():
    """どのアプリにも出す但し書き"""
    import streamlit as st

    st.divider()
    st.caption(
        "comptea — 植生学の組成表を構造化データにする道具．"
        "見本の画像は引用です(出典は README を見てください)．"
        "大きな折り込みや，まとめて処理する場合は CUI 版を使ってください．"
    )
