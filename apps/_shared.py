"""アプリから中核のモジュールを読めるようにする

中核は `comptea/` に平らに置いてあり，互いを `import locate` の形で読む．
Streamlit Cloud は入口のファイルを走らせるので，その場で `sys.path` を通す．
"""
import io
import os
import sys
import tempfile

# ultralytics は設定を `~/.config/Ultralytics` に書こうとする．
# Streamlit Cloud では書けないので，一時ディレクトリに向ける
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(),
                                                      "Ultralytics"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "comptea")
SAMPLE = os.path.join(ROOT, "examples", "sample.jpg")
# 見本の中間データ．前の工程を通さずに試せるようにしておく
SAMPLE_GRID = os.path.join(ROOT, "examples", "sample_grid.zip")
SAMPLE_READ = os.path.join(ROOT, "examples", "sample_read.zip")


class LocalFile(io.BytesIO):
    """アップロードされたファイルと同じ顔をする，手元のファイル

    「見本を使う」を選んだときに，読み込みの処理を分けずに済ませる．
    `zipfile` や `pandas.read_csv` にそのまま渡せるよう `BytesIO` を継ぐ．
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            super().__init__(f.read())
        self.name = os.path.basename(path)

    def getbuffer(self):
        pos = self.tell()
        self.seek(0)
        data = self.read()
        self.seek(pos)
        return data

# **平らな置き場 (`comptea/`) は入れない**(2026-09-07)．入れると
# `import locate` が通ってしまい，パッケージと二重に読み込まれる
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 重みの場所は**パッケージが持つ**(2026-09-07)．ここで組み立て直すと，
# 置き場を変えたときに 2 か所直すことになる
from comptea import WEIGHTS                          # noqa: E402,F401

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


# --- 結果を憶えておく ---------------------------------------------------
# **Streamlit はどのウィジェットを触っても全体を走らせ直す**(2026-09-06)．
# 重い処理を `if st.button(...)` の中だけに置くと，次の再実行では条件が偽に
# なり，出したはずの表も受け取りのボタンも消える(受け取りのボタンを押した
# ときにも再実行が起きるので，押した直後に画面から結果が消えていた)．
# 結果は `st.session_state` に預け，**入力が変わったときだけ捨てる**．


def upload_sig(*ups):
    """入力が変わったかを見分ける印(名前と大きさ)"""
    out = []
    for u in ups:
        if u is None:
            out.append(None)
        else:
            out.append((getattr(u, "name", ""), len(u.getbuffer())))
    return tuple(out)


def cached(key, sig):
    """憶えてある結果を返す(入力が変わっていれば None)"""
    import streamlit as st

    slot = st.session_state.get(key)
    return slot["value"] if slot and slot["sig"] == sig else None


def remember(key, sig, value):
    """結果を憶える(次の再実行でも出せるように)"""
    import streamlit as st

    st.session_state[key] = {"sig": sig, "value": value}


# 工程の一覧(番号・見出し・入口の名前・ひとこと)
APPS = [
    ("1", "折り込みを表ごとに切る", "1_split",
     "A0 級の折り込みや PDF を，空白の帯で**表ごとの画像**に切ります"),
    ("2", "検出して格子を作る", "2_grid",
     "表頭・種名の列・組成部を検出し，**行と列の格子**を作ります"),
    ("3", "セルを読む", "3_read",
     "格子のセルを読み，**被度・種名・階層**を補正します．表頭も表にします"),
    ("4", "縦持ちに組んで検査する", "4_table",
     "**1 行 = 1 地点 × 1 種**の表に組み，機械でできる検査にかけます"),
]
OVERVIEW = (
    "スキャンした**組成表**を，構造化データ (1 行 = 1 地点 × 1 種) にします．"
    "工程を 4 つに分けてあり，**間は zip で受け渡します**．"
    "1 枚に 1 つの表が写っていれば **2 から**始められます．"
)


def app_url(key):
    """入口の URL を返す(決まっていなければ None)

    Streamlit Cloud の Secrets に次のように書いておくと，
    ページどうしを行き来できるようになる．

        [urls]
        1_split = "https://..."
        2_grid  = "https://..."
    """
    try:
        import streamlit as st

        return st.secrets.get("urls", {}).get(key)
    except Exception:                            # noqa: BLE001
        return None


def nav(current):
    """全体像と，ほかの工程への案内を出す(`current` は `2_grid` のような入口の名前)"""
    import streamlit as st

    st.info(OVERVIEW)
    with st.sidebar:
        st.header("工程")
        for num, title, key, note in APPS:
            url = app_url(key)
            if key == current:
                st.markdown(f"**▶ {num}. {title}**")
            elif url:
                st.markdown(f"[{num}. {title}]({url})")
            else:
                st.markdown(f"{num}. {title}")
            st.caption(note)
        if not any(app_url(k) for _, _, k, _ in APPS):
            st.caption(
                "ほかの工程への行き先は，Secrets の `[urls]` に書くと"
                "リンクになります．"
            )


def footer():
    """どのアプリにも出す但し書き"""
    import streamlit as st

    st.divider()
    st.caption(
        "comptea — 植生学の組成表を構造化データにする道具．"
        "見本の画像は，宮脇昭 (編) 1984.『日本植生誌 近畿』至文堂 からの引用です．"
        "大きな折り込みや，まとめて処理する場合は CUI 版を使ってください．"
    )
