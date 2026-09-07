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

# 無料枠(1 GB ほど)に収まる大きさ．**測って決めた**(2026-09-07)．
# Streamlit Cloud が入れる CPU 版 torch で，見本の表を検出したときの山:
#   imgsz 1280 → 432 MB   1856 → 609 MB   2560 → 781 MB
#   imgsz 3200 → 857 MB   4288 → 1,091 MB(収まらない)
# 検出の縮尺は紙面の長辺で決まる(`split_sheet.auto_imgsz`)ので，
# **imgsz が 2560 までに収まる紙面**に限る(長辺およそ 6600 px)
MAX_IMGSZ = 2560
# 切り分けは検出を通さないので，効くのは画素数だけ．
# A0 の折り込み(9344 x 12873 = 120 Mpx)で山は 520 MB だった．
# **長辺で切ってはいけない**(2026-09-07 ユーザ報告)．折り込みの PDF は
# 長辺 12873 px あり，この道具が受け持つべき紙面そのものだった
MAX_SPLIT_PIXELS = 150_000_000


def too_big(img):
    """検出にかけるには大きすぎる画像なら，その旨の文言を返す(問題なければ None)"""
    from comptea import split_sheet

    sz = split_sheet.auto_imgsz(*img.size)
    if sz <= MAX_IMGSZ:
        return None
    return (f"この画像は {img.size[0]} x {img.size[1]} px あります．"
            f"学習時と同じ縮尺で検出するには imgsz={sz} が要り，"
            f"無料枠のメモリ(1 GB ほど)に収まりません"
            f"(ここでは imgsz={MAX_IMGSZ}，長辺 6600 px ほどまで)．"
            "**「1. 折り込みを表ごとに切る」で表ごとに切ってから**渡すか，"
            "大きな紙面は手元で CUI を使ってください．")


def too_big_to_split(img):
    """切り分けにも大きすぎる画像なら，その旨の文言を返す"""
    px = img.size[0] * img.size[1]
    if px <= MAX_SPLIT_PIXELS:
        return None
    return (f"この画像は {img.size[0]} x {img.size[1]} px "
            f"({px / 1e6:.0f} Mpx)あります．"
            f"ここでは {MAX_SPLIT_PIXELS / 1e6:.0f} Mpx までしか扱えません"
            "(無料枠のメモリに収まらないため)．手元で CUI を使ってください．")


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


def reload_core():
    """中核 (`comptea`) の**古い写しが残らない**ようにする(2026-09-07)

    Streamlit は書き換えたファイルを走らせ直すが，`import` した先は
    そのまま残る．公開先に新しい版を入れても，画面を触り直すだけでは
    古いモジュールが使われ，**足したばかりの関数が「無い」と言われる**
    (`split_sheet.cut_table` で実際に起きた．ユーザ報告)．

    毎回読み直すと重い(`ocr` は easyocr を読むので 5 秒)ので，
    **ファイルの更新時刻が変わったときだけ**捨てる．次に使うときに読み直される．
    """
    import pathlib
    import sys

    import streamlit as st

    newest = 0.0
    for q in pathlib.Path(CORE).rglob("*.py"):
        try:
            newest = max(newest, q.stat().st_mtime)
        except OSError:                          # noqa: PERF203  消えた写し
            continue
    key = "_core_mtime"
    old = st.session_state.get(key)
    st.session_state[key] = newest
    if old is None or newest <= old:
        return 0
    stale = [m for m in list(sys.modules) if m == "comptea" or m.startswith("comptea.")]
    for m in stale:
        del sys.modules[m]
    return len(stale)


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
    # **`st.secrets` は，ファイルが無いと画面に誤りを出す**(2026-09-07)．
    # 工程の一覧を作るたびに触るので，1 画面に 8 個並んでいた．
    # 置いてあるかを先に見て，無ければ触らない
    if not _secrets_file():
        return None
    try:
        import streamlit as st

        return st.secrets.get("urls", {}).get(key)
    except Exception:                            # noqa: BLE001
        return None


def _secrets_file():
    """Secrets のファイルがあるか(Cloud は `~/.streamlit/secrets.toml` に置く)"""
    import pathlib

    for p in (pathlib.Path.cwd() / ".streamlit" / "secrets.toml",
              pathlib.Path.home() / ".streamlit" / "secrets.toml"):
        if p.is_file():
            return p
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


# --- 4 つのアプリに共通の型 ---------------------------------------------
# 「頭を出す → 受け取る → 展開する → 無ければ知らせる → 走らせる →
#  憶える → zip を渡す」という並びは 4 つとも同じ．写しておくと，
# 直しが 4 か所に散る(2026-09-06 の不具合は 4 つ全部に同じ形で入っていた)


def start(key, tab, lead):
    """見出し・全体像・ほかの工程への案内までを出す

    見出しは `APPS` の 1 か所で決める(以前は各アプリに書いてあり，
    寄せるときに短い名前へ変わってしまった)．`tab` はブラウザのタブの名前．
    """
    import streamlit as st

    num, title = next((n, t) for n, t, k, _ in APPS if k == key)
    st.set_page_config(page_title=f"comptea {num} {tab}", layout="wide")
    st.title(f"{num}. {title}")
    st.write(lead)
    nav(key)


def stop_with(msg):
    """知らせて，但し書きを出して止める"""
    import streamlit as st

    st.info(msg)
    footer()
    st.stop()


def fail_with(msg, log=None):
    """できなかったことを知らせて止める(理由の末尾を添える)"""
    import streamlit as st

    st.error(msg)
    if log:
        st.code(log[-3000:])
    footer()
    st.stop()


def workdir():
    """この実行のための置き場を作る(再実行のたびに新しく作られる)"""
    import os
    import tempfile

    work = tempfile.mkdtemp(prefix="comptea_")
    wd = os.path.join(work, "out")
    os.makedirs(wd, exist_ok=True)
    return work, wd


def take_zip(upload, wd, want, hint):
    """受け取った zip(か CSV 1 枚)を展開し，要るファイルの場所を返す

    CSV を 1 枚だけ渡されたときは，その名前を `want` に読み替える
    (工程の側は決まった名前しか見ない)．
    """
    import os

    put = unzip_into(upload, wd)
    path = os.path.join(wd, want)
    if os.path.isfile(path):
        return path
    csvs = [n for n in put if n.lower().endswith(".csv")]
    if not csvs:
        fail_with(f"`{want}` が見つかりません．{hint}")
    os.replace(os.path.join(wd, csvs[0]), path)
    return path


def offer(zip_bytes, file_name, caption=None):
    """結果の受け取りを出す(中身が無ければ何もしない)"""
    import streamlit as st

    if not zip_bytes:
        return
    st.download_button("結果をまとめて受け取る (zip)", zip_bytes,
                       file_name=file_name, mime="application/zip",
                       type="primary")
    if caption:
        st.caption(caption)


# --- 表の見せ方 ---------------------------------------------------------

SHOW_CHOICES = [30, 100, 300, 1000, 0]        # 0 は「すべて」


def how_many(label, key, default=30):
    """何行まで出すかを選ばせる(長い表をそのまま出すと画面が重い)"""
    import streamlit as st

    n = st.selectbox(label, SHOW_CHOICES, index=SHOW_CHOICES.index(default),
                     format_func=lambda v: "すべて" if v == 0 else f"{v} 行",
                     key=key)
    return None if n == 0 else int(n)


def cell_images(df, image, max_w=260):
    """セルの箱を切り出して，`画像` の列に入れた写しを返す

    **読んだ字の隣に実物を置く**．`status` だけでは，どこが違うのか分からない
    (`S・K` を `So` と読んだセルは，読みだけ見ても気づけない)．
    Streamlit の `ImageColumn` は data URI をそのまま出せる．
    """
    import base64
    import io

    from PIL import Image

    if not image or not os.path.isfile(image):
        return df
    need = {"x1", "y1", "x2", "y2"}
    if not need <= set(df.columns):
        return df
    src = Image.open(image)
    out = []
    for _, r in df.iterrows():
        try:
            box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            cell = src.crop(box).convert("L")
            if cell.width > max_w:
                h = max(1, int(cell.height * max_w / cell.width))
                cell = cell.resize((max_w, h), Image.LANCZOS)
            buf = io.BytesIO()
            cell.save(buf, format="PNG")
            out.append("data:image/png;base64,"
                       + base64.b64encode(buf.getvalue()).decode())
        except Exception:                        # noqa: BLE001  切り出せない箱
            out.append(None)
    df = df.copy()
    df.insert(0, "画像", out)
    return df


def replace_in_zip(blob, name, text):
    """zip の中の 1 枚を書き換えた写しを返す(直した値を次の工程へ渡すため)"""
    import io
    import zipfile

    if not blob:
        return blob
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(blob)) as src, \
            zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = text.encode("utf-8-sig") if info.filename == name \
                else src.read(info.filename)
            dst.writestr(info.filename, data)
    return buf.getvalue()


def footer():
    """どのアプリにも出す但し書き"""
    import streamlit as st

    st.divider()
    st.caption(
        "comptea — 植生学の組成表を構造化データにする道具．"
        "見本の画像は，宮脇昭 (編) 1984.『日本植生誌 近畿』至文堂 からの引用です．"
        "大きな折り込みや，まとめて処理する場合は CUI 版を使ってください．"
    )
