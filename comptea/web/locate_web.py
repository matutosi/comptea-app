#Import All the Required Libraries
import streamlit as st
from pathlib import Path
import os
import pandas as pd
import sys
import traceback
import tempfile
import shutil

# Set WD import custom modules
if '__file__' not in locals():
    WD = os.getcwd()
else:
    WD = Path(__file__).parent

os.chdir(WD)
if WD not in sys.path:
    sys.path.append(str(WD))
import util_file

# local library
import locate
import draw_rect

# ページ設定
st.set_page_config(page_title="Location Finder", layout="wide")
st.title("🗺️ Table Cell Location Finder")

# サイドバー: 設定
with st.sidebar:
    st.header("⚙️ Configurations")

    # 1. 入力ファイル設定
    st.subheader("📁 Input File")
    use_example = st.checkbox(
        "Use example",
        value=True,
        help="Use detection_results.csv as example"
    )

    if use_example:
        path_detection = "detection_results.csv"
        st.info(f"Using: {path_detection}")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Upload detection results CSV",
            type=["csv"],
            help="Upload a CSV file from detect_web.py"
        )
        path_detection = None

    # 2. パラメータ設定
    st.subheader("⚙️ Parameters")
    threth_col = st.slider(
        "Overlap threshold (col)",
        0.0, 1.0, value=0.8, step=0.05,
        help="Threshold for removing duplicate columns (higher = more strict)"
    )
    threth_row = st.slider(
        "Overlap threshold (row)",
        0.0, 1.0, value=0.8, step=0.05,
        help="Threshold for removing duplicate rows (higher = more strict)"
    )

# メイン処理
try:
    # データ読み込み
    if use_example:
        if not os.path.exists(path_detection):
            st.error(f"❌ File not found: {path_detection}")
            st.info("💡 Please run detect_web.py first to generate detection results.")
            st.stop()
        df = pd.read_csv(path_detection)
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("📤 Please upload a detection results CSV file")
        st.stop()

    # データ検証
    required_columns = ["obj_name", "x1", "y1", "x2", "y2", "confidence", "source_image", "model"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info(f"Required columns: {required_columns}")
        st.stop()

    if len(df) == 0:
        st.warning("⚠️ No detections found in the CSV file")
        st.stop()

    # 位置計算
    with st.spinner("🔍 Calculating grid locations..."):
        df_located = locate.locate_items(
            df, threth_col=threth_col, threth_row=threth_row,
            image=st.session_state.get("source_image_path"))

    # 位置を決められなかったものの理由を伝える(黙って空を返さない)
    for w in df_located.attrs.get("warnings", []):
        st.warning(f"⚠️ {w}")

    if len(df_located) == 0:
        st.warning("⚠️ No items located (all detections filtered out)")
        st.info("💡 Try adjusting the overlap threshold parameter")
        st.stop()

    st.success(f"✅ Located {len(df_located)} items")

    # 結果表示
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📊 Located Results")
        st.dataframe(df_located, use_container_width=True, height=400)

    with col2:
        st.subheader("📈 Statistics")
        st.metric("Total Items", len(df_located))

        # オブジェクト別の統計
        obj_counts = df_located['obj_name'].value_counts()
        st.write("**Items by type:**")
        for obj_name, count in obj_counts.items():
            st.write(f"- {obj_name}: {count}")

        # 3. CSV ダウンロード (zip圧縮)
        st.subheader("💾 Download")

        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp()
        temp_csv_path = Path(temp_dir, "located_results.csv")
        df_located.to_csv(temp_csv_path, index=False)

        # zip圧縮（時刻付きファイル名）
        zip_file = util_file.zip_now(temp_dir, zip_name="located_results")

        # ダウンロードボタン
        with open(zip_file, "rb") as f:
            st.download_button(
                label="📦 Download as ZIP",
                data=f.read(),
                file_name=Path(zip_file).name,
                mime="application/zip"
            )

        # クリーンアップ
        shutil.rmtree(temp_dir)
        if Path(zip_file).exists():
            Path(zip_file).unlink()

    # 可視化
    st.subheader("🖼️ Visualization")
    with st.spinner("🎨 Drawing rectangles..."):
        image = draw_rect.draw_rects_df(df_located)

    if image is not None:
        st.image(image, caption="Located Items with Grid Coordinates", use_container_width=True)
        if (df_located.get('note', pd.Series(dtype=str)).fillna('') != '').any():
            st.caption(
                "太い枠は位置決めで気になったセル: "
                "🟡 gold = 検出漏れを内挿 / 🟠 orange = 文字を避けてずらした / "
                "🔴 red = ずらしても文字に重なったまま")
    else:
        st.error("❌ Failed to draw rectangles")

except FileNotFoundError as e:
    st.error(f"❌ File not found: {str(e)}")
    st.info("💡 Make sure the file path is correct and the file exists")

except pd.errors.EmptyDataError:
    st.error("❌ The CSV file is empty")

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    with st.expander("🔍 Show detailed error"):
        st.code(traceback.format_exc())
