import streamlit as st
from pathlib import Path
import os
import pandas as pd
import sys
import traceback
import tempfile
import shutil

# to avoid error
import torch
torch.classes.__path__ = []  # set to empty.

from comptea import util_file
from comptea import ocr
from comptea import locate
from comptea import correct_text

# ページ設定
st.set_page_config(page_title="OCR & Text Correction", layout="wide")
st.title("📖 OCR & Text Correction")

# サイドバー: 設定
with st.sidebar:
    st.header("⚙️ Configurations")

    # 1. 入力ファイル設定
    st.subheader("📁 Input File")
    use_example = st.checkbox(
        "Use example",
        value=True,
        help="Use located.csv as example"
    )

    if use_example:
        path_located = "located.csv"
        st.info(f"Using: {path_located}")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Upload located results CSV",
            type=["csv"],
            help="Upload a CSV file from locate_web.py"
        )
        path_located = None

# メイン処理
try:
    # データ読み込み
    if use_example:
        if not os.path.exists(path_located):
            st.error(f"❌ File not found: {path_located}")
            st.info("💡 Please run locate_web.py first to generate located results.")
            st.stop()
        df_all = pd.read_csv(path_located)
    elif uploaded_file:
        df_all = pd.read_csv(uploaded_file)
    else:
        st.info("📤 Please upload a located results CSV file")
        st.stop()

    # データ検証
    required_columns = ["source_image", "x1", "y1", "x2", "y2", "obj_name"]
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info(f"Required columns: {required_columns}")
        st.stop()

    if len(df_all) == 0:
        st.warning("⚠️ No data found in the CSV file")
        st.stop()

    # OCR実行
    st.subheader("🔍 Running OCR")
    df_image = df_all.groupby('source_image')

    # 画像ごとに処理
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    image_list = list(df_image.groups.keys())
    for idx, (image_path, df) in enumerate(df_image):
        status_text.text(f"Processing image {idx + 1}/{len(image_list)}: {Path(image_path).name}")

        with st.spinner(f"Running OCR on {Path(image_path).name}..."):
            result_df = ocr.ocr_images_df(df).dropna()
            all_results.append(result_df)

        progress_bar.progress((idx + 1) / len(image_list))

    # 結果を統合
    result_df = pd.concat(all_results, ignore_index=True)
    status_text.empty()
    progress_bar.empty()

    if len(result_df) == 0:
        st.warning("⚠️ No OCR results obtained")
        st.stop()

    st.success(f"✅ OCR completed: {len(result_df)} items extracted")

    # グリッド座標の付与
    with st.spinner("📊 Assigning grid coordinates..."):
        # 段が複数あるページでは，行は段をまたいで通し，列は段ごとに振る
        result_df = locate.number_cells(result_df)

    # テキスト自動修正
    with st.spinner("✏️ Correcting text..."):
        for i, row in result_df.iterrows():
            match row['obj_name']:
                case 'species_col':  # 和名
                    corrected = correct_text.correct_name(row['text'], target='j_name')
                case 'sname':  # 学名
                    corrected = correct_text.correct_name(row['text'], target='s_name')
                case 'layer':  # 階層
                    corrected = correct_text.correct_layer(row['text'])
                case 'comp':  # 組成値
                    corrected = correct_text.correct_comp(row['text'])
                case _:
                    corrected = {'corrected': None, 'status': None}

            # データ修正
            result_df.loc[i, 'corrected'] = corrected['corrected']
            result_df.loc[i, 'status'] = corrected['status']
            # 読みが短くて採らなかった候補(2026-09-08)．**誤りうる**ので，
            # 値には入れず，目視のための参考として別の列に置く
            if corrected.get('suggest'):
                result_df.loc[i, 'suggest'] = corrected['suggest']

    st.success("✅ Text correction completed")

    # 結果表示（2カラムレイアウト）
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📝 Edit OCR Results")
        st.info("💡 You can edit the 'corrected' column directly in the table below")

        corrected_df = st.data_editor(
            result_df[['obj_name', 'col', 'row', 'img_base64', 'text', 'corrected', 'status']],
            column_config={
                "img_base64": st.column_config.ImageColumn("original image", width="medium"),
                "obj_name": st.column_config.TextColumn("Type", width="small"),
                "col": st.column_config.NumberColumn("Col", width="small"),
                "row": st.column_config.NumberColumn("Row", width="small"),
                "text": st.column_config.TextColumn("OCR Text", width="medium"),
                "corrected": st.column_config.TextColumn("Corrected", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small")
            },
            height=600,
            use_container_width=True
        )

    with col2:
        st.subheader("📈 Statistics")
        st.metric("Total Items", len(result_df))

        # オブジェクトタイプ別の統計
        st.write("**Items by type:**")
        obj_counts = result_df['obj_name'].value_counts()
        for obj_name, count in obj_counts.items():
            st.write(f"- {obj_name}: {count}")

        # 修正ステータス別の統計
        st.write("**Correction status:**")
        if 'status' in result_df.columns:
            status_counts = result_df['status'].value_counts()
            for status, count in status_counts.items():
                st.write(f"- {status}: {count}")

        # オブジェクトタイプ別の修正数とエラー数
        st.write("**Corrections by type:**")
        for obj_name in result_df['obj_name'].unique():
            obj_df = result_df[result_df['obj_name'] == obj_name]
            # 修正数（text と corrected が異なる）
            corrected_count = len(obj_df[obj_df['text'] != obj_df['corrected']])
            # エラー数（status が error や None など）
            error_count = len(obj_df[obj_df['status'].isna() | (obj_df['status'] == 'error')])
            st.write(f"- **{obj_name}**")
            st.write(f"  - Corrected: {corrected_count}")
            st.write(f"  - Errors: {error_count}")

        # ダウンロード
        st.subheader("💾 Download")

        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp()
        temp_csv_path = Path(temp_dir, "ocred_results.csv")
        corrected_df[['obj_name', 'col', 'row', 'text', 'corrected', 'status']].to_csv(temp_csv_path, index=False)

        # zip圧縮（時刻付きファイル名）
        zip_file = util_file.zip_now(temp_dir, zip_name="ocred_results")

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

    # 最終結果表示
    st.subheader("✅ Final Results")
    st.dataframe(
        corrected_df[['obj_name', 'col', 'row', 'corrected']],
        use_container_width=True,
        height=300
    )

except FileNotFoundError as e:
    st.error(f"❌ File not found: {str(e)}")
    st.info("💡 Make sure the file path is correct and the file exists")

except pd.errors.EmptyDataError:
    st.error("❌ The CSV file is empty")

except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    with st.expander("🔍 Show detailed error"):
        st.code(traceback.format_exc())
