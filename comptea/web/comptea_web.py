# comptea_web.py - Unified Streamlit app for comptea pipeline
import streamlit as st
from pathlib import Path
import os
import sys
import glob
import re
import shutil
import subprocess
import tempfile
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from multiprocessing import freeze_support

# avoid error
#     https://stackoverflow.com/questions/79500227/why-am-i-getting-runtimeerror-no-running-event-loop-and-in-my-vs-code-when-i-am
#     https://discuss.streamlit.io/t/message-error-about-torch/90886/9
import torch
torch.classes.__path__ = []

# Set WD import custom modules
if '__file__' not in locals():
    WD = os.getcwd()
else:
    WD = Path(__file__).parent


from comptea import util_file
from comptea import detect
from comptea import locate
from comptea import draw_rect
from comptea import ocr
from comptea import correct_text
from comptea import comp_table
from comptea import plot_table

# Windowsでマルチプロセスを使用するために必要
freeze_support()

# Page config
st.set_page_config(page_title="comptea", layout="wide")
st.title("comptea - Composition Table to Data Easy")

# Sidebar: page selection
page = st.sidebar.radio(
    "Select page",
    [
        "1. Labelme2YOLO",
        "2. Preprocess Image",
        "3. Train",
        "4. Detect",
        "5. Locate",
        "6. OCR",
        "7. Comp Table",
    ],
)

# ============================================================
# 1. Labelme2YOLO
# ============================================================
def page_labelme2yolo():
    st.header("Labelme to YOLO Converter")

    # Create a directory for labelme data
    LABELEME_DIR = "labelme_data"
    os.makedirs(LABELEME_DIR, exist_ok=True)

    with st.sidebar:
        st.header("Configurations")
        val_size = float(st.number_input("% for validation", 0, 50, 20, 5)) / 100
        labelme_zip_file = st.file_uploader("Choose a zip file", type=["zip"], accept_multiple_files=False)

    def exec_labelme2yolo():
        result = subprocess.run(
            ['labelme2yolo', '--json_dir', LABELEME_DIR, '--val_size', str(val_size)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(f"labelme2yolo failed: {result.stderr}")
            return
        to_be_zipped_dir = Path(LABELEME_DIR, "YOLODataset")
        yolo_zip_file = util_file.zip_now(to_be_zipped_dir, zip_name="yolo_data")
        # Save to session_state for Train tab
        with open(yolo_zip_file, 'rb') as f:
            st.session_state["yolo_zip"] = f.read()
            st.session_state["yolo_zip_name"] = Path(yolo_zip_file).name
        st.download_button('DOWNLOAD YOLO DATA', open(yolo_zip_file, 'br'), yolo_zip_file)
        st.success("Results saved. You can use them in the Train tab.")

    if labelme_zip_file:
        zip_path = Path(LABELEME_DIR, labelme_zip_file.name)
        with open(zip_path, "wb") as f:
            f.write(labelme_zip_file.getbuffer())
        shutil.unpack_archive(zip_path, LABELEME_DIR)
        st.button('convert LABELME to YOLO', on_click=exec_labelme2yolo)


# ============================================================
# 2. Preprocess Image
# ============================================================
def page_preprocess():
    st.header("Image Preprocessing")

    DEFAULT_IMAGE = Path("images/example.jpg")

    with st.sidebar:
        st.header("Configurations")
        source_image = st.file_uploader("Choose an Image...", type=("jpg", "png", "jpeg", "bmp", "webp"))
        deskew = st.checkbox("deskew (correct skew)", value=False)
        gausian_blur_ksize = st.slider("gausian_blur kanel size", 1, 11, value=5, step=2)
        median_blur_ksize = st.slider("median_blur kanel size", 1, 11, value=1, step=2)
        binarize = st.checkbox("binarize", value=True)
        equalize = st.checkbox("equalize", value=True)
        resize_ratio = st.slider("resize", 0.1, 2.0, value=1.0, step=0.1)

    # Load image
    if source_image is None:
        image_original = cv2.imread(str(DEFAULT_IMAGE))
    else:
        file_bytes = np.frombuffer(source_image.read(), dtype=np.uint8)
        image_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = image_original.copy()

    # Deskew
    if deskew:
        from comptea.preprocess_image import correct_skew
        image_original = correct_skew(image_original)
        image = image_original.copy()

    # Preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gausian_blur_ksize > 1:
        image = cv2.GaussianBlur(image, (gausian_blur_ksize, gausian_blur_ksize), 0)

    if median_blur_ksize > 1:
        image = cv2.medianBlur(image, median_blur_ksize)

    if binarize:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if equalize:
        image = cv2.equalizeHist(image)

    if resize_ratio != 1:
        image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        if source_image is None:
            st.image(image_original, caption="Example Image")
        else:
            st.image(image_original, caption="Uploaded Image")

    with col2:
        st.image(image, caption="Converted Image")

    # Download & save to session_state
    is_success, buffer = cv2.imencode(".png", image)
    if is_success:
        image_bytes = buffer.tobytes()
        st.download_button("Download processed image", image_bytes,
                           file_name="preprocessed.png", mime="image/png")
        st.session_state["preprocessed_image"] = image_bytes


# ============================================================
# 3. Train
# ============================================================
def page_train():
    st.header("YOLO Training")

    TMP_DIR = tempfile.mkdtemp()
    YOLO_DIR = Path(TMP_DIR, "yolo_data")
    os.makedirs(YOLO_DIR, exist_ok=True)
    TRAINED_DATA_DIR = Path(YOLO_DIR, "runs")
    os.makedirs(TRAINED_DATA_DIR, exist_ok=True)

    is_available_cuda = torch.cuda.is_available()

    with st.sidebar:
        st.header("Training Configurations")
        # Use results from Labelme2YOLO tab
        use_pipeline = st.checkbox(
            "Use results from Labelme2YOLO tab",
            value="yolo_zip" in st.session_state,
            disabled="yolo_zip" not in st.session_state
        )
        if use_pipeline and "yolo_zip" in st.session_state:
            st.info(f"Using: {st.session_state.get('yolo_zip_name', 'yolo_data.zip')}")
            yolo_zip_file = None
        else:
            yolo_zip_file = st.file_uploader("Choose a zip file", type=["zip"], accept_multiple_files=False)

    def modify_yaml():
        yaml_files = glob.glob(f'{YOLO_DIR}/*.yaml')
        if len(yaml_files) != 1:
            raise Exception(f'zip file should contain ONE yaml file, not {len(yaml_files)}')
        data = yaml_files[0]
        with open(data, 'r', encoding='utf-8') as f:
            yaml_lines = f.read()
        yolo_dir = str(YOLO_DIR).replace("\\", "/")
        replaced_1 = re.sub("path: .+\\n", f'path: {yolo_dir}\\n', yaml_lines)
        replaced_2 = re.sub("\\ntrain: images/train\\n", "\\ntrain: ./images/train\\n", replaced_1)
        replaced_3 = re.sub("\\nval: images/val\\n", "\\nval: ./images/val\\n", replaced_2)
        with open(data, mode="w", encoding='utf-8') as f:
            f.write(replaced_3)
        return data

    def zip_trained_data():
        trained_zip_file = util_file.zip_now(TRAINED_DATA_DIR, zip_name="trained_data")
        moved_path = shutil.move(trained_zip_file, TMP_DIR)
        st.download_button('DOWNLOAD TRAINED DATA', open(moved_path, 'br'), trained_zip_file)

    def train_yolo():
        if __name__ == "__main__":
            subprocess.run(
                ['yolo', 'settings', f'runs_dir={TRAINED_DATA_DIR}'],
                capture_output=True, text=True
            )
            model = YOLO("yolo11n.pt")
            model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, batch=batch_size, project=str(TRAINED_DATA_DIR))
        zip_trained_data()

    # Load zip data
    zip_loaded = False
    if use_pipeline and "yolo_zip" in st.session_state:
        zip_path = Path(TMP_DIR, st.session_state.get("yolo_zip_name", "yolo_data.zip"))
        with open(zip_path, "wb") as f:
            f.write(st.session_state["yolo_zip"])
        shutil.unpack_archive(zip_path, YOLO_DIR)
        data = modify_yaml()
        zip_loaded = True
    elif yolo_zip_file:
        with open(Path(TMP_DIR, yolo_zip_file.name), "wb") as f:
            f.write(yolo_zip_file.getbuffer())
        shutil.unpack_archive(Path(TMP_DIR, yolo_zip_file.name), YOLO_DIR)
        data = modify_yaml()
        zip_loaded = True

    if zip_loaded:
        with st.sidebar:
            if is_available_cuda:
                device = st.selectbox("Device", ["cuda", "cpu"])
            else:
                device = "cpu"
            imgsz = int(st.number_input("Image size", 120, 1000, 120, 120))
            epochs = int(st.number_input("Epochs", 2, 100, 2))
            batch_size = int(st.number_input("Batch size", 1, 64, 16))
        st.button('Train data', on_click=train_yolo)


# ============================================================
# 4. Detect
# ============================================================
def page_detect():
    st.header("Compositional Table Detection with YOLO11")

    DEFAULT_IMAGE = Path("images/example.jpg")

    # Load the YOLO Model
    model_path = Path("weights/comptea.pt")
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(e)
        return

    with st.sidebar:
        st.header("Model Configurations")
        # 30 は 77枚(うち未ラベル46枚)で測って決めた値．
        # 60 だと行を12%取りこぼすのに，境界が文字に乗る割合は改善しなかった
        confidence_value = float(st.number_input(
            "Minimal Confidence Value", 5, 100, 30, 5,
            help="行・領域などの閾値．学習外を含む77枚で決めた値")) / 100
        # colは他のクラスより検出の信頼度が低く出るため，別に閾値を持たせる
        # 2026-08-24に30から20へ下げた．手元の全88枚で測ったところ，
        # 30ではcolが1本も取れない段が5つあり，その段の組成部が丸ごと空になっていた．
        # 20にすると1つに減る(kinki_005のみ)．10まで下げても減らず，
        # 余分な列が増える(ラベル付き33枚でcolのprecisionが0.97→0.83)．
        # 測り直すには yolo/eval_grid.py と yolo/scan_blocks.py を使う
        confidence_col = float(st.number_input(
            "Minimal Confidence Value (col)", 5, 100, 20, 5,
            help="col は検出の信頼度が低く出るため別扱い．"
                 "手元の全88枚で，段が丸ごと落ちる件数を数えて決めた値"
        )) / 100
        # 学習時と同じ大きさで推論する．ずれると成績が落ちる
        # (imgsz 1280 のモデルを 640 で推論すると，行の一致が 7/7 から 3/7 に落ちた)
        imgsz = int(st.number_input(
            "Image size", 320, 2048, 1280, 320,
            help="学習時と同じ値にする．weights/comptea.pt は 1280 で学習"))
        source_image = st.file_uploader("Choose an Image...", type=("jpg", "png", "jpeg", "bmp", "webp"))

    # main panel
    col1, col2 = st.columns(2)
    with col1:
        if source_image is None:
            source_image_path = str(DEFAULT_IMAGE)
            st.image(source_image_path, caption="Example Image")
        else:
            source_image_path = source_image
            st.image(source_image, caption="Uploaded Image")
        uploaded_image = Image.open(source_image_path)

    with col2:
        if st.sidebar.button("Detect Objects"):
            # 低い方の閾値で推論し，クラスごとの閾値で絞り込む
            detection_results = model.predict(
                uploaded_image, conf=min(confidence_value, confidence_col), imgsz=imgsz)
            detection_results = detect.filter_by_conf(
                detection_results, confidence_value, {"col": confidence_col})
            result_plotted = detection_results[0].plot()[:, :, ::-1]
            st.image(result_plotted, caption="Detected Image")
            yolo_result_csv = detect.yolo_results2csv(detection_results, source_image_path, model_path, output_dir=".")
            # Save to session_state for Locate tab
            detection_df = pd.read_csv(yolo_result_csv)
            st.session_state["detection_df"] = detection_df
            st.session_state["source_image_path"] = source_image_path
            st.download_button("DOWNLOAD DETECTION RESULT", open(yolo_result_csv, 'r'), yolo_result_csv)
            st.success("Results saved. You can use them in the Locate tab.")


# ============================================================
# 5. Locate
# ============================================================
def page_locate():
    st.header("Table Cell Location Finder")

    with st.sidebar:
        st.header("Configurations")

        st.subheader("Input File")
        use_pipeline = st.checkbox(
            "Use results from Detect tab",
            value="detection_df" in st.session_state,
            disabled="detection_df" not in st.session_state
        )

        use_example = False
        uploaded_file = None
        if use_pipeline and "detection_df" in st.session_state:
            st.info("Using detection results from Detect tab")
        else:
            use_example = st.checkbox("Use example", value=True, help="Use detection_results.csv as example")
            if not use_example:
                uploaded_file = st.file_uploader(
                    "Upload detection results CSV",
                    type=["csv"],
                    help="Upload a CSV file from detect_web.py"
                )

        st.subheader("Parameters")
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

    # Main
    try:
        # Load data
        if use_pipeline and "detection_df" in st.session_state:
            df = st.session_state["detection_df"]
        elif use_example:
            path_detection = "detection_results.csv"
            if not os.path.exists(path_detection):
                st.error(f"File not found: {path_detection}")
                st.info("Please run Detect first to generate detection results.")
                st.stop()
            df = pd.read_csv(path_detection)
        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a detection results CSV file or run Detect first")
            st.stop()

        # Validate
        required_columns = ["obj_name", "x1", "y1", "x2", "y2", "confidence", "source_image", "model"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        if len(df) == 0:
            st.warning("No detections found in the CSV file")
            st.stop()

        # Locate
        # 境界の調整に画像を使うため，Detectページで開いたものを渡す
        # (アップロードされた画像はdfのsource_imageからは開けない)
        with st.spinner("Calculating grid locations..."):
            df_located = locate.locate_items(
                df, threth_col=threth_col, threth_row=threth_row,
                image=st.session_state.get("source_image_path"))

        # 位置を決められなかったものの理由を伝える(黙って空を返さない)
        for w in df_located.attrs.get("warnings", []):
            st.warning(w)

        if len(df_located) == 0:
            st.warning("No items located (all detections filtered out)")
            st.info("Try adjusting the overlap threshold parameter")
            st.stop()

        st.success(f"Located {len(df_located)} items")

        # Save to session_state for OCR tab
        st.session_state["located_df"] = df_located

        # Display
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Located Results")
            st.dataframe(df_located, use_container_width=True, height=400)

        with col2:
            st.subheader("Statistics")
            st.metric("Total Items", len(df_located))

            obj_counts = df_located['obj_name'].value_counts()
            st.write("**Items by type:**")
            for obj_name, count in obj_counts.items():
                st.write(f"- {obj_name}: {count}")

            st.subheader("Download")
            temp_dir = tempfile.mkdtemp()
            temp_csv_path = Path(temp_dir, "located_results.csv")
            df_located.to_csv(temp_csv_path, index=False)

            zip_file = util_file.zip_now(temp_dir, zip_name="located_results")
            with open(zip_file, "rb") as f:
                st.download_button(
                    label="Download as ZIP",
                    data=f.read(),
                    file_name=Path(zip_file).name,
                    mime="application/zip"
                )
            shutil.rmtree(temp_dir)
            if Path(zip_file).exists():
                Path(zip_file).unlink()

        # Visualization
        st.subheader("Visualization")
        with st.spinner("Drawing rectangles..."):
            image = draw_rect.draw_rects_df(df_located)

        if image is not None:
            st.image(image, caption="Located Items with Grid Coordinates", use_container_width=True)
            if (df_located.get('note', pd.Series(dtype=str)).fillna('') != '').any():
                st.caption(
                    "太い枠は位置決めで気になったセル: "
                    "🟡 gold = 検出漏れを内挿 / 🟠 orange = 文字を避けてずらした / "
                    "🔴 red = ずらしても文字に重なったまま")
        else:
            st.error("Failed to draw rectangles")

    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())


# ============================================================
# 6. OCR
# ============================================================
def page_ocr():
    st.header("OCR & Text Correction")

    with st.sidebar:
        st.header("Configurations")

        st.subheader("Input File")
        use_pipeline = st.checkbox(
            "Use results from Locate tab",
            value="located_df" in st.session_state,
            disabled="located_df" not in st.session_state
        )

        use_example = False
        uploaded_file = None
        if use_pipeline and "located_df" in st.session_state:
            st.info("Using located results from Locate tab")
        else:
            use_example = st.checkbox("Use example", value=True, help="Use located.csv as example")
            if not use_example:
                uploaded_file = st.file_uploader(
                    "Upload located results CSV",
                    type=["csv"],
                    help="Upload a CSV file from locate_web.py"
                )

    # Main
    try:
        # Load data
        if use_pipeline and "located_df" in st.session_state:
            df_all = st.session_state["located_df"]
        elif use_example:
            path_located = "located.csv"
            if not os.path.exists(path_located):
                st.error(f"File not found: {path_located}")
                st.info("Please run Locate first to generate located results.")
                st.stop()
            df_all = pd.read_csv(path_located)
        elif uploaded_file:
            df_all = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a located results CSV file or run Locate first")
            st.stop()

        # Validate
        required_columns = ["source_image", "x1", "y1", "x2", "y2", "obj_name"]
        missing_columns = [col for col in required_columns if col not in df_all.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        if len(df_all) == 0:
            st.warning("No data found in the CSV file")
            st.stop()

        # OCR
        st.subheader("Running OCR")
        df_image = df_all.groupby('source_image')

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

        result_df = pd.concat(all_results, ignore_index=True)
        status_text.empty()
        progress_bar.empty()

        if len(result_df) == 0:
            st.warning("No OCR results obtained")
            st.stop()

        st.success(f"OCR completed: {len(result_df)} items extracted")

        # Grid coordinates
        with st.spinner("Assigning grid coordinates..."):
            # 段が複数あるページでは，行は段をまたいで通し，列は段ごとに振る
            result_df = locate.number_cells(result_df)

        # Text correction
        with st.spinner("Correcting text..."):
            for i, row in result_df.iterrows():
                # 振り分けは correct_text に1つだけ置く(スキルからも同じものを呼ぶ)．
                # 文章として読む領域(header / once_species)は corrected に
                # text をそのまま入れる．None にすると plot_table / comp_table が
                # 読む列が空になり，通しでは何も出なくなる
                corrected = correct_text.correct_cell(row['obj_name'], row['text'])

                result_df.loc[i, 'corrected'] = corrected['corrected']
                result_df.loc[i, 'status'] = corrected['status']

        st.success("Text correction completed")

        # Display
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Edit OCR Results")
            st.info("You can edit the 'corrected' column directly in the table below")

            # noteは位置決めで気になった点(内挿・スナップ・文字に重なったまま)
            edit_columns = ['obj_name', 'col', 'row', 'img_base64', 'text', 'corrected', 'status']
            if 'note' in result_df.columns:
                edit_columns.append('note')
            corrected_df = st.data_editor(
                result_df[edit_columns],
                column_config={
                    "img_base64": st.column_config.ImageColumn("original image", width="medium"),
                    "obj_name": st.column_config.TextColumn("Type", width="small"),
                    "col": st.column_config.NumberColumn("Col", width="small"),
                    "row": st.column_config.NumberColumn("Row", width="small"),
                    "text": st.column_config.TextColumn("OCR Text", width="medium"),
                    "corrected": st.column_config.TextColumn("Corrected", width="medium"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "note": st.column_config.TextColumn(
                        "Note", width="small",
                        help="位置決めで気になった点．interpolated=検出漏れを内挿"
                             " / snapped=文字を避けてずらした / on_text=文字に重なったまま"),
                },
                height=600,
                use_container_width=True
            )

            # Comp Tableページで使う．編集した内容を反映して渡す
            edited_df = result_df.copy()
            edited_df['corrected'] = corrected_df['corrected']
            st.session_state["ocr_df"] = edited_df

        with col2:
            st.subheader("Statistics")
            st.metric("Total Items", len(result_df))

            st.write("**Items by type:**")
            obj_counts = result_df['obj_name'].value_counts()
            for obj_name, count in obj_counts.items():
                st.write(f"- {obj_name}: {count}")

            st.write("**Correction status:**")
            if 'status' in result_df.columns:
                status_counts = result_df['status'].value_counts()
                for status, count in status_counts.items():
                    st.write(f"- {status}: {count}")

            st.write("**Corrections by type:**")
            for obj_name in result_df['obj_name'].unique():
                obj_df = result_df[result_df['obj_name'] == obj_name]
                corrected_count = len(obj_df[obj_df['text'] != obj_df['corrected']])
                error_count = len(obj_df[obj_df['status'].isna() | (obj_df['status'] == 'error')])
                st.write(f"- **{obj_name}**")
                st.write(f"  - Corrected: {corrected_count}")
                st.write(f"  - Errors: {error_count}")

            # Download
            st.subheader("Download")
            temp_dir = tempfile.mkdtemp()
            temp_csv_path = Path(temp_dir, "ocred_results.csv")
            corrected_df[['obj_name', 'col', 'row', 'text', 'corrected', 'status']].to_csv(temp_csv_path, index=False)

            zip_file = util_file.zip_now(temp_dir, zip_name="ocred_results")
            with open(zip_file, "rb") as f:
                st.download_button(
                    label="Download as ZIP",
                    data=f.read(),
                    file_name=Path(zip_file).name,
                    mime="application/zip"
                )
            shutil.rmtree(temp_dir)
            if Path(zip_file).exists():
                Path(zip_file).unlink()

        # Final results
        st.subheader("Final Results")
        st.dataframe(
            corrected_df[['obj_name', 'col', 'row', 'corrected']],
            use_container_width=True,
            height=300
        )

    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())


# ============================================================
# 7. Comp Table
# ============================================================
def page_comp_table():
    st.header("Composition Table")
    st.caption("OCR結果を縦持ち(1行 = 1地点 × 1種)に組み立てる")

    with st.sidebar:
        st.header("Configurations")

        st.subheader("Input File")
        use_pipeline = st.checkbox(
            "Use results from OCR tab",
            value="ocr_df" in st.session_state,
            disabled="ocr_df" not in st.session_state
        )

        uploaded_file = None
        if use_pipeline and "ocr_df" in st.session_state:
            st.info("Using OCR results from OCR tab")
        else:
            uploaded_file = st.file_uploader(
                "Upload OCR results CSV",
                type=["csv"],
                help="obj_name, col, row, corrected を持つCSV"
            )

        st.subheader("Parameters")
        keep_absent = st.checkbox(
            "Keep absent cells",
            value=False,
            help="非出現('・')のセルも行として残す"
        )

    # Main
    try:
        if use_pipeline and "ocr_df" in st.session_state:
            df = st.session_state["ocr_df"]
        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload an OCR results CSV file or run OCR first")
            st.stop()

        required_columns = ["obj_name", "col", "row"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        with st.spinner("Building composition table..."):
            df_long = comp_table.comp_table(df, keep_absent=keep_absent)

        # 組み立てに困ったことを伝える(黙って空を返さない)
        for w in df_long.attrs.get("warnings", []):
            st.warning(w)

        if len(df_long) == 0:
            st.warning("No rows built from the OCR results")
            st.stop()

        st.success(f"Built {len(df_long)} rows")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Long Format")
            st.dataframe(df_long, use_container_width=True, height=500)

        with col2:
            st.subheader("Statistics")
            st.metric("Rows", len(df_long))
            st.metric("Plots", df_long["plot"].nunique())
            st.metric("Species rows", df_long["row_no"].nunique())

            st.write("**Status:**")
            for status, count in df_long["status"].value_counts().items():
                st.write(f"- {status}: {count}")

            n_check = int((df_long["status"] == "Need Check").sum())
            if n_check:
                st.warning(f"{n_check} 件は被度・群度として読めない．要確認")

            st.subheader("Download")
            temp_dir = tempfile.mkdtemp()
            df_long.to_csv(Path(temp_dir, "comp_table_long.csv"), index=False)
            comp_table.to_wide(df_long).to_csv(Path(temp_dir, "comp_table_wide.csv"), index=False)

            # 表頭は粒度が違う(1行 = 1地点)ので別表として一緒に入れる
            df_plot = plot_table.plot_table(df)
            if len(df_plot):
                df_plot.to_csv(Path(temp_dir, "plot_table.csv"), index=False)

            zip_file = util_file.zip_now(temp_dir, zip_name="comp_table")
            with open(zip_file, "rb") as f:
                st.download_button(
                    label="Download as ZIP",
                    data=f.read(),
                    file_name=Path(zip_file).name,
                    mime="application/zip"
                )
            shutil.rmtree(temp_dir)
            if Path(zip_file).exists():
                Path(zip_file).unlink()

        # 表頭から作った地点の表．1行 = 1地点で，縦持ちとは粒度が違う
        st.subheader("Plot Table (表頭)")
        st.caption("1行 = 1地点．調査番号・調査日・標高などの属性")
        for w in df_plot.attrs.get("warnings", []):
            st.warning(w)
        if len(df_plot):
            st.dataframe(df_plot, use_container_width=True)
        else:
            st.info("表頭のセルが無いか，項目名を判別できなかった")

        # 確認用．出力の正は縦持ちの方
        with st.expander("Wide format (確認用)"):
            st.dataframe(comp_table.to_wide(df_long), use_container_width=True)

    except ValueError as e:
        st.error(str(e))
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())


# ============================================================
# Page routing
# ============================================================
if page == "1. Labelme2YOLO":
    page_labelme2yolo()
elif page == "2. Preprocess Image":
    page_preprocess()
elif page == "3. Train":
    page_train()
elif page == "4. Detect":
    page_detect()
elif page == "5. Locate":
    page_locate()
elif page == "6. OCR":
    page_ocr()
elif page == "7. Comp Table":
    page_comp_table()
