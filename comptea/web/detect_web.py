# Import All the Required Libraries
import streamlit as st
from pathlib import Path
import os
import sys
from ultralytics import YOLO
from PIL import Image

# Set working directory in Win
# import platform
# if platform.system() == "Windows":
#     from tempfile import mkdtemp
#     TMP_DIR = mkdtemp()
#     os.chdir(TMP_DIR)

# Set WD and import custom modules
if '__file__' not in locals():
    WD = os.getcwd()
else:
    WD = Path(__file__).parent


# import util_file
from comptea import detect

# avoid error shown below
#     https://stackoverflow.com/questions/79500227/why-am-i-getting-runtimeerror-no-running-event-loop-and-in-my-vs-code-when-i-am
#     https://discuss.streamlit.io/t/message-error-about-torch/90886/9
import torch
torch.classes.__path__ = [] # set to empty.

# Page config
st.set_page_config(page_title="YOLO Detection", layout="wide")
st.title("Compositional Table Detection with YOLO11")

# Image Config
# DEFAULT_IMAGE = Path(["images/example.jpg", "images/example2.jpg"])
DEFAULT_IMAGE = Path("images/example.jpg")

# Load the YOLO Model
model_path = Path("weights/comptea.pt")
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified path: {model_path}")
    st.error(e)

# SideBar
settings = st.sidebar

with settings:
    #header
    st.header("Model Configurations")
    #Select Confidence Value
    # 30 は 77枚(うち未ラベル46枚)で測って決めた値
    confidence_value = float(st.number_input("Minimal Confidence Value", 5, 100, 30, 5))/100
    # 学習時と同じ大きさで推論する(weights/comptea.pt は imgsz 1280 で学習)
    imgsz = int(st.number_input("Image size", 320, 2048, 1280, 320,
                                help="学習時と同じ値にする"))
    source_image = st.file_uploader("Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp"))

# main panel
col1, col2 = st.columns(2)
with col1:
    if source_image is None:
        source_image = str(DEFAULT_IMAGE)
        st.image(source_image, caption = "Example Image")
    else:
        st.image(source_image, caption = "Uploaded Image")
    uploaded_image = Image.open(source_image)
with col2:
    if st.sidebar.button("Detect Objects"):
        detection_results = model.predict(uploaded_image, conf = confidence_value, imgsz = imgsz)
        result_plotted = detection_results[0].plot()[:,:,::-1]
        st.image(result_plotted, caption = "Detected Image")
        yolo_result_csv = detect.yolo_results2csv(detection_results, source_image, model_path, output_dir=".")
        st.download_button("DOWNLOAD DETECTION RESULT", open(yolo_result_csv, 'r'), yolo_result_csv)
