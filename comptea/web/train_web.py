# https://discuss.streamlit.io/t/redirecting-logger-output-to-a-streamlit-widget/61070
# https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602/28
import glob
import os
import re
import sys
import shutil
import subprocess
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from multiprocessing import freeze_support

from comptea import util_file
# from progress import st_capture_stderr

# Page config
st.set_page_config(page_title="YOLO Training", layout="wide")
st.title("YOLO Training")

# Create a directory for yolo data
from tempfile import TemporaryDirectory
TMP_DIR = TemporaryDirectory().name
util_file.renew_dir(TMP_DIR)
YOLO_DIR = Path(TMP_DIR, "yolo_data")
os.makedirs(YOLO_DIR, exist_ok=True)
TRAINED_DATA_DIR = Path(YOLO_DIR, "runs")
os.makedirs(TRAINED_DATA_DIR, exist_ok=True)

# avoid error shown below
#     https://stackoverflow.com/questions/79500227/why-am-i-getting-runtimeerror-no-running-event-loop-and-in-my-vs-code-when-i-am
#     https://discuss.streamlit.io/t/message-error-about-torch/90886/9
import torch
torch.classes.__path__ = [] # set to empty.

# availability of cuda
is_available_cuda = torch.cuda.is_available()

# Windowsでマルチプロセスを使用するために必要
freeze_support()

# SideBar
settings = st.sidebar

with settings:
    # header
    st.header("Taining Configurations")
    yolo_zip_file = st.file_uploader("Choose a zip file", type=["zip"], accept_multiple_files=False)

def modify_yaml():
    yaml = glob.glob(f'{YOLO_DIR}/*.yaml')
    if len(yaml) != 1:
        raise Exception(f'zip file should contain ONE yaml file, not {len(yaml)}')
    data = yaml[0]
    with open(data, 'r', encoding='utf-8') as f:
        yaml_lines = f.read()
    yolo_dir = str(YOLO_DIR).replace("\\","/")
    replaced_1 = re.sub("path: .+\\n"              , f'path: {yolo_dir}\\n'       , yaml_lines) # "^path: foo/bar"      -> "path: {yolo_dir}"}"
    replaced_2 = re.sub("\\ntrain: images/train\\n", "\\ntrain: ./images/train\\n", replaced_1) # "train: images/train" -> "train: ./images/train"
    replaced_3 = re.sub("\\nval: images/val\\n"    , "\\nval: ./images/val\\n"    , replaced_2) # "val: images/val"     -> "val: ./images/val"
    with open(data, mode="w", encoding='utf-8') as f:
        f.write(replaced_3)
    return(data)

def zip_trained_data():
    to_be_zipped_dir = TRAINED_DATA_DIR
    # to_be_zipped_dir = Path(glob.glob('../**/runs/detect/', recursive=True)[0]).parent # search parent directory just to be on the safe side
    trained_zip_file = util_file.zip_now(to_be_zipped_dir, zip_name = "trained_data")
    moved_path = shutil.move(trained_zip_file, TMP_DIR)
    st.download_button('DOWNLOAD TRAINED DATA', open(moved_path, 'br'), trained_zip_file)

def train_yolo():
    if __name__ == "__main__":
        subprocess.run(
            ['yolo', 'settings', f'runs_dir={TRAINED_DATA_DIR}'],
            capture_output=True, text=True
        )
        model = YOLO("yolo11n.pt") # load a pretrained model (recommended for training)
        model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, batch=batch_size, project=str(TRAINED_DATA_DIR))
    # util_file.tree(TMP_DIR)      # for debug
    # util_file.tree("/mount/src") # for debug
    zip_trained_data()

# unzip yolo file
if yolo_zip_file:
    with open(Path(TMP_DIR, yolo_zip_file.name), "wb") as f:
        f.write(yolo_zip_file.getbuffer())
        shutil.unpack_archive(Path(TMP_DIR, yolo_zip_file.name), YOLO_DIR)
        data = modify_yaml()
    print(f'yaml: {data}')
    with settings: # side panel
        if is_available_cuda:
            device = st.selectbox("Device", ["cuda", "cpu"])
        else:
            device = "cpu"
        imgsz = int(st.number_input("Image size", 120, 1000, 120, 120)) # setting for test
        epochs = int(st.number_input("Epochs", 2, 100, 2))              # setting for test
        # imgsz = int(st.number_input("Image size", 400, 1000, 640, 120)) # image size (pixels)
        # epochs = int(st.number_input("Epochs", 10, 100, 10))            # number of epochs
        batch_size = int(st.number_input("Batch size", 1, 64, 16))      # batch size
    # main panel
    st.button('Train data', on_click=train_yolo)
