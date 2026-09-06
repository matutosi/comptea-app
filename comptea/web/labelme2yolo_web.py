# Import All the Required Libraries
import streamlit as st
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Set WD import custom modules
if '__file__' not in locals():
    WD = os.getcwd()
else:
    WD = Path(__file__).parent

from comptea import util_file

# Page config
st.set_page_config(page_title="Labelme to YOLO", layout="wide")
st.title("Labelme to YOLO Converter")

# Create a directory for labelme data
LABELEME_DIR = "labelme_data"
os.makedirs(LABELEME_DIR, exist_ok=True)

def exec_labelme2yolo():
    # execute labelme2yolo
    result = subprocess.run(
        ['labelme2yolo', '--json_dir', LABELEME_DIR, '--val_size', str(val_size)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        st.error(f"labelme2yolo failed: {result.stderr}")
        return
    # zip
    to_be_zipped_dir = Path(LABELEME_DIR, "YOLODataset")
    yolo_zip_file = util_file.zip_now(to_be_zipped_dir, zip_name = "yolo_data")
    # main panel: download
    st.download_button('DOWNLOAD YOLO DATA', open(yolo_zip_file, 'br'), yolo_zip_file)

# Header
st.header("wrapper for labelme2yolo")

# SideBar
settings = st.sidebar

with settings:
    # header
    st.header("Configurations")
    val_size = float(st.number_input("% for validation", 0, 50, 20, 5))/100
    labelme_zip_file = st.file_uploader("Choose a zip file", type=["zip"], accept_multiple_files=False)

# unzip file
if labelme_zip_file:
    zip_path = Path(LABELEME_DIR, labelme_zip_file.name)
    with open(zip_path, "wb") as f:
        f.write(labelme_zip_file.getbuffer())
    shutil.unpack_archive(zip_path, LABELEME_DIR)
    # main panel
    st.button('convert LABELEME to YOLO', on_click=exec_labelme2yolo)
