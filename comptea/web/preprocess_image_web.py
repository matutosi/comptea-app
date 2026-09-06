#Import All the Required Libraries
import streamlit as st
from pathlib import Path
import os
import sys

import cv2
import numpy as np

# Set WD import custom modules
if '__file__' not in locals():
    WD = os.getcwd()
else:
    WD = Path(__file__).parent

# import util_file

# Page config
st.set_page_config(page_title="Image Preprocessing", layout="wide")
st.title("Image Preprocessing")

# Image Config
DEFAULT_IMAGE = Path("images/example.jpg")

# SideBar
settings = st.sidebar

with settings:
    # header
    st.header("Configurations")
    source_image = st.file_uploader("Choose an Image...", type=("jpg", "png", "jpeg", "bmp", "webp"))
    deskew = st.checkbox("deskew (correct skew)", value=False)
    # convert_to_gray = st.checkbox("convert to gray", value=True)
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
    #  creverse black-white: v2.THRESH_BINARY -> v2.THRESH_BINARY_INV
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

# Download
is_success, buffer = cv2.imencode(".png", image)
if is_success:
    st.download_button("Download processed image", buffer.tobytes(),
                       file_name="preprocessed.png", mime="image/png")
