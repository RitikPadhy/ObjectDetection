import streamlit as st
import cv2
import numpy as np
import tempfile
import os

def mask_custom_diagonal(input_path, output_path, x_top, y_left):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    polygon = np.array([
        [0, height],       # bottom-left
        [width, height],   # bottom-right
        [width, 0],        # top-right (keep right edge intact)
        [x_top, 0],        # top edge intercept (fixed)
        [0, y_left]        # left edge intercept (fixed)
    ], np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(masked_frame)

    cap.release()
    out.release()

st.title("🎞️ Diagonal Cropper with Fixed Line")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
        temp_in.write(uploaded_file.read())
        input_path = temp_in.name

    cap_tmp = cv2.VideoCapture(input_path)
    width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_tmp.release()

    # Fixed intercept values
    x_top = 915
    y_left = 585

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
        output_path = temp_out.name

    st.info("Processing video...")
    mask_custom_diagonal(input_path, output_path, x_top, y_left)
    st.success("Done!")

    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
    st.video(video_bytes)

    with open(output_path, "rb") as f:
        st.download_button("Download Cropped Video", f, file_name="diagonal_cropped.mp4")

    os.remove(input_path)
    os.remove(output_path)