import streamlit as st

st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

import sys


st.write("Python version:", sys.version)


import cv2


st.write("OpenCV version:", cv2.__version__)
st.write("OpenCV path:", cv2.__file__)



import os
import tempfile
from detector import detect_backhands

st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

st.title("🎾 Tennis Backhand Detection Demo")
st.markdown(
    "Upload a tennis match video. The model will automatically "
    "detect and extract **backhand strokes**."
)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Settings")

skel_threshold = st.sidebar.slider(
    "Skeleton confidence threshold",
    0.5, 0.99, 0.85
)

reject_threshold = st.sidebar.slider(
    "Rejector threshold",
    0.0, 1.0, 0.5
)

# -----------------------
# File upload
# -----------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("▶️ Run Backhand Detection"):
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)

        log_area = st.empty()
        logs = []

        def streamlit_logger(msg):
            logs.append(msg)
            log_area.code("\n".join(logs[-12:]))

        with st.spinner("Analyzing video..."):
            clips = detect_backhands(
                video_path=video_path,
                output_dir=output_dir,
                log_callback=streamlit_logger
            )

        st.success(f"Detected {len(clips)} backhand(s)!")

        for clip in clips:
            st.video(clip)
