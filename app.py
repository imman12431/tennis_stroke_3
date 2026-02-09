import os
import time
import tempfile
import threading

# --------------------------------------------------
# Quiet TensorFlow
# --------------------------------------------------
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import streamlit as st
from detector import detect_backhands

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "processing" not in st.session_state:
    st.session_state.processing = False
if "clips" not in st.session_state:
    st.session_state.clips = []
if "worker" not in st.session_state:
    st.session_state.worker = None

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎾 Tennis Backhand Detection")
st.write(
    "Upload a tennis match video and the app will automatically detect "
    "and extract backhand shots."
)

if st.session_state.processing:
    st.info("⏳ Processing video… this may take a minute. Please don’t refresh.")

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a video file (MP4)",
    type=["mp4"],
    disabled=st.session_state.processing
)

# --------------------------------------------------
# Worker thread
# --------------------------------------------------
def worker_fn(video_path, output_dir):
    clips = detect_backhands(
        video_path=video_path,
        output_dir=output_dir,
        log_callback=None  # no logs in production
    )
    st.session_state.clips = clips
    st.session_state.processing = False

# --------------------------------------------------
# Start job
# --------------------------------------------------
if uploaded_file and not st.session_state.processing:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.subheader("Uploaded video")
    st.video(video_path)

    if st.button("▶️ Run Backhand Detection"):
        st.session_state.processing = True
        st.session_state.clips = []

        output_dir = os.path.abspath("data/outputs")
        os.makedirs(output_dir, exist_ok=True)

        st.session_state.worker = threading.Thread(
            target=worker_fn,
            args=(video_path, output_dir),
            daemon=True
        )
        st.session_state.worker.start()

# --------------------------------------------------
# Results
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)")

    for i, clip in enumerate(st.session_state.clips, 1):
        clip = os.path.abspath(clip)

        st.subheader(f"Backhand {i}")
        st.video(clip)
