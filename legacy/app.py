import os
import time
import tempfile
import threading
import queue

import streamlit as st
from detector import detect_backhands

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide",
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEMO_VIDEOS = {
    "Jannik Sinner": os.path.join(BASE_DIR, "sinner.mp4"),
    "Novak Djokovic": os.path.join(BASE_DIR, "djokovic.mp4"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "processing" not in st.session_state:
    st.session_state.processing = False
if "clips" not in st.session_state:
    st.session_state.clips = []
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "worker" not in st.session_state:
    st.session_state.worker = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🎾 Tennis Backhand Detection Demo")
st.markdown("""
## Project Overview

This project detects and extracts tennis backhand shots from match videos using a **frame-wise, pose-based classification pipeline with multi-stage filtering**.

Each video is processed frame by frame using **MediaPipe Pose Landmarker (lite model)** to extract 33 body keypoints. From these keypoints, a **hand-engineered, normalized skeletal feature vector** is constructed using joint positions relative to the mid-hip and scaled by shoulder width, with landmark visibility included as additional features.

The resulting feature vectors are classified using a trained **TensorFlow/Keras neural network**, followed by a second **binary rejector model** that filters false positives. High-confidence backhand detections trigger a **cooldown window** to prevent duplicate detections of the same stroke.

For performance, the system uses **multi-threaded frame decoding** and performs detection in a **first pass**, followed by a **second pass** that cuts short MP4 clips around each detected backhand using **OpenCV and FFmpeg**.

Users can test the pipeline using **preloaded professional match footage** or upload their own videos, with each detected backhand exported as a **downloadable clip**.
""")

if st.session_state.processing:
    st.info("⏳ Processing video… please wait.")

# --------------------------------------------------
# VIDEO SELECTION (always visible)
# --------------------------------------------------
st.subheader("1️⃣ Choose a video")

video_source = st.radio(
    "Video source",
    ["Use demo video", "Upload your own"],
    disabled=st.session_state.processing,
)

# -------------------------
# Demo videos
# -------------------------
if video_source == "Use demo video":
    demo_choice = st.radio(
        "Choose a demo clip",
        list(DEMO_VIDEOS.keys()),
        disabled=st.session_state.processing,
    )

    demo_path = DEMO_VIDEOS[demo_choice]

    if os.path.exists(demo_path):
        st.session_state.video_path = demo_path
        col1, col2 = st.columns([2, 3])
        with col1:
            st.video(demo_path)
    else:
        st.error(f"Demo video not found: {demo_path}")

# -------------------------
# Upload
# -------------------------
else:
    uploaded_file = st.file_uploader(
        "Upload a tennis video (MP4)",
        type=["mp4"],
        disabled=st.session_state.processing,
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.video_path = tmp.name

        col1, col2 = st.columns([2, 3])
        with col1:
            st.video(st.session_state.video_path)

# --------------------------------------------------
# Worker thread
# --------------------------------------------------
def worker_fn(video_path, output_dir, log_queue):
    def logger(msg):
        log_queue.put(msg)

    clips = detect_backhands(
        video_path=video_path,
        output_dir=output_dir,
        log_callback=logger,
    )

    log_queue.put(("__DONE__", clips))

# --------------------------------------------------
# Run detection
# --------------------------------------------------
st.subheader("2️⃣ Run detection")

if st.session_state.video_path and not st.session_state.processing:
    if st.button("▶️ Run Backhand Detection"):
        st.session_state.processing = True
        st.session_state.clips = []
        st.session_state.logs.clear()

        st.session_state.worker = threading.Thread(
            target=worker_fn,
            args=(
                st.session_state.video_path,
                OUTPUT_DIR,
                st.session_state.log_queue,
            ),
            daemon=True,
        )
        st.session_state.worker.start()

# --------------------------------------------------
# Drain log queue (EVERY rerun)
# --------------------------------------------------
while not st.session_state.log_queue.empty():
    item = st.session_state.log_queue.get()

    if isinstance(item, tuple) and item[0] == "__DONE__":
        clips = item[1]

        # Wait until files exist on disk
        for _ in range(30):
            if all(os.path.exists(p) for p in clips):
                break
            time.sleep(0.1)

        st.session_state.clips = clips
        st.session_state.processing = False
    else:
        st.session_state.logs.append(item)

# --------------------------------------------------
# Progress logs (clean)
# --------------------------------------------------
if st.session_state.logs:
    st.subheader("Progress")
    st.code("\n".join(st.session_state.logs[-15:]))

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.subheader("✅ Detected Backhands")

    for i, clip in enumerate(st.session_state.clips, 1):
        if not os.path.exists(clip):
            continue

        st.markdown(f"### 🎾 Backhand {i}")

        col1, col2 = st.columns([2, 3])
        with col1:
            st.video(clip)

        with open(clip, "rb") as f:
            st.download_button(
                label="⬇️ Download clip",
                data=f.read(),
                file_name=os.path.basename(clip),
                mime="video/mp4",
                key=f"download_{i}",
            )

# --------------------------------------------------
# Force rerun while processing
# --------------------------------------------------
if st.session_state.processing:
    time.sleep(1)
    st.rerun()
