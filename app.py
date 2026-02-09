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
    layout="wide"
)

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

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🎾 Tennis Backhand Detection")

st.write(
    "Try the demo videos below, or upload your own tennis match clip to detect backhands automatically."
)

# --------------------------------------------------
# Video source selection
# --------------------------------------------------
st.subheader("Choose a video")

source = st.radio(
    "Video source",
    ["Demo video", "Upload my own"],
    disabled=st.session_state.processing
)

video_path = None

if source == "Demo video":
    demo_choice = st.selectbox(
        "Select a demo clip",
        ["sinner.mp4", "djokovic.mp4"],
        disabled=st.session_state.processing
    )

    video_path = os.path.abspath(demo_choice)

    if not os.path.exists(video_path):
        st.error(f"Demo video not found: {demo_choice}")
        video_path = None

else:
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4"],
        disabled=st.session_state.processing
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

# --------------------------------------------------
# Show selected video
# --------------------------------------------------
if video_path:
    st.video(video_path)

# --------------------------------------------------
# Worker thread
# --------------------------------------------------
def worker_fn(video_path, output_dir, log_queue):
    def logger(msg):
        log_queue.put(msg)

    clips = detect_backhands(
        video_path=video_path,
        output_dir=output_dir,
        log_callback=logger
    )

    log_queue.put(("__DONE__", clips))

# --------------------------------------------------
# Start detection
# --------------------------------------------------
if video_path and not st.session_state.processing:
    if st.button("▶️ Run Backhand Detection"):
        st.session_state.processing = True
        st.session_state.clips = []
        st.session_state.logs.clear()

        output_dir = os.path.abspath("data/outputs")
        os.makedirs(output_dir, exist_ok=True)

        thread = threading.Thread(
            target=worker_fn,
            args=(video_path, output_dir, st.session_state.log_queue),
            daemon=True
        )
        thread.start()

# --------------------------------------------------
# Drain log queue
# --------------------------------------------------
while not st.session_state.log_queue.empty():
    item = st.session_state.log_queue.get()

    if isinstance(item, tuple) and item[0] == "__DONE__":
        clips = item[1]

        # wait until files exist
        for _ in range(30):
            if all(os.path.exists(p) for p in clips):
                break
            time.sleep(0.1)

        st.session_state.clips = clips
        st.session_state.processing = False
    else:
        st.session_state.logs.append(item)

# --------------------------------------------------
# Progress logs (kept intentionally)
# --------------------------------------------------
if st.session_state.processing:
    st.info("⏳ Processing video…")

if st.session_state.logs:
    st.subheader("Progress")
    st.code("\n".join(st.session_state.logs[-15:]))

# --------------------------------------------------
# Results
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)")

    for i, clip in enumerate(st.session_state.clips, 1):
        st.subheader(f"🎾 Backhand {i}")
        st.video(clip)

        with open(clip, "rb") as f:
            st.download_button(
                "⬇️ Download clip",
                data=f.read(),
                file_name=os.path.basename(clip),
                mime="video/mp4",
                key=f"download_{i}"
            )

# --------------------------------------------------
# Auto-refresh while processing
# --------------------------------------------------
if st.session_state.processing:
    time.sleep(1)
    st.rerun()
