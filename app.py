import os
import time
import tempfile
import threading
import queue

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
if "logs" not in st.session_state:
    st.session_state.logs = []
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
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
    st.info("⏳ Processing video… please don’t refresh the page.")

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
def worker_fn(video_path, output_dir, log_queue):
    def logger(msg):
        # Keep messages short & human-readable
        log_queue.put(msg)

    clips = detect_backhands(
        video_path=video_path,
        output_dir=output_dir,
        log_callback=logger
    )

    log_queue.put(("__DONE__", clips))

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
        st.session_state.logs.clear()

        output_dir = os.path.abspath("data/outputs")
        os.makedirs(output_dir, exist_ok=True)

        st.session_state.worker = threading.Thread(
            target=worker_fn,
            args=(video_path, output_dir, st.session_state.log_queue),
            daemon=True
        )
        st.session_state.worker.start()

# --------------------------------------------------
# Drain log queue (EVERY rerun)
# --------------------------------------------------
while not st.session_state.log_queue.empty():
    item = st.session_state.log_queue.get()

    if isinstance(item, tuple) and item[0] == "__DONE__":
        st.session_state.clips = item[1]
        st.session_state.processing = False
    else:
        st.session_state.logs.append(item)

# --------------------------------------------------
# Progress log (user-friendly)
# --------------------------------------------------
if st.session_state.processing and st.session_state.logs:
    st.subheader("Progress")
    st.code(
        "\n".join(st.session_state.logs[-6:]),
        language="text"
    )

# --------------------------------------------------
# Results
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)")

    for i, clip in enumerate(st.session_state.clips, 1):
        clip = os.path.abspath(clip)

        st.subheader(f"Backhand {i}")
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
