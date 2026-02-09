import os
import time
import tempfile
import threading
import queue

import streamlit as st
from detector import detect_backhands

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

# --------------------------------------------------
# Demo videos (must exist in repo root)
# --------------------------------------------------
DEMO_VIDEOS = {
    "🎾 Jannik Sinner (demo)": "sinner.mp4",
    "🎾 Novak Djokovic (demo)": "djokovic.mp4",
}

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
# UI
# --------------------------------------------------
st.title("🎾 Tennis Backhand Detection")

st.write(
    "Upload your own tennis video **or** try one of the demo clips below. "
    "The app will automatically detect and extract backhand shots."
)

if st.session_state.processing:
    st.info("⏳ Processing video… please wait.")

# --------------------------------------------------
# Video source selector
# --------------------------------------------------
st.subheader("Choose a video")

source = st.radio(
    "Video source",
    ["Upload your own", *DEMO_VIDEOS.keys()],
    disabled=st.session_state.processing
)

video_path = None

if source == "Upload your own":
    uploaded_file = st.file_uploader(
        "Upload an MP4 file",
        type=["mp4"],
        disabled=st.session_state.processing
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
else:
    video_path = os.path.abspath(DEMO_VIDEOS[source])

# --------------------------------------------------
# Video preview + start button
# --------------------------------------------------
if video_path and not st.session_state.processing:
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
# Drain log queue (every rerun)
# --------------------------------------------------
while not st.session_state.log_queue.empty():
    item = st.session_state.log_queue.get()

    if isinstance(item, tuple) and item[0] == "__DONE__":
        clips = item[1]

        # wait briefly for files to appear
        for _ in range(30):
            if all(os.path.exists(p) for p in clips):
                break
            time.sleep(0.1)

        st.session_state.clips = clips
        st.session_state.processing = False
    else:
        st.session_state.logs.append(item)

# --------------------------------------------------
# Progress log (clean, user-facing)
# --------------------------------------------------
if st.session_state.logs:
    with st.expander("Processing progress", expanded=True):
        st.code("\n".join(st.session_state.logs[-15:]), language="text")

# --------------------------------------------------
# Results
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)")

    for i, clip in enumerate(st.session_state.clips, 1):
        clip = os.path.abspath(clip)

        st.subheader(f"🎾 Backhand {i}")

        if os.path.exists(clip):
            st.video(clip)

            with open(clip, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="⬇️ Download clip",
                data=video_bytes,
                file_name=os.path.basename(clip),
                mime="video/mp4",
                key=f"download_{i}"
            )
        else:
            st.error("Clip could not be loaded.")

# --------------------------------------------------
# Auto-refresh while processing
# --------------------------------------------------
if st.session_state.processing:
    time.sleep(1)
    st.rerun()
