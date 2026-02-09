import os
import time
import tempfile
import threading
import queue
import subprocess

# --------------------------------------------------
# TensorFlow env
# --------------------------------------------------
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import streamlit as st
from detector import detect_backhands

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
if "worker" not in st.session_state:
    st.session_state.worker = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# --------------------------------------------------
# ffmpeg availability check
# --------------------------------------------------
with st.sidebar:
    st.header("Environment")
    try:
        v = subprocess.check_output(["ffmpeg", "-version"]).decode().splitlines()[0]
        st.success(f"ffmpeg OK: {v}")
    except Exception as e:
        st.error(f"ffmpeg NOT available: {e}")

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎾 Tennis Backhand Detection Demo")

if st.session_state.processing:
    st.info("⏳ Processing video… please wait. Do not refresh the page.")

st.write("Working directory:", os.getcwd())

# --------------------------------------------------
# Sidebar (logs)
# --------------------------------------------------
with st.sidebar:
    st.header("Debug")

    if st.button("🔍 View Debug Log", disabled=st.session_state.processing):
        if os.path.exists("detector_debug.log"):
            st.text_area("Debug Log", open("detector_debug.log").read(), height=400)

    if st.button("🗑️ Clear Debug Log", disabled=st.session_state.processing):
        if os.path.exists("detector_debug.log"):
            os.remove("detector_debug.log")
            st.success("Debug log cleared!")

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4"],
    disabled=st.session_state.processing
)


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
# Start job
# --------------------------------------------------
if uploaded_file and not st.session_state.processing:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

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
        clips = item[1]

        # --- CRITICAL FIX: wait until files actually exist ---
        for _ in range(30):  # up to ~3 seconds
            if all(os.path.exists(p) for p in clips):
                break
            time.sleep(0.1)

        st.session_state.clips = clips
        st.session_state.processing = False
    else:
        st.session_state.logs.append(item)

# --------------------------------------------------
# Log display
# --------------------------------------------------
if st.session_state.logs:
    st.subheader("Processing Log")
    st.code("\n".join(st.session_state.logs[-20:]), language="text")

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)!")

    for i, clip in enumerate(st.session_state.clips, 1):
        clip = os.path.abspath(clip)

        with st.expander(f"🎾 Backhand {i}", expanded=(i == 1)):
            st.write("Path:", clip)
            st.write("Exists:", os.path.exists(clip))

            if os.path.exists(clip):
                size = os.path.getsize(clip)
                st.write("Size (bytes):", size)

                # Debug: Check video properties
                import cv2

                cap = cv2.VideoCapture(clip)
                if cap.isOpened():
                    st.write(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                    st.write(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
                    st.write(f"Codec: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
                    cap.release()

                st.markdown("**Playback:**")
                st.video(clip)

                with open(clip, "rb") as f:
                    video_bytes = f.read()

                st.download_button(
                    "⬇️ Download clip",
                    data=video_bytes,
                    file_name=os.path.basename(clip),
                    mime="video/avi",
                    key=f"download_{i}"
                )
            else:
                st.error("File does NOT exist at render time!")

# --------------------------------------------------
# Force rerun while processing
# --------------------------------------------------
if st.session_state.processing:
    time.sleep(1)
    st.rerun()