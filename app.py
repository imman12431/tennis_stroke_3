import os
import time
import tempfile
import threading
import queue

import streamlit as st
from detector import detect_backhands

# --------------------------------------------------
# Streamlit config
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
if "worker" not in st.session_state:
    st.session_state.worker = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_label" not in st.session_state:
    st.session_state.video_label = None

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

if st.session_state.processing:
    st.info("⏳ Processing video… please wait.")

st.subheader("Choose a video")

col1, col2, col3 = st.columns(3)

with col1:
    uploaded_file = st.file_uploader(
        "Upload your own video",
        type=["mp4"],
        disabled=st.session_state.processing
    )

with col2:
    if st.button("▶️ Demo: Sinner", disabled=st.session_state.processing):
        st.session_state.video_path = os.path.abspath("sinner.mp4")
        st.session_state.video_label = "Sinner demo"

with col3:
    if st.button("▶️ Demo: Djokovic", disabled=st.session_state.processing):
        st.session_state.video_path = os.path.abspath("djokovic.mp4")
        st.session_state.video_label = "Djokovic demo"


# --------------------------------------------------
# Handle upload
# --------------------------------------------------
if uploaded_file and not st.session_state.processing:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.video_path = tmp.name
        st.session_state.video_label = "Uploaded video"


# --------------------------------------------------
# Selected video preview + run button
# --------------------------------------------------
if st.session_state.video_path:
    st.markdown(f"**Selected video:** {st.session_state.video_label}")
    st.video(st.session_state.video_path)

    if st.button("▶️ Run Backhand Detection", disabled=st.session_state.processing):
        st.session_state.processing = True
        st.session_state.clips = []
        st.session_state.logs.clear()

        output_dir = os.path.abspath("data/outputs")
        os.makedirs(output_dir, exist_ok=True)

        st.session_state.worker = threading.Thread(
            target=worker_fn,
            args=(
                st.session_state.video_path,
                output_dir,
                st.session_state.log_queue
            ),
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
# Progress log (kept, but clean)
# --------------------------------------------------
if st.session_state.processing and st.session_state.logs:
    st.subheader("Progress")
    st.code("\n".join(st.session_state.logs[-10:]), language="text")


# --------------------------------------------------
# Results
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:
    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)")

    for i, clip in enumerate(st.session_state.clips, 1):
        st.subheader(f"Backhand {i}")

        st.video(clip)

        with open(clip, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            "⬇️ Download clip",
            data=video_bytes,
            file_name=os.path.basename(clip),
            mime="video/mp4",
            key=f"download_{i}"
        )


# --------------------------------------------------
# Force rerun while processing
# --------------------------------------------------
if st.session_state.processing:
    time.sleep(1)
    st.rerun()
