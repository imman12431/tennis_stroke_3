import os
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

st.title("🎾 Tennis Backhand Detection Demo")
st.markdown(
    "Upload a tennis match video. The model will automatically "
    "detect and extract **backhand strokes**."
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "processing" not in st.session_state:
    st.session_state.processing = False

if "clips" not in st.session_state:
    st.session_state.clips = []

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

if "worker_started" not in st.session_state:
    st.session_state.worker_started = False

# --------------------------------------------------
# Thread communication
# --------------------------------------------------
log_queue = queue.Queue()

def streamlit_logger(msg: str):
    log_queue.put(msg)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Debug")
    if st.button("🗑️ Clear logs"):
        st.session_state.log_lines.clear()

# --------------------------------------------------
# Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    disabled=st.session_state.processing
)

if uploaded_file and not st.session_state.processing:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("▶️ Run Backhand Detection"):
        st.session_state.processing = True
        st.session_state.worker_started = False
        st.session_state.clips.clear()
        st.session_state.log_lines.clear()

        # 🔑 Per-run output directory (cloud-safe)
        output_dir = tempfile.mkdtemp(prefix="backhands_")

        def worker():
            try:
                clips = detect_backhands(
                    video_path=video_path,
                    output_dir=output_dir,
                    log_callback=streamlit_logger
                )
                log_queue.put(("__DONE__", clips))
            except Exception as e:
                log_queue.put(("__ERROR__", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        st.session_state.worker_started = True

# --------------------------------------------------
# Processing UI
# --------------------------------------------------
if st.session_state.processing:

    st.warning("⏳ Processing video… please wait. Do not refresh the page.")

    log_box = st.empty()

    # Drain queue
    while not log_queue.empty():
        item = log_queue.get()

        if isinstance(item, tuple) and item[0] == "__DONE__":
            raw_clips = item[1]

            # 🔑 ABSOLUTE PATH FIX (THIS IS THE KEY)
            st.session_state.clips = [
                os.path.abspath(c) for c in raw_clips if os.path.exists(c)
            ]

            st.session_state.processing = False

        elif isinstance(item, tuple) and item[0] == "__ERROR__":
            st.session_state.processing = False
            st.error(item[1])

        else:
            st.session_state.log_lines.append(item)

    # Show last logs
    log_box.code(
        "\n".join(st.session_state.log_lines[-20:]),
        language="text"
    )

    st.experimental_rerun()

# --------------------------------------------------
# Show results
# --------------------------------------------------
if not st.session_state.processing and st.session_state.clips:

    st.success(f"✅ Detected {len(st.session_state.clips)} backhand(s)")

    for i, clip in enumerate(st.session_state.clips, 1):
        with st.expander(f"🎾 Backhand {i}", expanded=(i == 1)):

            if os.path.exists(clip):
                st.video(clip)

                # Optional download
                with open(clip, "rb") as f:
                    st.download_button(
                        label="⬇️ Download clip",
                        data=f,
                        file_name=os.path.basename(clip),
                        mime="video/mp4"
                    )
            else:
                st.warning(f"Clip missing: {clip}")
