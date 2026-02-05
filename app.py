import os
import time
import tempfile

# --------------------------------------------------
# Configure TensorFlow BEFORE any imports that use it
# --------------------------------------------------
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import streamlit as st

st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

from detector import detect_backhands

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "processing" not in st.session_state:
    st.session_state.processing = False

if "clips" not in st.session_state:
    st.session_state.clips = []

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🎾 Tennis Backhand Detection Demo")
st.markdown(
    "Upload a tennis match video. The model will automatically "
    "detect and extract **backhand strokes**."
)

st.info("💡 Videos are processed in batches to handle memory efficiently.")

# Soft status banner (DO NOT block rendering)
if st.session_state.processing:
    st.info("⏳ Processing video… please wait. Do not refresh the page.")

# --------------------------------------------------
# Sidebar debug tools
# --------------------------------------------------
with st.sidebar:
    st.header("Debug")

    if st.button("🔍 View Debug Log", disabled=st.session_state.processing):
        log_path = "detector_debug.log"
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                st.text_area("Debug Log", f.read(), height=400)
        else:
            st.info("No debug log found yet.")

    if st.button("🗑️ Clear Debug Log", disabled=st.session_state.processing):
        log_path = "detector_debug.log"
        if os.path.exists(log_path):
            os.remove(log_path)
            st.success("Debug log cleared!")

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    disabled=st.session_state.processing
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    # --------------------------------------------------
    # Run button (safe against reruns)
    # --------------------------------------------------
    run_clicked = st.button(
        "▶️ Run Backhand Detection",
        disabled=st.session_state.processing
    )

    if run_clicked and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.clips = []

        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Log UI
        st.subheader("Processing Log")
        log_area = st.empty()
        logs = []

        # Throttled logger (prevents Streamlit overload)
        last_log_time = [0.0]

        def streamlit_logger(msg):
            logs.append(msg)
            now = time.time()
            if now - last_log_time[0] > 1.5:
                log_area.code("\n".join(logs[-15:]), language="text")
                last_log_time[0] = now

        try:
            with st.spinner("Analyzing video in batches..."):
                clips = detect_backhands(
                    video_path=video_path,
                    output_dir=output_dir,
                    log_callback=streamlit_logger
                )

            st.session_state.clips = clips
            st.success(f"✅ Detected {len(clips)} backhand(s)!")

            if clips:
                st.subheader("Detected Backhands")
                for i, clip in enumerate(clips, 1):
                    with st.expander(f"🎾 Backhand {i}", expanded=(i == 1)):
                        st.video(clip)
            else:
                st.info("No backhands detected in this video.")

        except Exception as e:
            st.error(f"❌ Error during detection: {str(e)}")

            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())

            log_path = "detector_debug.log"
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    st.subheader("🔍 Debug Log (Last 50 lines)")
                    st.code("".join(f.readlines()[-50:]), language="text")

        finally:
            # Cleanup temp file
            try:
                if os.path.exists(video_path):
                    os.unlink(video_path)
            except:
                pass

            st.session_state.processing = False

# --------------------------------------------------
# Show previous results
# --------------------------------------------------
elif st.session_state.clips:
    st.success(f"✅ Previously detected {len(st.session_state.clips)} backhand(s)")
    st.subheader("Detected Backhands")
    for i, clip in enumerate(st.session_state.clips, 1):
        with st.expander(f"🎾 Backhand {i}"):
            if os.path.exists(clip):
                st.video(clip)
            else:
                st.warning(f"Clip file not found: {clip}")
