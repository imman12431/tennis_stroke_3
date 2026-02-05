import os

# Configure TensorFlow BEFORE any imports that use it
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import streamlit as st

st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

import tempfile
from detector import detect_backhands

st.title("🎾 Tennis Backhand Detection Demo")
st.markdown(
    "Upload a tennis match video. The model will automatically "
    "detect and extract **backhand strokes**."
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'clips' not in st.session_state:
    st.session_state.clips = []

# -----------------------
# File upload
# -----------------------
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

    if st.button("▶️ Run Backhand Detection", disabled=st.session_state.processing):
        st.session_state.processing = True
        st.session_state.clips = []

        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Create placeholder for logs
        log_container = st.container()
        with log_container:
            st.subheader("Processing Log")
            log_area = st.empty()

        logs = []


        def streamlit_logger(msg):
            logs.append(msg)
            # Show last 15 lines
            log_area.code("\n".join(logs[-15:]), language="text")


        try:
            with st.spinner("Analyzing video... This may take several minutes for long videos."):
                clips = detect_backhands(
                    video_path=video_path,
                    output_dir=output_dir,
                    log_callback=streamlit_logger
                )

            st.session_state.clips = clips
            st.session_state.processing = False

            st.success(f"✅ Detected {len(clips)} backhand(s)!")

            if clips:
                st.subheader("Detected Backhands")
                for i, clip in enumerate(clips, 1):
                    with st.expander(f"🎾 Backhand {i}", expanded=(i == 1)):
                        st.video(clip)
            else:
                st.info("No backhands detected in this video.")

        except Exception as e:
            st.session_state.processing = False
            st.error(f"❌ Error during detection: {str(e)}")

            with st.expander("Error Details"):
                import traceback

                st.code(traceback.format_exc())

        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(video_path):
                    os.unlink(video_path)
            except:
                pass

            st.session_state.processing = False

# Show previously detected clips if they exist
elif st.session_state.clips:
    st.success(f"✅ Previously detected {len(st.session_state.clips)} backhand(s)")
    st.subheader("Detected Backhands")
    for i, clip in enumerate(st.session_state.clips, 1):
        with st.expander(f"🎾 Backhand {i}", expanded=False):
            if os.path.exists(clip):
                st.video(clip)
            else:
                st.warning(f"Clip file not found: {clip}")