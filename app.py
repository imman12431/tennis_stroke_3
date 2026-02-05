import streamlit as st
import os

# Suppress warnings before other imports
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
            with st.spinner("Analyzing vide