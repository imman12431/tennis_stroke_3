import streamlit as st
import os
import tempfile

# --------------------------------------------------
# Setup
# --------------------------------------------------

st.set_page_config(page_title="Tennis Backhand Detector", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEMO_VIDEOS = {
    "Jannik Sinner": "sinner.mp4",
    "Novak Djokovic": "djokovic.mp4",
}

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🎾 Tennis Backhand Detector")

st.markdown(
    """
Choose **how you want to try the app**:
- Use a **preloaded demo video**
- Upload **your own video**
"""
)

input_mode = st.radio(
    "Video source",
    options=["Use demo video", "Upload my own video"],
)

video_path = None

# --------------------------------------------------
# Demo video branch
# --------------------------------------------------

if input_mode == "Use demo video":
    demo_choice = st.radio(
        "Choose a demo video:",
        options=list(DEMO_VIDEOS.keys()),
    )

    video_filename = DEMO_VIDEOS[demo_choice]
    video_path = os.path.join(BASE_DIR, video_filename)

    st.video(video_path)
    st.success(f"Using demo video: {demo_choice}")

# --------------------------------------------------
# Upload branch
# --------------------------------------------------

else:
    uploaded_file = st.file_uploader(
        "Upload a tennis video (mp4)",
        type=["mp4", "mov"],
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(video_path)
        st.success("Using uploaded video")

# --------------------------------------------------
# Run detection
# --------------------------------------------------

st.divider()

if video_path is not None:
    if st.button("🚀 Run backhand detection"):
        st.info("Running detection...")
        st.write("Video path:", video_path)

        # ---- CALL YOUR PIPELINE HERE ----
        # clips = detect_backhands(video_path, ...)
        # st.success(f"Detected {len(clips)} backhands")

else:
    st.warning("Please select or upload a video to continue.")
