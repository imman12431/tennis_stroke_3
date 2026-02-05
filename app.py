import streamlit as st

st.set_page_config(
    page_title="Tennis Backhand Detector",
    layout="wide"
)

import os
import tempfile
from detector import detect_backhands

st.title("🎾 Tennis Backhand Detection Demo")
st.markdown(
    "Upload a tennis match video. The model will automatically "
    "detect and extract **backhand strokes**."
)

# -----------------------
# File upload
# -----------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("▶️ Run Backhand Detection"):
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)

        log_area = st.empty()
        logs = []


        def streamlit_logger(msg):
            logs.append(msg)
            log_area.code("\n".join(logs[-12:]))


        try:
            with st.spinner("Analyzing video..."):
                clips = detect_backhands(
                    video_path=video_path,
                    output_dir=output_dir,
                    log_callback=streamlit_logger
                )

            st.success(f"Detected {len(clips)} backhand(s)!")

            if clips:
                for i, clip in enumerate(clips, 1):
                    st.subheader(f"Backhand {i}")
                    st.video(clip)
            else:
                st.info("No backhands detected in this video.")

        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

        finally:
            # Cleanup temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)