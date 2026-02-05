if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    # Check video length
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    st.video(video_path)
    st.info(f"📹 Video duration: {duration:.1f} seconds ({total_frames} frames @ {fps}fps)")

    if duration > 180:  # 3 minutes
        st.warning(
            "⚠️ Video is longer than 3 minutes. Processing may take a while or fail on limited resources. Consider using a shorter clip for testing.")