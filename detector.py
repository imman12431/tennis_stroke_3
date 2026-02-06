def detect_backhands(
    video_path,
    output_dir,
    log_callback=print
):
    """
    Optimized tennis backhand detection.
    Same logic, faster structure.
    """

    import os
    import cv2
    import gc
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    import joblib
    import psutil
    from datetime import datetime
    import imageio.v2 as imageio

    # -------------------------
    # Environment setup
    # -------------------------
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.config.set_visible_devices([], "GPU")

    os.makedirs(output_dir, exist_ok=True)

    process = psutil.Process()
    log_file_path = "detector_debug.log"

    def dual_log(msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        mem = process.memory_info().rss / 1024 / 1024
        line = f"[{ts}] [MEM {mem:.1f}MB] {msg}"
        try:
            with open(log_file_path, "a") as f:
                f.write(line + "\n")
        except:
            pass
        try:
            log_callback(msg)
        except:
            pass

    dual_log("🚀 Starting backhand detection (OPTIMIZED)")

    # -------------------------
    # Video metadata
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # -------------------------
    # Models
    # -------------------------
    keras_model = tf.keras.models.load_model(
        "models/tennis_stroke/tennis_model_keras", compile=False
    )
    rejector = tf.keras.models.load_model(
        "models/tennis_stroke/skeleton_rejector", compile=False
    )
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    base_options = mp.tasks.python.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task",
        delegate=mp.tasks.python.BaseOptions.Delegate.CPU
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # -------------------------
    # Detection parameters
    # -------------------------
    POSE_EVERY_N_FRAMES = 3
    PRE_BUFFER_FRAMES = int(fps * 0.7)
    RECORD_FRAMES = int(fps * 1.5)
    COOLDOWN_FRAMES = int(fps * 1.2)

    # -------------------------
    # Phase 1: Detection
    # -------------------------
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    cooldown = 0
    stroke_active = False
    global_clip_count = 0

    frame_buffer = []
    clips = []  # list of (start_frame, end_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(1000 * frame_idx / fps)

        # Maintain rolling buffer
        frame_buffer.append(frame_idx)
        if len(frame_buffer) > PRE_BUFFER_FRAMES:
            frame_buffer.pop(0)

        if cooldown > 0:
            cooldown -= 1

        # Pose detection only every N frames
        if (
            frame_idx % POSE_EVERY_N_FRAMES == 0
            and not stroke_active
            and cooldown == 0
        ):
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                coords = np.array(
                    [(lm.x * width, lm.y * height) for lm in landmarks]
                )

                mid_hip = (coords[23] + coords[24]) / 2
                shoulder_dist = np.linalg.norm(coords[11] - coords[12]) or 1.0

                idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                feats = []
                for i in idxs:
                    lm = landmarks[i]
                    feats.extend([
                        (coords[i][0] - mid_hip[0]) / shoulder_dist,
                        (coords[i][1] - mid_hip[1]) / shoulder_dist,
                        lm.visibility
                    ])

                X = np.array([feats], dtype=np.float32)
                pred = keras_model.predict(X, verbose=0)[0]
                label = le.inverse_transform([np.argmax(pred)])[0]
                conf = np.max(pred)

                if label.lower() == "backhand" and conf > 0.85:
                    if rejector.predict(X, verbose=0)[0][0] > 0.9:
                        global_clip_count += 1
                        stroke_active = True

                        start = frame_buffer[0]
                        end = min(frame_idx + RECORD_FRAMES, total_frames - 1)

                        clips.append((start, end))
                        cooldown = COOLDOWN_FRAMES

                        dual_log(f"🎾 BACKHAND #{global_clip_count}")

        if stroke_active:
            # End stroke after RECORD_FRAMES
            if frame_idx >= clips[-1][1]:
                stroke_active = False

        if frame_idx % (fps * 5) == 0:
            dual_log(f"⏳ Progress: {(frame_idx / total_frames) * 100:.1f}%")

        frame_idx += 1

    cap.release()
    landmarker.close()
    gc.collect()

    # -------------------------
    # Phase 2: Write clips
    # -------------------------
    dual_log("✂️ Writing clips to disk")

    def write_clip(path, frames, fps):
        writer = imageio.get_writer(
            path,
            format="ffmpeg",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p"
        )
        for f in frames:
            writer.append_data(f[:, :, ::-1])
        writer.close()

    cap = cv2.VideoCapture(video_path)
    current_clip = 0
    current_frames = []

    clip_ranges = {
        i: (start, end)
        for i, (start, end) in enumerate(clips)
    }

    for i in range(len(clips)):
        current_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_ranges[i][0])

        for _ in range(clip_ranges[i][1] - clip_ranges[i][0]):
            ret, frame = cap.read()
            if not ret:
                break
            current_frames.append(frame)

        clip_path = os.path.abspath(
            os.path.join(output_dir, f"backhand_{i + 1}.mp4")
        )
        write_clip(clip_path, current_frames, fps)

    cap.release()

    dual_log(f"✅ DONE — {len(clips)} backhands detected")
    return [
        os.path.abspath(os.path.join(output_dir, f"backhand_{i + 1}.mp4"))
        for i in range(len(clips))
    ]
