def detect_backhands(
    video_path,
    output_dir,
    log_callback=print
):
    """
    Runs backhand detection on a video in batches.
    Outputs browser-compatible H.264 MP4 clips.
    """

    # ----------------------------
    # Imports
    # ----------------------------
    import os
    import cv2
    import gc
    import psutil
    import joblib
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    import imageio.v3 as iio

    from collections import deque
    from datetime import datetime
    from mediapipe.tasks import python

    # ----------------------------
    # TensorFlow safety
    # ----------------------------
    tf.config.set_visible_devices([], "GPU")

    # ----------------------------
    # Logging
    # ----------------------------
    log_file_path = "detector_debug.log"
    process = psutil.Process()

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

    dual_log("🚀 Starting backhand detection (BATCH MODE)")

    # ----------------------------
    # Video metadata
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ----------------------------
    # Models
    # ----------------------------
    keras_model = tf.keras.models.load_model(
        "models/tennis_stroke/tennis_model_keras",
        compile=False
    )
    rejector = tf.keras.models.load_model(
        "models/tennis_stroke/skeleton_rejector",
        compile=False
    )
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    # ----------------------------
    # MediaPipe
    # ----------------------------
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task",
        delegate=python.BaseOptions.Delegate.CPU
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # ----------------------------
    # H.264 writer helper
    # ----------------------------
    def write_mp4_h264(path, frames, fps):
        writer = iio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p"
        )
        try:
            for f in frames:
                writer.append_data(f[:, :, ::-1])  # BGR → RGB
        finally:
            writer.close()

    # ----------------------------
    # Batch config
    # ----------------------------
    BATCH_DURATION = 10
    batch_size = fps * BATCH_DURATION
    num_batches = (total_frames + batch_size - 1) // batch_size

    os.makedirs(output_dir, exist_ok=True)

    all_clip_paths = []
    global_clip_count = 0

    # ----------------------------
    # Processing loop
    # ----------------------------
    for batch_idx in range(num_batches):
        dual_log(f"📦 Batch {batch_idx + 1}/{num_batches}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, batch_idx * batch_size)

        frame_buffer = deque(maxlen=int(fps * 0.7))

        stroke_active = False
        cooldown = 0
        frames_to_record = 0
        current_clip_frames = []

        frame_idx = batch_idx * batch_size
        end_frame = min(frame_idx + batch_size, total_frames)

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_idx / fps)
            frame_idx += 1

            if not stroke_active and cooldown == 0:
                frame_buffer.append(frame)

            if cooldown > 0:
                cooldown -= 1

            # Recording phase
            if frames_to_record > 0:
                current_clip_frames.append(frame)
                frames_to_record -= 1

                if frames_to_record == 0:
                    clip_path = os.path.join(
                        output_dir,
                        f"backhand_{global_clip_count}.mp4"
                    )

                    write_mp4_h264(clip_path, current_clip_frames, fps)

                    all_clip_paths.append(clip_path)
                    current_clip_frames = []
                    stroke_active = False

                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and not stroke_active:
                landmarks = result.pose_landmarks[0]

                coords = np.array(
                    [(lm.x * width, lm.y * height) for lm in landmarks]
                )

                mid_hip = (coords[23] + coords[24]) / 2
                shoulder_dist = np.linalg.norm(coords[11] - coords[12]) or 1.0

                idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                features = []

                for i in idxs:
                    features.extend([
                        (coords[i][0] - mid_hip[0]) / shoulder_dist,
                        (coords[i][1] - mid_hip[1]) / shoulder_dist,
                        landmarks[i].visibility
                    ])

                X = np.array([features], dtype=np.float32)
                pred = keras_model.predict(X, verbose=0)[0]
                label = le.inverse_transform([np.argmax(pred)])[0]
                conf = np.max(pred)

                if label.lower() == "backhand" and conf > 0.85 and cooldown == 0:
                    if rejector.predict(X, verbose=0)[0][0] > 0.9:
                        global_clip_count += 1
                        stroke_active = True

                        current_clip_frames = list(frame_buffer)
                        frames_to_record = int(fps * 1.5)
                        cooldown = fps * 2

            gc.collect()

        cap.release()
        gc.collect()

    landmarker.close()

    dual_log(f"✅ DONE — {len(all_clip_paths)} backhands detected")
    return all_clip_paths
