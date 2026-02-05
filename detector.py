def detect_backhands(
        video_path,
        output_dir,
        log_callback=print
):
    """
    Runs backhand detection on a video in batches to avoid memory issues.
    Logs progress and returns a list of output clip paths.
    """

    # ----------------------------
    # Environment + imports
    # ----------------------------
    import os
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import cv2
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    import joblib
    from collections import deque
    from mediapipe.tasks import python
    import gc
    from datetime import datetime
    import psutil
    import subprocess

    # 🔒 Force CPU only
    tf.config.set_visible_devices([], 'GPU')

    log_file_path = "detector_debug.log"
    process = psutil.Process()

    # ----------------------------
    # Logging helper
    # ----------------------------
    def dual_log(msg):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        mem_mb = process.memory_info().rss / 1024 / 1024
        log_msg = f"[{timestamp}] [MEM: {mem_mb:.1f}MB] {msg}"

        try:
            with open(log_file_path, "a") as f:
                f.write(log_msg + "\n")
        except:
            pass

        try:
            log_callback(msg)
        except:
            pass

    # ----------------------------
    # 🔑 H.264 re-encode helper
    # ----------------------------
    def reencode_h264(path):
        tmp = path.replace(".mp4", "_h264.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", path,
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            "-vcodec", "libx264",
            "-profile:v", "main",
            "-level", "3.1",
            tmp
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            os.replace(tmp, path)
            return True
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            return False

    dual_log("🚀 Starting backhand detection (BATCH MODE)")

    # ----------------------------
    # Video info
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    BATCH_DURATION = 10
    batch_size = fps * BATCH_DURATION
    num_batches = (total_frames + batch_size - 1) // batch_size

    os.makedirs(output_dir, exist_ok=True)
    all_clip_paths = []
    global_clip_count = 0

    # ----------------------------
    # Load models
    # ----------------------------
    keras_model = tf.keras.models.load_model(
        "models/tennis_stroke/tennis_model_keras", compile=False
    )
    rejector = tf.keras.models.load_model(
        "models/tennis_stroke/skeleton_rejector", compile=False
    )
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    # ----------------------------
    # MediaPipe setup (unchanged)
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
    # Batch processing
    # ----------------------------
    for batch_num in range(num_batches):
        start_frame = batch_num * batch_size
        end_frame = min(start_frame + batch_size, total_frames)

        dual_log(f"📦 Batch {batch_num + 1}/{num_batches}")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_buffer = deque(maxlen=int(fps * 0.7))

        current_writer = None
        frames_to_record = 0
        cooldown_frames = 0
        stroke_active = False
        frame_count = start_frame
        last_progress_log = 0

        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_count / fps)
            frame_count += 1

            if not stroke_active and cooldown_frames == 0:
                frame_buffer.append(frame)

            if frame_count / fps - last_progress_log >= 5.0:
                dual_log(
                    f"⏳ Progress: {(frame_count / total_frames) * 100:.1f}% | Clips: {global_clip_count}"
                )
                last_progress_log = frame_count / fps

            if cooldown_frames > 0:
                cooldown_frames -= 1

            if frames_to_record > 0:
                current_writer.write(frame)
                frames_to_record -= 1

                if frames_to_record == 0:
                    current_writer.release()
                    reencode_h264(clip_path)  # 🔑 FIX
                    current_writer = None
                    stroke_active = False

                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and not stroke_active:
                landmarks = result.pose_landmarks[0]
                all_coords = np.array(
                    [(lm.x * width, lm.y * height) for lm in landmarks]
                )

                mid_hip = (all_coords[23] + all_coords[24]) / 2
                shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12]) or 1.0

                indices = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                features = []
                for idx in indices:
                    lm = landmarks[idx]
                    features.extend([
                        (all_coords[idx][0] - mid_hip[0]) / shoulder_dist,
                        (all_coords[idx][1] - mid_hip[1]) / shoulder_dist,
                        lm.visibility
                    ])

                X = np.array([features], dtype=np.float32)
                skel_pred = keras_model.predict(X, verbose=0)[0]
                label = le.inverse_transform([np.argmax(skel_pred)])[0]
                conf = np.max(skel_pred)

                if label.lower() == "backhand" and conf > 0.85 and cooldown_frames == 0:
                    if rejector.predict(X, verbose=0)[0][0] > 0.9:
                        global_clip_count += 1
                        stroke_active = True

                        clip_path = os.path.join(
                            output_dir, f"backhand_{global_clip_count}.mp4"
                        )

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        current_writer = cv2.VideoWriter(
                            clip_path, fourcc, fps, (width, height)
                        )

                        dual_log("🎞️ VideoWriter opened with codec: mp4v")

                        for f in frame_buffer:
                            current_writer.write(f)

                        frames_to_record = int(fps * 1.5)
                        cooldown_frames = fps * 2
                        all_clip_paths.append(clip_path)

            gc.collect()

        cap.release()
        gc.collect()

    landmarker.close()

    dual_log(f"✅ DONE — {len(all_clip_paths)} backhands detected")
    return all_clip_paths
