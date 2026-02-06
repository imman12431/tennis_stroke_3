import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
from datetime import datetime
import psutil
from mediapipe.tasks import python
import mediapipe as mp
import imageio.v2 as imageio


def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Detects backhand strokes in a video using frame-by-frame MediaPipe Tasks API.
    All clips are written at the end using imageio + H.264 (browser-safe).
    Returns list of absolute clip paths.
    """

    os.makedirs(output_dir, exist_ok=True)
    process = psutil.Process()

    # -------------------------------------------------
    # Logging helper (minimal)
    # -------------------------------------------------
    def log(msg):
        try:
            log_callback(msg)
        except:
            pass

    log("🚀 Starting backhand detection")

    # -------------------------------------------------
    # Video setup
    # -------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_buffer = deque(maxlen=fps)  # 1 second pre-roll

    # -------------------------------------------------
    # Load models
    # -------------------------------------------------
    keras_model = tf.keras.models.load_model(
        "models/tennis_stroke/tennis_model_keras",
        compile=False
    )
    rejector = tf.keras.models.load_model(
        "models/tennis_stroke/skeleton_rejector",
        compile=False
    )
    rejector.trainable = False
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    # -------------------------------------------------
    # MediaPipe Tasks API
    # -------------------------------------------------
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task"
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # -------------------------------------------------
    # State
    # -------------------------------------------------
    all_clips = []   # list of (clip_path, frames)
    clip_count = 0

    frames_to_record = 0
    cooldown_frames = 0
    current_clip_frames = None

    frame_idx = 0

    log("🧍 MediaPipe pose detection running")

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_idx / fps)
            frame_idx += 1

            frame_buffer.append(frame)

            if cooldown_frames > 0:
                cooldown_frames -= 1

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            is_backhand = False

            if result.pose_landmarks and cooldown_frames == 0:
                landmarks = result.pose_landmarks[0]

                coords = np.array([
                    (lm.x * width, lm.y * height) for lm in landmarks
                ])

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

                pred = keras_model.predict_on_batch(X)[0]
                class_idx = np.argmax(pred)
                confidence = pred[class_idx]
                label = le.inverse_transform([class_idx])[0]

                if label.lower() == "backhand" and confidence > 0.85:
                    rejector_score = rejector.predict(X, verbose=0)[0][0]
                    if rejector_score > 0.9:
                        is_backhand = True

            # -------------------------------------------------
            # Clip collection (NO writing yet)
            # -------------------------------------------------
            if is_backhand and frames_to_record <= 0:
                clip_count += 1
                log(f"🎾 Backhand detected #{clip_count}")

                current_clip_frames = list(frame_buffer)
                frames_to_record = int(fps * 1.5)
                cooldown_frames = int(fps * 2)

            elif frames_to_record > 0:
                current_clip_frames.append(frame)
                frames_to_record -= 1

                if frames_to_record == 0:
                    clip_path = os.path.abspath(
                        os.path.join(output_dir, f"backhand_{clip_count}.mp4")
                    )
                    all_clips.append((clip_path, current_clip_frames))
                    current_clip_frames = None

            # aggressive cleanup
            gc.collect()

    cap.release()
    gc.collect()

    # -------------------------------------------------
    # FINAL PHASE — write all videos (H.264 safe)
    # -------------------------------------------------
    log("🎬 Writing final videos (H.264 / browser-safe)")

    final_paths = []

    for path, frames in all_clips:
        with imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            format="mp4",
            pixelformat="yuv420p"
        ) as writer:
            for f in frames:
                writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

        final_paths.append(path)

        # free memory immediately
        frames.clear()
        gc.collect()

    log(f"✅ DONE — {len(final_paths)} backhands written")

    return final_paths
