import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
import subprocess
from collections import deque
from mediapipe.tasks import python
import mediapipe as mp


def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Phase 1: detect backhand frame indices
    Phase 2: write all clips at the end
    """

    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        try:
            log_callback(msg)
        except:
            pass

    # ============================
    # PHASE 0: Setup
    # ============================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    log("Loading models")

    keras_model = tf.keras.models.load_model(
        "models/tennis_stroke/tennis_model_keras"
    )
    rejector = tf.keras.models.load_model(
        "models/tennis_stroke/skeleton_rejector"
    )
    rejector.trainable = False

    le = joblib.load(
        "models/tennis_stroke/label_encoder_keras.pkl"
    )

    log("Initializing MediaPipe")

    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task"
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # ============================
    # PHASE 1: Detection only
    # ============================
    log("Starting detection pass")

    accepted_frames = []
    cooldown_frames = 0
    frame_count = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_count / fps)
            frame_count += 1

            if cooldown_frames > 0:
                cooldown_frames -= 1
                del frame
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                all_coords = np.empty((33, 2), dtype=np.float32)
                for i, lm in enumerate(landmarks):
                    all_coords[i, 0] = lm.x * width
                    all_coords[i, 1] = lm.y * height

                mid_hip = (all_coords[23] + all_coords[24]) * 0.5
                shoulder_dist = np.linalg.norm(
                    all_coords[11] - all_coords[12]
                )
                if shoulder_dist < 1e-3:
                    shoulder_dist = 1.0

                idxs = (0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24)
                feats = np.empty(len(idxs) * 3, dtype=np.float32)

                j = 0
                for i in idxs:
                    feats[j] = (all_coords[i, 0] - mid_hip[0]) / shoulder_dist
                    feats[j + 1] = (all_coords[i, 1] - mid_hip[1]) / shoulder_dist
                    feats[j + 2] = landmarks[i].visibility
                    j += 3

                X = feats.reshape(1, -1)

                pred = keras_model.predict_on_batch(X)[0]
                class_idx = int(np.argmax(pred))
                confidence = float(pred[class_idx])
                label = le.inverse_transform([class_idx])[0]

                if label.lower() == "backhand" and confidence > 0.85:
                    rejector_score = float(
                        rejector.predict(X, verbose=0)[0][0]
                    )

                    if rejector_score > 0.9:
                        accepted_frames.append(frame_count)
                        cooldown_frames = fps * 2
                        log(f"Backhand accepted at frame {frame_count}")

                del all_coords, feats, X, pred

            del frame, mp_image, result

    cap.release()
    gc.collect()

    log(f"Detection finished ({len(accepted_frames)} backhands)")

    # ============================
    # PHASE 2: MJPG → H.264 (ffmpeg)
    # ============================
    if not accepted_frames:
        log("No clips to write")
        return []

    log("Starting video writing pass")

    cap = cv2.VideoCapture(video_path)
    clip_paths = []

    pre_frames = int(fps * 1.0)
    post_frames = int(fps * 1.5)

    for i, center_frame in enumerate(accepted_frames, 1):
        start = max(0, center_frame - pre_frames)
        end = center_frame + post_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        avi_path = os.path.abspath(
            os.path.join(output_dir, f"backhand_{i}.avi")
        )

        writer = cv2.VideoWriter(
            avi_path,
            fourcc,
            fps,
            (width, height)
        )

        f = start
        while f <= end:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            del frame
            f += 1

        writer.release()

        # ---- re-encode to H.264 (Streamlit-safe) ----
        mp4_path = avi_path.replace(".avi", ".mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", avi_path,
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level", "4.1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            mp4_path
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        os.remove(avi_path)

        clip_paths.append(mp4_path)
        log(f"Wrote clip #{i}")
        gc.collect()

    cap.release()
    log("All clips written")

    return clip_paths
