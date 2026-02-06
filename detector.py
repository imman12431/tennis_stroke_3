import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
from mediapipe.tasks import python
import mediapipe as mp


def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Memory-conscious backhand detection using MediaPipe Tasks (VIDEO mode).
    """

    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        try:
            log_callback(msg)
        except:
            pass

    # ----------------------------
    # Video setup
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frame_buffer = deque(maxlen=fps)

    # ----------------------------
    # Load models (ONCE)
    # ----------------------------
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

    # ----------------------------
    # MediaPipe setup
    # ----------------------------
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task"
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # ----------------------------
    # State
    # ----------------------------
    clip_paths = []
    clip_count = 0
    frames_to_record = 0
    cooldown_frames = 0
    current_writer = None
    frame_count = 0

    # ----------------------------
    # Main loop
    # ----------------------------
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_count / fps)
            frame_count += 1

            frame_buffer.append(frame)

            if cooldown_frames > 0:
                cooldown_frames -= 1

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            is_backhand = False

            if result.pose_landmarks and cooldown_frames == 0:
                landmarks = result.pose_landmarks[0]

                # ---- feature extraction ----
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

                # ---- model inference ----
                pred = keras_model.predict_on_batch(X)[0]
                class_idx = int(np.argmax(pred))
                confidence = float(pred[class_idx])
                label = le.inverse_transform([class_idx])[0]

                if label.lower() == "backhand" and confidence > 0.85:
                    rejector_score = float(
                        rejector.predict(X, verbose=0)[0][0]
                    )

                    if rejector_score > 0.9:
                        is_backhand = True
                        log(
                            f"Backhand accepted at frame {frame_count} "
                            f"(conf={confidence:.2f}, rej={rejector_score:.2f})"
                        )

                # ---- aggressively delete temp arrays ----
                del all_coords, feats, X, pred

            # ---- clip writing ----
            if is_backhand and frames_to_record <= 0:
                clip_count += 1
                log(f"Recording clip #{clip_count}")

                clip_path = os.path.abspath(
                    os.path.join(output_dir, f"backhand_{clip_count}.mp4")
                )

                current_writer = cv2.VideoWriter(
                    clip_path, fourcc, fps, (width, height)
                )

                while frame_buffer:
                    current_writer.write(frame_buffer.popleft())

                frames_to_record = int(fps * 1.5)
                cooldown_frames = fps * 2
                clip_paths.append(clip_path)

            elif frames_to_record > 0:
                current_writer.write(frame)
                frames_to_record -= 1

                if frames_to_record == 0:
                    current_writer.release()
                    current_writer = None
                    log(f"Finished clip #{clip_count}")
                    gc.collect()

            # ---- delete per-frame heavy objects ----
            del frame, mp_image, result

    # ----------------------------
    # Cleanup
    # ----------------------------
    cap.release()
    if current_writer:
        current_writer.release()

    gc.collect()
    log(f"Done — {len(clip_paths)} backhands detected")

    return clip_paths
