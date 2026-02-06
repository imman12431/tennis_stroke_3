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
    Writes browser-playable MP4 clips.
    Returns list of absolute clip paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    process = psutil.Process()

    # ----------------------------
    # Logging helper
    # ----------------------------
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

    dual_log("🚀 Starting backhand detection (frame-by-frame)")

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
    frame_buffer = deque(maxlen=fps)  # 1-second pre-buffer

    # ----------------------------
    # Load models
    # ----------------------------
    keras_model = tf.keras.models.load_model("models/tennis_stroke/tennis_model_keras")
    rejector = tf.keras.models.load_model("models/tennis_stroke/skeleton_rejector")
    rejector.trainable = False
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    # ----------------------------
    # MediaPipe Tasks API
    # ----------------------------
    MODEL_ASSET_PATH = "pose_landmarker_heavy.task"
    base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    all_clip_paths = []
    clip_count = 0
    frames_to_record = 0
    current_writer = None
    cooldown_frames = 0
    frame_count = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_count / fps)
            frame_count += 1

            frame_buffer.append(frame.copy())
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

                # ----------------------------
                # Skeleton feature extraction
                # ----------------------------
                all_coords = np.array([(lm.x * width, lm.y * height) for lm in landmarks])
                mid_hip = (all_coords[23] + all_coords[24]) / 2
                shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12])
                if shoulder_dist < 1e-3:
                    shoulder_dist = 1.0

                idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                feats = []
                for i in idxs:
                    lm = landmarks[i]
                    feats.extend([
                        (all_coords[i][0] - mid_hip[0]) / shoulder_dist,
                        (all_coords[i][1] - mid_hip[1]) / shoulder_dist,
                        lm.visibility
                    ])
                X = np.array([feats], dtype=np.float32)

                # ----------------------------
                # Skeleton model prediction
                # ----------------------------
                pred = keras_model.predict_on_batch(X)[0]
                class_idx = np.argmax(pred)
                confidence = pred[class_idx]
                label = le.inverse_transform([class_idx])[0]

                dual_log(f"[Frame {frame_count}] Skeleton → {label} ({confidence:.2f})")

                # ----------------------------
                # CNN verification
                # ----------------------------
                if label.lower() == "backhand" and confidence > 0.85:
                    rejector_score = rejector.predict(X, verbose=0)[0][0]
                    dual_log(f"           Rejector → backhand prob: {rejector_score:.2f}")
                    if rejector_score > 0.9:
                        is_backhand = True
                        dual_log(f"           ✅ ACCEPTED by rejector")

            # ----------------------------
            # Clip saving
            # ----------------------------
            if is_backhand and frames_to_record <= 0:
                clip_count += 1
                clip_path = os.path.abspath(os.path.join(output_dir, f"backhand_{clip_count}.mp4"))
                current_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                for buf_frame in frame_buffer:
                    current_writer.write(buf_frame)
                frames_to_record = int(fps * 1.5)
                cooldown_frames = fps * 2
                all_clip_paths.append(clip_path)
                dual_log(f"🎬 Recording clip #{clip_count}")

            elif frames_to_record > 0:
                current_writer.write(frame)
                frames_to_record -= 1
                if frames_to_record == 0:
                    current_writer.release()
                    current_writer = None
                    dual_log(f"🟢 Finished clip #{clip_count}")

            gc.collect()

    cap.release()
    if current_writer:
        current_writer.release()

    dual_log(f"✅ DONE — {len(all_clip_paths)} backhands detected")
    return all_clip_paths
