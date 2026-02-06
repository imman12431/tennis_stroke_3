import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
from datetime import datetime
import psutil
import mediapipe as mp
from mediapipe.tasks import python


def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Batch-first backhand detection.
    1) Extract pose features for all frames
    2) Run skeleton + rejector in batch
    3) Write clips based on accepted frames
    """

    os.makedirs(output_dir, exist_ok=True)
    process = psutil.Process()

    def log(msg):
        try:
            log_callback(msg)
        except:
            pass

    log("🚀 Starting backhand detection (batch-first)")

    # ----------------------------
    # Video setup
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frames = []
    features = []
    frame_indices = []

    # ----------------------------
    # Load models
    # ----------------------------
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

    # ----------------------------
    # MediaPipe Tasks
    # ----------------------------
    base_options = python.BaseOptions(
        model_asset_path="pose_landmarker_heavy.task"
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # ----------------------------
    # PASS 1: Pose extraction
    # ----------------------------
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_idx / fps)
            frame_idx += 1

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frames.append(frame)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                coords = np.array([(p.x * width, p.y * height) for p in lm])

                mid_hip = (coords[23] + coords[24]) / 2
                shoulder_dist = np.linalg.norm(coords[11] - coords[12])
                if shoulder_dist < 1e-3:
                    shoulder_dist = 1.0

                idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                feat = []
                for i in idxs:
                    feat.extend([
                        (coords[i][0] - mid_hip[0]) / shoulder_dist,
                        (coords[i][1] - mid_hip[1]) / shoulder_dist,
                        lm[i].visibility
                    ])

                features.append(feat)
                frame_indices.append(frame_idx - 1)
            else:
                features.append(None)
                frame_indices.append(frame_idx - 1)

    cap.release()
    gc.collect()

    # ----------------------------
    # PASS 2: Batch inference
    # ----------------------------
    valid_idxs = [i for i, f in enumerate(features) if f is not None]
    X = np.array([features[i] for i in valid_idxs], dtype=np.float32)

    preds = keras_model.predict(X, batch_size=512, verbose=0)
    reject_preds = rejector.predict(X, batch_size=512, verbose=0)

    accepted_frames = set()

    for i, p in enumerate(preds):
        class_idx = np.argmax(p)
        conf = p[class_idx]
        label = le.inverse_transform([class_idx])[0]

        if label.lower() == "backhand" and conf > 0.85:
            if reject_preds[i][0] > 0.9:
                accepted_frames.add(frame_indices[valid_idxs[i]])

    # ----------------------------
    # PASS 3: Clip writing
    # ----------------------------
    clip_count = 0
    cooldown = 0
    frame_buffer = deque(maxlen=fps)
    writer = None
    frames_to_record = 0
    all_clip_paths = []

    for i, frame in enumerate(frames):
        frame_buffer.append(frame)

        if cooldown > 0:
            cooldown -= 1

        if i in accepted_frames and cooldown == 0 and frames_to_record == 0:
            clip_count += 1
            path = os.path.abspath(
                os.path.join(output_dir, f"backhand_{clip_count}.mp4")
            )

            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            for f in frame_buffer:
                writer.write(f)

            frames_to_record = int(fps * 1.5)
            cooldown = int(fps * 2)
            all_clip_paths.append(path)

            log(f"🎾 Backhand #{clip_count} accepted")

        if frames_to_record > 0 and writer is not None:
            writer.write(frame)
            frames_to_record -= 1
            if frames_to_record == 0:
                writer.release()
                writer = None

    if writer:
        writer.release()

    log(f"✅ DONE — {len(all_clip_paths)} backhands detected")
    return all_clip_paths
