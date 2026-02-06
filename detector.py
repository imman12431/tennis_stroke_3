import os
from collections import deque
from datetime import datetime
import gc

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import psutil
import imageio.v2 as imageio


def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Frame-by-frame tennis backhand detection.
    Writes browser-playable H.264 MP4 clips.
    Returns list of absolute clip paths.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Logging
    # ----------------------------
    process = psutil.Process()
    log_file = "detector_debug.log"

    def dual_log(msg):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        mem = process.memory_info().rss / 1024 / 1024
        line = f"[{ts}] [MEM {mem:.1f}MB] {msg}"

        try:
            with open(log_file, "a") as f:
                f.write(line + "\n")
        except:
            pass

        try:
            log_callback(msg)
        except:
            pass

    dual_log("🚀 Starting backhand detection (FRAME-BY-FRAME)")

    # ----------------------------
    # Load models
    # ----------------------------
    keras_model = tf.keras.models.load_model("models/tennis_stroke/tennis_model_keras", compile=False)
    rejector = tf.keras.models.load_model("models/tennis_stroke/skeleton_rejector", compile=False)
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    # ----------------------------
    # Video metadata
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ----------------------------
    # MediaPipe Pose
    # ----------------------------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)

    # ----------------------------
    # Helpers
    # ----------------------------
    frame_buffer = deque(maxlen=fps)  # 1 second pre-buffer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def write_clip(path, frames):
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for f in frames:
            writer.write(f)
        writer.release()

    # ----------------------------
    # Detection loop
    # ----------------------------
    frame_count = 0
    clip_count = 0
    frames_to_record = 0
    current_writer = None
    cooldown_frames = 0
    current_clip_frames = []
    all_clips = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_buffer.append(frame.copy())

        if cooldown_frames > 0:
            cooldown_frames -= 1

        # Convert frame for MediaPipe
        mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(mp_frame)
        is_backhand = False

        if results.pose_landmarks and cooldown_frames == 0:
            landmarks = results.pose_landmarks.landmark
            coords = np.array([(lm.x * width, lm.y * height) for lm in landmarks])

            mid_hip = (coords[23] + coords[24]) / 2
            shoulder_dist = np.linalg.norm(coords[11] - coords[12]) or 1.0

            idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
            features = []

            for idx in idxs:
                lm = landmarks[idx]
                features.extend([
                    (coords[idx][0] - mid_hip[0]) / shoulder_dist,
                    (coords[idx][1] - mid_hip[1]) / shoulder_dist,
                    lm.visibility
                ])

            X = np.array([features], dtype=np.float32)

            # Skeleton prediction
            skel_pred = keras_model.predict_on_batch(X)[0]
            label = le.inverse_transform([np.argmax(skel_pred)])[0]
            conf = np.max(skel_pred)

            if label.lower() == "backhand" and conf > 0.85:
                reject_score = rejector.predict_on_batch(X)[0][0]
                if reject_score > 0.9:
                    is_backhand = True
                    dual_log(f"🎾 Backhand detected at frame {frame_count}")

        # ----------------------------
        # Clip recording
        # ----------------------------
        if is_backhand and frames_to_record <= 0:
            clip_count += 1
            clip_path = os.path.abspath(os.path.join(output_dir, f"backhand_{clip_count}.mp4"))

            current_clip_frames = list(frame_buffer)
            frames_to_record = int(fps * 1.5)
            cooldown_frames = int(fps * 2)
            all_clips.append(clip_path)

            dual_log(f"🎬 Recording clip #{clip_count}")

        if frames_to_record > 0:
            current_clip_frames.append(frame)
            frames_to_record -= 1

            if frames_to_record == 0:
                write_clip(all_clips[-1], current_clip_frames)
                current_clip_frames = []

    cap.release()
    pose.close()
    dual_log(f"✅ Done! Total backhands: {clip_count}")

    return all_clips
