import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import psutil
from collections import deque
from datetime import datetime

def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Faster backhand detection using MediaPipe Solutions API + OpenCV VideoWriter.
    Returns a list of absolute clip paths.
    """

    # ----------------------------
    # Environment setup
    # ----------------------------
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.config.set_visible_devices([], "GPU")  # CPU-only safe for Streamlit

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

    dual_log("🚀 Starting backhand detection (Faster CPU mode)")

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
    cap.release()

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Load ML models
    # ----------------------------
    keras_model = tf.keras.models.load_model("models/tennis_stroke/tennis_model_keras")
    rejector = tf.keras.models.load_model("models/tennis_stroke/skeleton_rejector")
    rejector.trainable = False
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    # ----------------------------
    # MediaPipe setup (Solutions API)
    # ----------------------------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    # ----------------------------
    # VideoWriter settings
    # ----------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_buffer = deque(maxlen=int(fps * 1.0))  # 1 sec pre-buffer

    all_clip_paths = []
    clip_count = 0
    frames_to_record = 0
    current_writer = None
    cooldown_frames = 0

    # ----------------------------
    # Detection loop
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp_s = frame_idx / fps

        frame_buffer.append(frame.copy())

        if cooldown_frames > 0:
            cooldown_frames -= 1

        # ----------------------------
        # Pose detection
        # ----------------------------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        is_backhand = False

        if results.pose_landmarks and cooldown_frames == 0:
            landmarks = results.pose_landmarks.landmark
            all_coords = np.array([(lm.x * width, lm.y * height) for lm in landmarks])
            mid_hip = (all_coords[23] + all_coords[24]) / 2
            shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12])
            shoulder_dist = max(shoulder_dist, 1.0)

            relevant_indices = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
            input_vector = []
            for idx in relevant_indices:
                norm_x = (all_coords[idx][0] - mid_hip[0]) / shoulder_dist
                norm_y = (all_coords[idx][1] - mid_hip[1]) / shoulder_dist
                input_vector.extend([norm_x, norm_y, landmarks[idx].visibility])

            input_data = np.array([input_vector], dtype=np.float32)

            # Skeleton prediction
            preds = keras_model.predict_on_batch(input_data)
            class_idx = np.argmax(preds[0])
            label = le.inverse_transform([class_idx])[0]
            confidence = preds[0][class_idx]

            # Rejector check
            if label.lower() == "backhand" and confidence > 0.85:
                rejector_score = rejector.predict(input_data, verbose=0)[0][0]
                if rejector_score > 0.9:
                    is_backhand = True

        # ----------------------------
        # Clip writing
        # ----------------------------
        if is_backhand and frames_to_record <= 0:
            clip_count += 1
            clip_path = os.path.abspath(os.path.join(output_dir, f"backhand_{clip_count}.mp4"))
            current_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            for buf_frame in frame_buffer:
                current_writer.write(buf_frame)
            frames_to_record = int(fps * 1.5)
            cooldown_frames = int(fps * 2)
            all_clip_paths.append(clip_path)
            dual_log(f"🎾 BACKHAND accepted #{clip_count}")

        elif frames_to_record > 0:
            current_writer.write(frame)
            frames_to_record -= 1
            if frames_to_record == 0:
                current_writer.release()
                current_writer = None

        if frame_idx % 50 == 0:
            gc.collect()  # periodic cleanup

    cap.release()
    if current_writer:
        current_writer.release()
    pose.close()

    dual_log(f"✅ DONE — {clip_count} backhands detected")
    return all_clip_paths
