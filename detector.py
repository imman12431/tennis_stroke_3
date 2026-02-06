import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
from mediapipe.tasks import python
import mediapipe as mp
import imageio.v2 as imageio


def write_mp4_h264(path, frames, fps):
    writer = imageio.get_writer(
        path,
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-movflags", "+faststart"]
    )
    for f in frames:
        writer.append_data(f[:, :, ::-1])  # BGR → RGB
    writer.close()


def detect_backhands(video_path, output_dir, log_callback=print, batch_size=16):
    """
    Phase 1: detect backhand frame indices (batched predictions)
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

    log("Loading models")
    keras_model = tf.keras.models.load_model("models/tennis_stroke/tennis_model_keras")
    rejector = tf.keras.models.load_model("models/tennis_stroke/skeleton_rejector")
    rejector.trainable = False
    le = joblib.load("models/tennis_stroke/label_encoder_keras.pkl")

    log("Initializing MediaPipe")
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_heavy.task")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # ============================
    # PHASE 1: Detection only (BATCHED)
    # ============================
    log("Starting detection pass (batched)")

    accepted_frames = []
    cooldown_frames = 0
    frame_count = 0

    # buffers for batching
    feature_batch = []
    frame_indices_batch = []

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(1000 * frame_count / fps)
            frame_count += 1

            if cooldown_frames > 0:
                cooldown_frames -= 1
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                all_coords = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)
                mid_hip = (all_coords[23] + all_coords[24]) * 0.5
                shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12])
                shoulder_dist = max(shoulder_dist, 1e-3)

                idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                feats = []
                for i in idxs:
                    feats.extend([
                        (all_coords[i][0] - mid_hip[0]) / shoulder_dist,
                        (all_coords[i][1] - mid_hip[1]) / shoulder_dist,
                        landmarks[i].visibility
                    ])

                feature_batch.append(feats)
                frame_indices_batch.append(frame_count)

            # Run batch if full
            if len(feature_batch) >= batch_size:
                X_batch = np.array(feature_batch, dtype=np.float32)
                preds = keras_model.predict_on_batch(X_batch)
                rejector_scores = rejector.predict(X_batch, verbose=0).flatten()
                labels = le.inverse_transform(np.argmax(preds, axis=1))

                for idx, label, conf, reject in zip(frame_indices_batch, labels, np.max(preds, axis=1), rejector_scores):
                    if label.lower() == "backhand" and conf > 0.85 and reject > 0.9:
                        if cooldown_frames == 0:
                            accepted_frames.append(idx)
                            cooldown_frames = fps * 2
                            log(f"Backhand accepted at frame {idx}")

                feature_batch = []
                frame_indices_batch = []

            del frame, mp_image, result, landmarks, all_coords

    # flush remaining batch
    if feature_batch:
        X_batch = np.array(feature_batch, dtype=np.float32)
        preds = keras_model.predict_on_batch(X_batch)
        rejector_scores = rejector.predict(X_batch, verbose=0).flatten()
        labels = le.inverse_transform(np.argmax(preds, axis=1))

        for idx, label, conf, reject in zip(frame_indices_batch, labels, np.max(preds, axis=1), rejector_scores):
            if label.lower() == "backhand" and conf > 0.85 and reject > 0.9:
                if cooldown_frames == 0:
                    accepted_frames.append(idx)
                    cooldown_frames = fps * 2
                    log(f"Backhand accepted at frame {idx}")

    cap.release()
    gc.collect()
    log(f"Detection finished ({len(accepted_frames)} backhands)")

    # ============================
    # PHASE 2: DIRECT H.264 MP4 WRITE
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

        frames = []
        f = start
        while f <= end:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            f += 1

        mp4_path = os.path.abspath(os.path.join(output_dir, f"backhand_{i}.mp4"))
        write_mp4_h264(mp4_path, frames, fps)

        clip_paths.append(mp4_path)
        log(f"Wrote clip #{i}")

        del frames
        gc.collect()

    cap.release()
    log("All clips written")
    return clip_paths
