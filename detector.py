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


def detect_backhands(video_path, output_dir, log_callback=print, batch_size=256):
    """
    Phase 1: detect backhand frame indices using batch predictions
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
    # PHASE 1: Detection pass
    # ============================
    log("Starting detection pass")

    accepted_frames = []
    skeleton_features = []
    skeleton_frame_indices = []
    cooldown_frames = 0
    frame_count = 0

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if cooldown_frames > 0:
                cooldown_frames -= 1
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(
                mp_image, int(1000 * frame_count / fps)
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                all_coords = np.empty((33, 2), dtype=np.float32)
                for i, lm in enumerate(landmarks):
                    all_coords[i, 0] = lm.x * width
                    all_coords[i, 1] = lm.y * height

                mid_hip = (all_coords[23] + all_coords[24]) * 0.5
                shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12])
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

                skeleton_features.append(feats)
                skeleton_frame_indices.append(frame_count)

                del all_coords, feats

            del frame, mp_image, result

    cap.release()
    gc.collect()

    log(f"Collected {len(skeleton_features)} skeletons. Running batched predictions...")

    # ============================
    # PHASE 1B: Batched predictions
    # ============================
    skeleton_features = np.array(skeleton_features, dtype=np.float32)
    num_skeletons = skeleton_features.shape[0]

    for i in range(0, num_skeletons, batch_size):
        batch_X = skeleton_features[i:i+batch_size]
        preds = keras_model.predict(batch_X, verbose=0)
        rejector_scores = rejector.predict(batch_X, verbose=0)

        for j, pred in enumerate(preds):
            class_idx = int(np.argmax(pred))
            confidence = float(pred[class_idx])
            label = le.inverse_transform([class_idx])[0]

            if label.lower() == "backhand" and confidence > 0.85:
                if rejector_scores[j][0] > 0.9:
                    frame_idx = skeleton_frame_indices[i + j]
                    accepted_frames.append(frame_idx)

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

        mp4_path = os.path.abspath(
            os.path.join(output_dir, f"backhand_{i}.mp4")
        )

        write_mp4_h264(mp4_path, frames, fps)

        clip_paths.append(mp4_path)
        log(f"Wrote clip #{i}")

        del frames
        gc.collect()

    cap.release()
    log("All clips written")

    return clip_paths
