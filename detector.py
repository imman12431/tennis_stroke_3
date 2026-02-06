import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
from mediapipe.tasks import python
import mediapipe as mp
import imageio.v2 as imageio
import threading
import queue

# --------------------------------------------------
# Helper: write H.264 MP4 clips
# --------------------------------------------------
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

# --------------------------------------------------
# Worker: frame reading thread
# --------------------------------------------------
def frame_reader(cap, frame_queue, stop_event):
    """Read frames from video and put them in a queue."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    stop_event.set()  # signal that reading is done

# --------------------------------------------------
# Main detection function
# --------------------------------------------------
def detect_backhands(video_path, output_dir, log_callback=print):
    """
    Phase 1: detect backhand frame indices
    Phase 2: write all clips at the end
    Uses multi-threaded frame reading to speed up video decoding.
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
        model_asset_path="pose_landmarker_light.task"
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # ============================
    # PHASE 1: Detection only (multi-threaded)
    # ============================
    log("Starting detection pass")

    accepted_frames = []
    cooldown_frames = 0
    frame_count = 0

    # -----------------------------
    # Start threaded frame reader
    # -----------------------------
    frame_queue = queue.Queue(maxsize=50)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event))
    reader_thread.start()

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

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

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

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

    # -----------------------------
    # Cleanup threaded reader
    # -----------------------------
    stop_event.set()
    reader_thread.join()
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
