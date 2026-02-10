import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import joblib
from mediapipe.tasks import python
import mediapipe as mp
import threading
import queue


# --------------------------------------------------
# Helper: write video clips
# --------------------------------------------------
def write_video_clip(output_path, frames, fps):
    """Write frames to MP4 via AVI intermediate (for browser compatibility)"""
    if not frames:
        return output_path

    height, width = frames[0].shape[:2]

    # First write as AVI with XVID codec (reliable)
    temp_avi = output_path.replace('.mp4', '_temp.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    writer = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temp_avi}")

    for frame in frames:
        writer.write(frame)

    writer.release()

    # Convert AVI to MP4 using ffmpeg if available
    try:
        import subprocess
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_avi,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            output_path
        ], check=True, capture_output=True, timeout=10)

        os.remove(temp_avi)
        return output_path

    except Exception as e:
        os.rename(temp_avi, output_path)
        return output_path


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
    stop_event.set()


# --------------------------------------------------
# Main detection function
# --------------------------------------------------
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

    # Processing resolution (lower = faster)
    PROCESS_WIDTH = 640
    PROCESS_HEIGHT = 360

    log(f"Video: {width}x{height}, Processing at: {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
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
        model_asset_path="pose_landmarker_lite.task"
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    # ============================
    # PHASE 1: Detection (optimized)
    # ============================
    log("Starting detection pass")

    accepted_frames = []
    cooldown_frames = 0
    frame_count = 0

    frame_queue = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event))
    reader_thread.start()

    # Pre-allocate arrays to avoid repeated allocation
    all_coords = np.empty((33, 2), dtype=np.float32)
    feats = np.empty(33, dtype=np.float32)

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
                continue

            # Resize frame for faster processing
            frame_small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Use processing resolution for coordinates
                for i, lm in enumerate(landmarks):
                    all_coords[i, 0] = lm.x * PROCESS_WIDTH
                    all_coords[i, 1] = lm.y * PROCESS_HEIGHT

                mid_hip = (all_coords[23] + all_coords[24]) * 0.5
                shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12])
                if shoulder_dist < 1e-3:
                    shoulder_dist = 1.0

                # Optimized feature extraction
                idxs = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                j = 0
                for i in idxs:
                    feats[j] = (all_coords[i, 0] - mid_hip[0]) / shoulder_dist
                    feats[j + 1] = (all_coords[i, 1] - mid_hip[1]) / shoulder_dist
                    feats[j + 2] = landmarks[i].visibility
                    j += 3

                X = feats.reshape(1, -1)

                # Batch prediction
                pred = keras_model.predict_on_batch(X)[0]
                class_idx = int(np.argmax(pred))
                confidence = float(pred[class_idx])
                label = le.inverse_transform([class_idx])[0]

                if label.lower() == "backhand" and confidence > 0.85:
                    rejector_score = float(rejector.predict(X, verbose=0)[0][0])

                    if rejector_score > 0.9:
                        accepted_frames.append(frame_count)
                        cooldown_frames = fps * 2
                        log(f"Backhand accepted at frame {frame_count}")

    stop_event.set()
    reader_thread.join()
    cap.release()
    gc.collect()

    log(f"Detection finished ({len(accepted_frames)} backhands)")

    # ============================
    # PHASE 2: WRITE CLIPS (using original resolution)
    # ============================
    if not accepted_frames:
        log("No clips to write")
        return []

    log("Starting video writing pass")

    # Re-open video for writing clips at original resolution
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
            frames.append(frame)  # Keep original resolution for output
            f += 1

        clip_path = os.path.abspath(
            os.path.join(output_dir, f"backhand_{i}.mp4")
        )

        write_video_clip(clip_path, frames, fps)
        clip_paths.append(clip_path)
        log(f"Wrote clip #{i}")

        del frames
        gc.collect()

    cap.release()
    log("All clips written")

    return clip_paths