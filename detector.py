def detect_backhands(
        video_path,
        output_dir,
        log_callback=print
):
    """
    Runs backhand detection on a video.
    Logs only ACCEPTED / REJECTED backhands.
    Returns a list of output clip paths.
    """

    # Suppress MediaPipe GPU warnings
    import os
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import cv2
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    import joblib
    from collections import deque
    from mediapipe.tasks import python
    import gc

    # ----------------------------
    # Load models
    # ----------------------------
    MODEL_ASSET_PATH = "pose_landmarker_heavy.task"
    KERAS_MODEL_PATH = "models/tennis_stroke/tennis_model_keras"
    REJECTOR_MODEL_PATH = "models/tennis_stroke/skeleton_rejector"
    ENCODER_PATH = "models/tennis_stroke/label_encoder_keras.pkl"

    try:
        log_callback("🔄 Loading models...")
        keras_model = tf.keras.models.load_model(
            KERAS_MODEL_PATH,
            compile=False
        )

        rejector = tf.keras.models.load_model(
            REJECTOR_MODEL_PATH,
            compile=False
        )

        le = joblib.load(ENCODER_PATH)
        log_callback("✅ Models loaded successfully")
    except Exception as e:
        log_callback(f"❌ Error loading models: {str(e)}")
        raise

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Video setup
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    log_callback(f"📹 Video: {width}x{height} @ {fps}fps | {total_frames} frames ({duration:.1f}s)")

    # Check video size
    if total_frames > 18000:  # ~10 min at 30fps
        log_callback(f"⚠️ Warning: Long video ({duration / 60:.1f} minutes). Processing may take several minutes.")

    frame_buffer = deque(maxlen=fps)  # 1 second pre-buffer

    # ----------------------------
    # MediaPipe — FORCE CPU (critical for Streamlit)
    # ----------------------------
    log_callback("🔄 Initializing pose detector...")
    base_options = python.BaseOptions(
        model_asset_path=MODEL_ASSET_PATH,
        delegate=python.BaseOptions.Delegate.CPU
    )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    clip_paths = []
    current_writer = None

    # ----------------------------
    # Main loop
    # ----------------------------
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            log_callback("✅ Pose detector ready. Starting analysis...")
            frame_count = 0
            clip_count = 0
            frames_to_record = 0
            cooldown_frames = 0
            stroke_active = False
            last_progress_log = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    log_callback(f"📍 Reached end of video at frame {frame_count}")
                    break

                timestamp_ms = int(1000 * frame_count / fps)
                frame_count += 1
                frame_buffer.append(frame.copy())

                # Progress logging every 2 seconds
                current_time = frame_count / fps
                if current_time - last_progress_log >= 2.0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    log_callback(
                        f"⏳ {progress:.1f}% | {current_time:.1f}s / {duration:.1f}s | {clip_count} backhands found")
                    last_progress_log = current_time

                    # Force garbage collection every 100 frames to prevent memory buildup
                    if frame_count % 100 == 0:
                        gc.collect()

                if cooldown_frames > 0:
                    cooldown_frames -= 1

                try:
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    log_callback(f"⚠️ Frame {frame_count} skipped: {str(e)}")
                    continue

                is_backhand = False

                # ----------------------------
                # Skeleton + Rejector logic
                # ----------------------------
                if result.pose_landmarks and not stroke_active:
                    landmarks = result.pose_landmarks[0]
                    all_coords = np.array(
                        [(lm.x * width, lm.y * height) for lm in landmarks]
                    )

                    mid_hip = (all_coords[23] + all_coords[24]) / 2
                    shoulder_dist = np.linalg.norm(all_coords[11] - all_coords[12])
                    if shoulder_dist < 1e-6:
                        shoulder_dist = 1.0

                    relevant_indices = [0, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24]
                    features = []

                    for idx in relevant_indices:
                        lm = landmarks[idx]
                        features.extend([
                            (all_coords[idx][0] - mid_hip[0]) / shoulder_dist,
                            (all_coords[idx][1] - mid_hip[1]) / shoulder_dist,
                            lm.visibility
                        ])

                    X = np.array([features], dtype=np.float32)

                    try:
                        skel_pred = keras_model.predict(X, verbose=0)[0]
                        skel_label = le.inverse_transform([np.argmax(skel_pred)])[0]
                        skel_conf = np.max(skel_pred)

                        if (
                                skel_label.lower() == "backhand"
                                and skel_conf > 0.85
                                and cooldown_frames == 0
                        ):
                            reject_score = rejector.predict(X, verbose=0)[0][0]

                            if reject_score > 0.9:
                                is_backhand = True
                                stroke_active = True

                                log_callback(
                                    f"[{frame_count / fps:6.2f}s] ✅ BACKHAND ACCEPTED | "
                                    f"Skel: {skel_conf:.2f} | Rejector: {reject_score:.2f}"
                                )
                            else:
                                log_callback(
                                    f"[{frame_count / fps:6.2f}s] ❌ BACKHAND REJECTED | "
                                    f"Skel: {skel_conf:.2f} | Rejector: {reject_score:.2f}"
                                )
                    except Exception as e:
                        log_callback(f"⚠️ Prediction error at frame {frame_count}: {str(e)}")
                        continue

                # ----------------------------
                # Clip writing (H.264 MP4)
                # ----------------------------
                if is_backhand and frames_to_record <= 0:
                    clip_count += 1
                    clip_path = os.path.join(output_dir, f"backhand_{clip_count}.mp4")

                    try:
                        current_writer = cv2.VideoWriter(
                            clip_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),  # Use mp4v for better compatibility
                            fps,
                            (width, height)
                        )

                        if not current_writer.isOpened():
                            log_callback(f"❌ Failed to create video writer for {clip_path}")
                            current_writer = None
                            continue

                        for f in frame_buffer:
                            current_writer.write(f)

                        frames_to_record = int(fps * 1.5)
                        cooldown_frames = fps * 2
                        clip_paths.append(clip_path)
                        log_callback(f"📹 Recording clip {clip_count}...")

                    except Exception as e:
                        log_callback(f"❌ VideoWriter error: {str(e)}")
                        if current_writer is not None:
                            current_writer.release()
                            current_writer = None
                        continue

                elif frames_to_record > 0:
                    if current_writer is not None:
                        current_writer.write(frame)
                    frames_to_record -= 1

                    if frames_to_record == 0:
                        if current_writer is not None:
                            current_writer.release()
                            current_writer = None
                        stroke_active = False
                        log_callback(f"✅ Clip {clip_count} saved")

            log_callback(f"🏁 Finished processing {frame_count} frames")

    except KeyboardInterrupt:
        log_callback("⚠️ Processing interrupted by user")
    except Exception as e:
        log_callback(f"❌ Error during processing: {str(e)}")
        import traceback
        log_callback(traceback.format_exc())
        raise

    finally:
        if current_writer is not None:
            current_writer.release()
        cap.release()
        log_callback(f"✅ Processing complete! Found {len(clip_paths)} backhand(s)")

    return clip_paths