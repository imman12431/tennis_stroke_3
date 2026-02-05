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

    # Suppress warnings and configure threading
    import os
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'

    import cv2
    import numpy as np
    import tensorflow as tf
    import mediapipe as mp
    import joblib
    from collections import deque
    from mediapipe.tasks import python
    import gc
    import sys
    from datetime import datetime

    # Configure TensorFlow for single-threaded execution (important for Streamlit)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Create log file
    log_file_path = "detector_debug.log"

    def dual_log(msg):
        """Log to both callback and file"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] {msg}"

        # Write to file immediately with flush
        with open(log_file_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()

        # Also call the callback
        log_callback(msg)

    dual_log("=" * 50)
    dual_log("🚀 Starting backhand detection")
    dual_log(f"Python version: {sys.version}")
    dual_log(f"OpenCV version: {cv2.__version__}")
    dual_log(f"TensorFlow version: {tf.__version__}")
    dual_log("=" * 50)

    # ----------------------------
    # Load models
    # ----------------------------
    MODEL_ASSET_PATH = "pose_landmarker_heavy.task"
    KERAS_MODEL_PATH = "models/tennis_stroke/tennis_model_keras"
    REJECTOR_MODEL_PATH = "models/tennis_stroke/skeleton_rejector"
    ENCODER_PATH = "models/tennis_stroke/label_encoder_keras.pkl"

    try:
        dual_log("🔄 Loading skeleton model...")
        keras_model = tf.keras.models.load_model(
            KERAS_MODEL_PATH,
            compile=False
        )
        dual_log("✅ Skeleton model loaded")

        dual_log("🔄 Loading rejector model...")
        rejector = tf.keras.models.load_model(
            REJECTOR_MODEL_PATH,
            compile=False
        )
        dual_log("✅ Rejector model loaded")

        dual_log("🔄 Loading label encoder...")
        le = joblib.load(ENCODER_PATH)
        dual_log("✅ Label encoder loaded")

    except Exception as e:
        dual_log(f"❌ Error loading models: {str(e)}")
        raise

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Video setup
    # ----------------------------
    dual_log(f"🔄 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    dual_log(f"📹 Video: {width}x{height} @ {fps}fps | {total_frames} frames ({duration:.1f}s)")

    # Check video size
    if total_frames > 18000:  # ~10 min at 30fps
        dual_log(f"⚠️ Warning: Long video ({duration / 60:.1f} minutes). Processing may take several minutes.")

    frame_buffer = deque(maxlen=fps)  # 1 second pre-buffer

    # ----------------------------
    # MediaPipe — FORCE CPU (critical for Streamlit)
    # ----------------------------
    dual_log("🔄 Initializing MediaPipe pose detector...")
    try:
        base_options = python.BaseOptions(
            model_asset_path=MODEL_ASSET_PATH,
            delegate=python.BaseOptions.Delegate.CPU
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        dual_log("✅ MediaPipe options configured")
    except Exception as e:
        dual_log(f"❌ MediaPipe configuration failed: {str(e)}")
        raise

    clip_paths = []
    current_writer = None

    # ----------------------------
    # Main loop
    # ----------------------------
    try:
        dual_log("🔄 Creating pose landmarker...")
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            dual_log("✅ Pose detector ready. Starting frame-by-frame analysis...")
            dual_log("=" * 50)

            frame_count = 0
            clip_count = 0
            frames_to_record = 0
            cooldown_frames = 0
            stroke_active = False
            last_progress_log = 0

            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        dual_log(f"📍 Reached end of video at frame {frame_count}")
                        break

                    timestamp_ms = int(1000 * frame_count / fps)
                    frame_count += 1
                    frame_buffer.append(frame.copy())

                    # Progress logging every 2 seconds
                    current_time = frame_count / fps
                    if current_time - last_progress_log >= 2.0:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        dual_log(
                            f"⏳ {progress:.1f}% | Frame {frame_count}/{total_frames} | {current_time:.1f}s/{duration:.1f}s | {clip_count} clips")
                        last_progress_log = current_time

                        # Force garbage collection
                        if frame_count % 100 == 0:
                            gc.collect()

                    if cooldown_frames > 0:
                        cooldown_frames -= 1

                    # MediaPipe processing
                    try:
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )

                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    except Exception as e:
                        dual_log(f"⚠️ Frame {frame_count} MediaPipe error: {str(e)}")
                        continue

                    is_backhand = False

                    # ----------------------------
                    # Skeleton + Rejector logic
                    # ----------------------------
                    if result.pose_landmarks and not stroke_active:
                        try:
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

                                    dual_log(
                                        f"[{frame_count / fps:6.2f}s] ✅ BACKHAND ACCEPTED | "
                                        f"Skel: {skel_conf:.2f} | Rejector: {reject_score:.2f}"
                                    )
                                else:
                                    dual_log(
                                        f"[{frame_count / fps:6.2f}s] ❌ BACKHAND REJECTED | "
                                        f"Skel: {skel_conf:.2f} | Rejector: {reject_score:.2f}"
                                    )
                        except Exception as e:
                            dual_log(f"⚠️ Prediction error at frame {frame_count}: {str(e)}")
                            continue

                    # ----------------------------
                    # Clip writing
                    # ----------------------------
                    if is_backhand and frames_to_record <= 0:
                        clip_count += 1
                        clip_path = os.path.join(output_dir, f"backhand_{clip_count}.mp4")
                        dual_log(f"📹 Creating clip {clip_count}: {clip_path}")

                        try:
                            current_writer = cv2.VideoWriter(
                                clip_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                (width, height)
                            )

                            if not current_writer.isOpened():
                                dual_log(f"❌ Failed to create video writer for {clip_path}")
                                current_writer = None
                                continue

                            dual_log(f"✅ Video writer created, writing buffered frames...")
                            for f in frame_buffer:
                                current_writer.write(f)

                            frames_to_record = int(fps * 1.5)
                            cooldown_frames = fps * 2
                            clip_paths.append(clip_path)
                            dual_log(f"📹 Recording clip {clip_count} ({frames_to_record} more frames)...")

                        except Exception as e:
                            dual_log(f"❌ VideoWriter error: {str(e)}")
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
                            dual_log(f"✅ Clip {clip_count} saved successfully")

                except Exception as e:
                    dual_log(f"❌ Error processing frame {frame_count}: {str(e)}")
                    import traceback
                    dual_log(traceback.format_exc())
                    # Continue to next frame instead of crashing
                    continue

            dual_log("=" * 50)
            dual_log(f"🏁 Finished processing all {frame_count} frames")

    except KeyboardInterrupt:
        dual_log("⚠️ Processing interrupted by user")
    except Exception as e:
        dual_log(f"❌ FATAL ERROR during processing: {str(e)}")
        import traceback
        dual_log(traceback.format_exc())
        raise

    finally:
        dual_log("🧹 Cleaning up resources...")
        if current_writer is not None:
            current_writer.release()
            dual_log("✅ Video writer released")
        cap.release()
        dual_log("✅ Video capture released")
        dual_log("=" * 50)
        dual_log(f"✅ Processing complete! Found {len(clip_paths)} backhand(s)")
        dual_log("=" * 50)

    return clip_paths