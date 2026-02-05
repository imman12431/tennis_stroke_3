def detect_backhands(
        video_path,
        output_dir,
        log_callback=print
):
    """
    Runs backhand detection on a video in batches to avoid memory issues.
    Logs only ACCEPTED / REJECTED backhands.
    Returns a list of output clip paths.
    """

    # Suppress warnings
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
    import sys
    from datetime import datetime
    import psutil  # For memory monitoring

    # Create log file
    log_file_path = "detector_debug.log"

    # Get current process
    process = psutil.Process()

    def dual_log(msg):
        """Log to both callback and file with memory info"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Get memory usage
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB

        log_msg = f"[{timestamp}] [MEM: {mem_mb:.1f}MB] {msg}"

        try:
            with open(log_file_path, "a") as f:
                f.write(log_msg + "\n")
                f.flush()
        except:
            pass

        try:
            log_callback(msg)
        except:
            pass

    dual_log("=" * 50)
    dual_log("🚀 Starting backhand detection (BATCH MODE)")
    dual_log(f"Python version: {sys.version}")
    dual_log(f"TensorFlow version: {tf.__version__}")
    dual_log("=" * 50)

    # ----------------------------
    # Video info first
    # ----------------------------
    dual_log(f"🔄 Analyzing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    dual_log(f"📹 Video: {width}x{height} @ {fps}fps | {total_frames} frames ({duration:.1f}s)")

    # Calculate batch size (process 10 seconds at a time - smaller batches)
    BATCH_DURATION = 10  # Reduced from 15 to 10
    batch_size = fps * BATCH_DURATION
    num_batches = (total_frames + batch_size - 1) // batch_size

    dual_log(f"📦 Processing in {num_batches} batches of ~{BATCH_DURATION}s each")

    os.makedirs(output_dir, exist_ok=True)
    all_clip_paths = []
    global_clip_count = 0

    # ----------------------------
    # Load models once
    # ----------------------------
    MODEL_ASSET_PATH = "pose_landmarker_heavy.task"
    KERAS_MODEL_PATH = "models/tennis_stroke/tennis_model_keras"
    REJECTOR_MODEL_PATH = "models/tennis_stroke/skeleton_rejector"
    ENCODER_PATH = "models/tennis_stroke/label_encoder_keras.pkl"

    try:
        dual_log("🔄 Loading models...")
        keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
        dual_log("✅ Skeleton model loaded")

        rejector = tf.keras.models.load_model(REJECTOR_MODEL_PATH, compile=False)
        dual_log("✅ Rejector model loaded")

        le = joblib.load(ENCODER_PATH)
        dual_log("✅ All models loaded")
    except Exception as e:
        dual_log(f"❌ Error loading models: {str(e)}")
        raise

    # ----------------------------
    # MediaPipe setup
    # ----------------------------
    dual_log("🔄 Initializing MediaPipe...")
    try:
        base_options = python.BaseOptions(
            model_asset_path=MODEL_ASSET_PATH,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        dual_log("✅ MediaPipe configured")
    except Exception as e:
        dual_log(f"❌ MediaPipe error: {str(e)}")
        raise

    # ----------------------------
    # Process each batch
    # ----------------------------
    for batch_num in range(num_batches):
        start_frame = batch_num * batch_size
        end_frame = min(start_frame + batch_size, total_frames)

        dual_log("=" * 50)
        dual_log(f"📦 BATCH {batch_num + 1}/{num_batches}")
        dual_log(f"   Frames {start_frame} to {end_frame}")
        dual_log("=" * 50)

        # Check memory before starting batch
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        dual_log(f"💾 Memory before batch: {mem_mb:.1f}MB")

        # Open video for this batch
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_buffer = deque(maxlen=fps)
        current_writer = None

        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                frame_count = start_frame
                frames_to_record = 0
                cooldown_frames = 0
                stroke_active = False
                last_progress_log = 0
                last_mem_check = 0

                while frame_count < end_frame:
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            dual_log(f"📍 End of video at frame {frame_count}")
                            break

                        timestamp_ms = int(1000 * frame_count / fps)
                        frame_count += 1

                        if not stroke_active and cooldown_frames == 0:
                            frame_buffer.append(frame.copy())

                        # Progress logging every 2 seconds
                        current_time = frame_count / fps
                        if current_time - last_progress_log >= 2.0:
                            batch_progress = ((frame_count - start_frame) / (end_frame - start_frame)) * 100
                            overall_progress = (frame_count / total_frames) * 100
                            dual_log(
                                f"⏳ Batch: {batch_progress:.1f}% | Overall: {overall_progress:.1f}% | {global_clip_count} clips")
                            last_progress_log = current_time

                        # Memory check every 5 seconds
                        if current_time - last_mem_check >= 5.0:
                            mem_info = process.memory_info()
                            mem_mb = mem_info.rss / 1024 / 1024
                            dual_log(f"💾 Current memory: {mem_mb:.1f}MB")

                            # Warning if approaching 1GB limit
                            if mem_mb > 800:
                                dual_log(f"⚠️ WARNING: High memory usage ({mem_mb:.1f}MB)")

                            last_mem_check = current_time
                            gc.collect()

                        if cooldown_frames > 0:
                            cooldown_frames -= 1

                        # Skip MediaPipe if recording
                        if frames_to_record > 0:
                            if current_writer is not None:
                                current_writer.write(frame)
                            frames_to_record -= 1

                            if frames_to_record == 0:
                                if current_writer is not None:
                                    current_writer.release()
                                    current_writer = None
                                stroke_active = False
                                dual_log(f"✅ Clip {global_clip_count} saved")
                                gc.collect()
                            continue

                        # MediaPipe processing
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)

                        is_backhand = False

                        # Skeleton + Rejector logic
                        if result.pose_landmarks and not stroke_active:
                            landmarks = result.pose_landmarks[0]
                            all_coords = np.array([(lm.x * width, lm.y * height) for lm in landmarks])

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

                            if (skel_label.lower() == "backhand" and skel_conf > 0.85 and cooldown_frames == 0):
                                reject_score = rejector.predict(X, verbose=0)[0][0]

                                if reject_score > 0.9:
                                    is_backhand = True
                                    stroke_active = True
                                    dual_log(
                                        f"[{frame_count / fps:6.2f}s] ✅ BACKHAND | Skel: {skel_conf:.2f} | Rej: {reject_score:.2f}")

                        # Clip writing
                        if is_backhand and frames_to_record <= 0:
                            global_clip_count += 1
                            clip_path = os.path.join(output_dir, f"backhand_{global_clip_count}.mp4")
                            dual_log(f"📹 Creating clip {global_clip_count}")

                            current_writer = cv2.VideoWriter(
                                clip_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                (width, height)
                            )

                            if not current_writer.isOpened():
                                dual_log(f"❌ Video writer failed")
                                current_writer = None
                                continue

                            for f in frame_buffer:
                                current_writer.write(f)

                            frames_to_record = int(fps * 1.5)
                            cooldown_frames = fps * 2
                            all_clip_paths.append(clip_path)

                    except Exception as e:
                        dual_log(f"❌ Frame {frame_count} error: {str(e)}")
                        continue

                dual_log(f"✅ Batch {batch_num + 1} complete")

        except Exception as e:
            dual_log(f"❌ Batch {batch_num + 1} error: {str(e)}")
            import traceback
            dual_log(traceback.format_exc())

        finally:
            if current_writer is not None:
                current_writer.release()
            cap.release()

            # Aggressive cleanup between batches
            del frame_buffer
            if 'mp_image' in locals():
                del mp_image
            if 'frame' in locals():
                del frame
            gc.collect()

            # Check memory after cleanup
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            dual_log(f"💾 Memory after batch cleanup: {mem_mb:.1f}MB")
            dual_log(f"🧹 Batch {batch_num + 1} resources released")

    dual_log("=" * 50)
    dual_log(f"✅ ALL BATCHES COMPLETE! {len(all_clip_paths)} backhands found")
    dual_log("=" * 50)

    return all_clip_paths
