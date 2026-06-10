"""
FastAPI backend for the Tennis Backhand Detector.

Wraps the long-running ``detect_backhands`` pipeline in an async job model:
    POST /detect      -> submit a job (upload mp4 OR pick a demo), returns job_id
    GET  /status/{id} -> poll status + streamed progress logs + result clip names
    GET  /clips/{id}/{name} -> stream/download a result clip
    GET  /demos       -> list available demo videos
    GET  /health      -> cheap liveness check (uptime pinger target)

Detection is CPU/RAM heavy, so jobs run in background threads and are
serialized via a global lock so the small free container can't be OOM'd by
two large videos at once. Job state lives in memory (lost on restart — fine
for a demo).
"""

import os
import shutil
import time
import uuid
import threading
import logging
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core.detector import detect_backhands

# Configure logging
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS_DIR = os.path.join(APP_ROOT, "demos")
# Per-job working dirs. Defaults to a tmp path so it's writable even when the
# container runs as a non-root user (e.g. Hugging Face Spaces); override with
# the JOBS_DIR env var.
JOBS_DIR = os.environ.get("JOBS_DIR", "/tmp/tennis_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# Cap upload size to protect the free container (frames are buffered in memory).
MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # 60 MB
# Discard job working dirs older than this on each new submission.
JOB_TTL_SECONDS = 60 * 60  # 1 hour

# Demo videos shipped in the image: id -> (display name, filename in demos/)
DEMOS = {
    "sinner": ("Jannik Sinner", "sinner.mp4"),
    "djokovic": ("Novak Djokovic", "djokovic.mp4"),
}

# In-memory job store: job_id -> {status, logs, clips, error, created}
jobs = {}
jobs_lock = threading.Lock()           # guards the `jobs` dict
detection_lock = threading.Lock()      # serializes the heavy pipeline (1 at a time)

app = FastAPI(title="Tennis Backhand Detector API")

# CORS — allow the Vercel frontend + local dev. Comma-separated origins via env.
_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _sweep_old_jobs():
    """Delete working dirs (and job entries) older than the TTL."""
    now = time.time()
    with jobs_lock:
        stale = [
            jid for jid, j in jobs.items()
            if now - j.get("created", now) > JOB_TTL_SECONDS
        ]
        for jid in stale:
            jobs.pop(jid, None)

    if stale:
        logger.info(f"Cleaning up {len(stale)} expired jobs")
        for jid in stale:
            try:
                shutil.rmtree(os.path.join(JOBS_DIR, jid), ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to delete job dir {jid}: {e}")


def _run_job(job_id, video_path, output_dir):
    """Background worker: run detection (serialized) and record results."""
    def log(msg):
        text = msg if isinstance(msg, str) else str(msg)
        with jobs_lock:
            jobs[job_id]["logs"].append(text)

    logger.info(f"Job {job_id}: queued")
    # Serialize the heavy work so two videos don't blow the container's RAM.
    log("Queued — waiting for a free worker…")
    with detection_lock:
        logger.info(f"Job {job_id}: starting detection")
        try:
            log("Starting…")
            clips = detect_backhands(
                video_path=video_path,
                output_dir=output_dir,
                log_callback=log,
            )
            with jobs_lock:
                jobs[job_id]["clips"] = [os.path.basename(p) for p in clips]
                jobs[job_id]["status"] = "done"
            logger.info(f"Job {job_id}: completed with {len(clips)} clips")
        except Exception as e:
            logger.exception(f"Job {job_id}: detection failed")
            with jobs_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = str(e)
            log(f"Error: {e}")


def _start_job(video_path):
    """Register a job and kick off its background thread."""
    job_id = uuid.uuid4().hex
    output_dir = os.path.join(JOBS_DIR, job_id, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "logs": [],
            "clips": [],
            "error": None,
            "created": time.time(),
        }

    threading.Thread(
        target=_run_job,
        args=(job_id, video_path, output_dir),
        daemon=True,
    ).start()
    logger.debug(f"Started job {job_id} for {video_path}")
    return job_id


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demos")
def list_demos():
    return [{"id": k, "name": v[0]} for k, v in DEMOS.items()]


@app.get("/demo/{demo_id}")
def get_demo(demo_id: str):
    """Stream a demo video by id (sinner, djokovic, etc)."""
    if demo_id not in DEMOS:
        raise HTTPException(status_code=404, detail="Demo not found")
    _, filename = DEMOS[demo_id]
    path = os.path.join(DEMOS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/detect", status_code=202)
async def detect(file: UploadFile = File(None), demo: str = Form(None)):
    """Submit a detection job from either an uploaded mp4 or a demo id."""
    _sweep_old_jobs()

    if demo:
        if demo not in DEMOS:
            logger.warning(f"Unknown demo requested: {demo}")
            raise HTTPException(404, f"Unknown demo '{demo}'")
        video_path = os.path.join(DEMOS_DIR, DEMOS[demo][1])
        if not os.path.exists(video_path):
            logger.error(f"Demo video missing: {video_path}")
            raise HTTPException(500, "Demo video missing on server")
        logger.info(f"Requested demo: {demo}")
        return {"job_id": _start_job(video_path)}

    if file is None:
        logger.warning("POST /detect with no file or demo")
        raise HTTPException(400, "Provide either an uploaded 'file' or a 'demo' id")

    if file.content_type not in ("video/mp4", "application/octet-stream"):
        logger.warning(f"Invalid content type: {file.content_type}")
        raise HTTPException(415, "Only MP4 uploads are supported")

    # Read with a hard size cap so a huge upload can't exhaust memory/disk.
    job_id = uuid.uuid4().hex
    job_root = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_root, exist_ok=True)
    video_path = os.path.join(job_root, "input.mp4")

    size = 0
    logger.info(f"Job {job_id}: receiving upload {file.filename}")
    with open(video_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(job_root, ignore_errors=True)
                logger.warning(f"Job {job_id}: upload exceeds limit ({size} bytes)")
                raise HTTPException(
                    413,
                    f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
                )
            out.write(chunk)

    if size == 0:
        shutil.rmtree(job_root, ignore_errors=True)
        logger.warning(f"Job {job_id}: empty upload")
        raise HTTPException(400, "Empty upload")

    logger.info(f"Job {job_id}: upload complete ({size} bytes)")
    # Reuse the pre-created job_id/dir for this upload job.
    output_dir = os.path.join(job_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "logs": [],
            "clips": [],
            "error": None,
            "created": time.time(),
        }
    threading.Thread(
        target=_run_job,
        args=(job_id, video_path, output_dir),
        daemon=True,
    ).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            logger.debug(f"Status requested for unknown job: {job_id}")
            raise HTTPException(404, "Unknown job")
        return {
            "status": job["status"],
            "logs": list(job["logs"]),
            "clips": list(job["clips"]),
            "error": job["error"],
        }


@app.get("/clips/{job_id}/{filename}")
def get_clip(job_id: str, filename: str):
    # Guard against path traversal — only allow plain basenames.
    if filename != os.path.basename(filename):
        logger.warning(f"Path traversal attempt: {job_id}/{filename}")
        raise HTTPException(400, "Invalid filename")
    path = os.path.join(JOBS_DIR, job_id, "outputs", filename)
    if not os.path.exists(path):
        logger.debug(f"Clip not found: {path}")
        raise HTTPException(404, "Clip not found")
    logger.debug(f"Serving clip: {path}")
    return FileResponse(path, media_type="video/mp4", filename=filename)
