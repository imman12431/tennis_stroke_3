# Deployment Guide

This project is split into two deployable pieces:

- **`backend/`** — FastAPI + Docker, deployed to a **Hugging Face Space** (the ML/compute).
- **`frontend/`** — Next.js, deployed to **Vercel** (the UI).

The Vercel frontend calls the Hugging Face backend over HTTP. Only the backend
does heavy work; Vercel just renders the UI and forwards requests.

---

## 1. Deploy the backend → Hugging Face Space

1. Create a new Space at https://huggingface.co/new-space
   - **SDK:** Docker
   - **Hardware:** CPU basic (free) is enough for the lite model.
2. Push the **contents of `backend/`** to the Space's git repo (the Space repo
   root must contain `Dockerfile`, `main.py`, `README.md`, `pyproject.toml`,
   `uv.lock`, `models/`, `demos/`, `pose_landmarker_lite.task`):

   ```bash
   # one-time
   git clone https://huggingface.co/spaces/<you>/tennis-backhand-detector hf-space
   cp -R backend/* backend/.dockerignore hf-space/
   cd hf-space
   git add -A && git commit -m "Deploy backend" && git push
   ```

   The Space's `README.md` already has the required YAML header
   (`sdk: docker`, `app_port: 7860`).
3. HF builds the Dockerfile (Linux x86_64 — the mediapipe/TensorFlow wheels
   resolve correctly there) and serves the API at:
   `https://<you>-tennis-backhand-detector.hf.space`
4. Verify: open `https://<that-url>/health` → should return `{"status":"ok"}`.

### Backend env vars (Space → Settings → Variables)
- `ALLOWED_ORIGINS` — set to your Vercel URL, e.g.
  `https://tennis-backhand.vercel.app` (comma-separate multiple). **Required**
  or the browser will block frontend calls (CORS).
- `JOBS_DIR` — optional; defaults to `/tmp/tennis_jobs` (writable on HF).

---

## 2. Deploy the frontend → Vercel

1. Push this repo to GitHub (the `frontend/` folder).
2. In Vercel: **New Project → import the repo → set Root Directory to `frontend`.**
   Framework preset auto-detects Next.js.
3. Add an environment variable:
   - `NEXT_PUBLIC_API_URL` = your HF Space URL (no trailing slash), e.g.
     `https://<you>-tennis-backhand-detector.hf.space`
4. Deploy. Vercel's free Hobby tier is sufficient (UI only).
5. Copy the resulting Vercel URL back into the backend's `ALLOWED_ORIGINS`.

---

## 3. Keep the Space warm (uptime pinger)

Free HF Spaces sleep after ~48h idle and cold-start (30–90s) on the next hit.
The frontend already shows a "Warming up the model…" state, but to avoid cold
starts entirely, ping `/health` on a schedule:

**Option A — UptimeRobot (recommended, zero code)**
- Create a free monitor, type HTTP(s), URL `https://<space-url>/health`,
  interval 5 min. Also alerts you if the backend goes down.

**Option B — GitHub Actions cron** (lives in the repo)
- `.github/workflows/ping.yml`:
  ```yaml
  name: ping-backend
  on:
    schedule: [{ cron: "*/10 * * * *" }]   # every 10 min
  jobs:
    ping:
      runs-on: ubuntu-latest
      steps:
        - run: curl -fsS https://<space-url>/health
  ```

---

## Local development

**Backend** (needs Docker, builds for the Linux target):
```bash
docker build --platform linux/amd64 -t tennis-api backend
docker run --rm -p 7860:7860 -e ALLOWED_ORIGINS=http://localhost:3000 tennis-api
# → http://localhost:7860/health
```
Or without Docker via uv (note: on Apple Silicon, TensorFlow 2.13 has no native
wheel, so Docker is the reliable local path):
```bash
cd backend && uv sync --frozen && uv run uvicorn main:app --port 7860
```

**Frontend:**
```bash
cd frontend
cp .env.local.example .env.local   # points at http://localhost:7860
npm install
npm run dev                         # → http://localhost:3000
```

---

## Notes / limits
- Upload cap: **60 MB MP4** (enforced in `main.py`). Long videos use a lot of
  RAM (frames are buffered) — keep demo clips short.
- Detection is **serialized** (one job at a time) so the small free container
  isn't overwhelmed.
- Job state is **in-memory** and cleared after 1 hour / on restart — fine for a
  demo, just re-run if the Space restarts mid-job.
