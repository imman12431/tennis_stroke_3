# 🎾 Tennis Backhand Detector

A full-stack ML application that detects and extracts tennis backhand shots from match videos using pose-based classification.

## Project Structure

```
tennis_stroke_3/
├── backend/                  # FastAPI backend (Hugging Face Spaces)
│   ├── api/                  # API endpoints
│   │   └── main.py
│   ├── core/                 # ML/detection logic
│   │   └── detector.py
│   ├── models/               # Pre-trained models
│   ├── demos/                # Demo videos for testing
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── README.md
│
├── frontend/                 # Next.js frontend (Vercel)
│   ├── app/                  # Next.js app router
│   ├── components/           # React components
│   ├── lib/                  # Utilities & API client
│   ├── types/                # TypeScript types
│   └── package.json
│
├── docs/                     # Documentation
│   └── DEPLOY.md             # Deployment guide
│
├── data/                     # Training data & samples
│   ├── backhand_clips/
│   ├── verified_backhands/
│   └── tennis_stroke_3.ipynb # Training notebook
│
└── legacy/                   # Old/unused code
    └── (Streamlit app, old models)
```

## Quick Start

**Backend:**
```bash
cd backend
uv sync --frozen
uvicorn api.main:app --reload --port 7860
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev  # http://localhost:3000
```

## Deployment

See [`docs/DEPLOY.md`](docs/DEPLOY.md) for step-by-step instructions for deploying to Hugging Face Spaces and Vercel.

## Architecture

- **Backend**: FastAPI + TensorFlow/Keras + MediaPipe + OpenCV
  - Pose-based classification pipeline
  - Async job queue with background workers
  - Multi-threaded video processing

- **Frontend**: Next.js + React + TypeScript
  - Real-time polling with exponential backoff retry
  - Cold-start handling for Spaces warm-up
  - Responsive UI with dark theme

## Key Features

- 📹 Upload videos or use demo footage
- 🎯 Detects tennis backhand shots using pose landmarks
- 🎬 Exports detected shots as downloadable clips
- ⚡ Asynchronous processing with live progress logs
- 🔄 Automatic retry on transient network errors
- ♿ Accessible, responsive design
