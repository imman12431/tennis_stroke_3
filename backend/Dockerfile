# uv-managed Python 3.11 image (TensorFlow 2.13 caps at 3.11)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# System deps: ffmpeg (clip conversion) + libgl1 (OpenCV runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV TF_USE_LEGACY_KERAS=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    JOBS_DIR=/tmp/tennis_jobs

# Install dependencies first (cached layer) from the locked manifest.
# A cache mount keeps downloaded wheels across builds so retries don't re-fetch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# App code + model assets.
COPY . .

# uv installs into .venv — put it on PATH so `uvicorn` resolves.
ENV PATH="/app/.venv/bin:$PATH"

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
