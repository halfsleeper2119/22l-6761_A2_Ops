# VisionGuard image - used for BOTH the web app and the model API (different start commands).
# Task 6.2 / 6.4: slim base, multi-stage build, no pip cache, non-root user, health check.

# ---------- stage 1: build the Python environment ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Dependencies first: this layer is rebuilt only when requirements-docker.txt changes,
# so editing source code does not re-download ~600 MB of TensorFlow (layer cache).
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# ---------- stage 2: runtime image ----------
FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOST=0.0.0.0 \
    PORT=8000
# Unprivileged user: if the app is ever compromised, the attacker is not root in the container.
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# Source code last (changes most often). .dockerignore keeps .env, data, .venv, models out.
COPY --chown=app:app visionguard/ visionguard/
COPY --chown=app:app webapp/ webapp/
COPY --chown=app:app model_api/ model_api/
COPY --chown=app:app run_web.py run_api.py ./
USER app
EXPOSE 8000
# Docker marks the container unhealthy if /health stops answering 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)"
# Default: web app. The model API service overrides this with `python run_api.py`.
CMD ["python", "run_web.py"]
