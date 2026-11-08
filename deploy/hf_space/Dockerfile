# Task 7 - model API image for Hugging Face Spaces (Docker SDK).
# Same recipe as the project Dockerfile, but: starts the model API, listens on 8001, and the
# champion model is COPIED in (a Space has no host folder to mount it from).
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 HOST=0.0.0.0 PORT=8001 MODEL_PATH=/app/models/champion.keras
# Spaces run the container as user id 1000.
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app visionguard/ visionguard/
COPY --chown=app:app model_api/ model_api/
COPY --chown=app:app run_api.py ./
COPY --chown=app:app models/ models/
USER app
EXPOSE 8001
CMD ["python", "run_api.py"]
