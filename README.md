# VisionGuard

[![CI](https://github.com/halfsleeper2119/22l-6761_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/halfsleeper2119/22l-6761_A2_Ops/actions/workflows/ci.yml)

Web service that classifies an uploaded image as AI-generated or real.
TensorFlow/Keras model, experiments tracked with MLflow on DagsHub, Flask UI,
PostgreSQL request log, Docker Compose, cloud-hosted model API.

Run everything with one command (needs Docker and a `.env` copied from `.env.example`, plus the
champion model in `models/` from `python scripts/registry.py export`):

```
docker compose up -d --build      # then open http://localhost:8000
```

Step-by-step instructions: see `GUIDE.md`.

## Layout

```
visionguard/          shared Python package (config, preprocessing, later: model + inference)
scripts/              one script per step (check_env, audit_dataset, eda, make_splits, ...)
data/raw/             Kaggle images (tracked by DVC, not Git)
data/splits/          train / val / test lists (small CSVs, tracked by Git)
reports/figures/      figures for the PDF report
reports/tables/       audit and split tables
```
