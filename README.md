# VisionGuard

Web service that classifies an uploaded image as AI-generated or real.
TensorFlow/Keras model, experiments tracked with MLflow on DagsHub, Flask UI,
PostgreSQL request log, Docker Compose, cloud-hosted model API.

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
