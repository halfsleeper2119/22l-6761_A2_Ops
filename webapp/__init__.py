"""Task 4 / 5 / 8 - VisionGuard web app (Flask).

Pages:  GET /          upload form
        POST /predict  validate -> predict -> log to DB -> result page
        GET /history   every past prediction, from the database
        GET /health    JSON status (used by Docker's health check)

All settings come from environment variables (Task 4.4), see .env.example:
  INFERENCE_MODE   local (load the model in this process) | remote (call the model API)
  MODEL_PATH, THRESHOLD, MODEL_VERSION       local mode
  MODEL_API_URL, MODEL_API_KEY               remote mode
  DATABASE_URL     e.g. postgresql+psycopg://user:pass@db:5432/visionguard
  MAX_UPLOAD_MB    upload size limit (default 5)
"""
import base64
import io
import os
from datetime import timezone
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template, request
from flask.wrappers import Request

from visionguard.config import ROOT
from visionguard.preprocess import open_rgb
from visionguard.validation import UploadError, check_decodable, check_extension, clean_filename
from webapp.db import Database


class InMemoryRequest(Request):
    """Werkzeug normally spools uploads larger than 500 KB to a temporary file on disk.
    Returning a BytesIO keeps every upload in memory only (Task 4.2: no image on disk)."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def _thumbnail_data_uri(data: bytes) -> str:
    """Small preview embedded in the result page itself, so nothing is written to disk."""
    im = open_rgb(data)
    im.thumbnail((360, 360))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def create_app(predictor=None, database_url=None) -> Flask:
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "5"))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)   # larger bodies -> 413 before reading
    display_tz = ZoneInfo(os.getenv("DISPLAY_TZ", "Asia/Karachi"))

    mode = os.getenv("INFERENCE_MODE", "local").lower()
    if predictor is None:
        if mode == "remote":
            from webapp.remote import RemotePredictor
            predictor = RemotePredictor(os.getenv("MODEL_API_URL", ""), os.getenv("MODEL_API_KEY", ""))
        else:
            from visionguard.inference import Predictor
            predictor = Predictor()          # loaded ONCE, here, at start-up
    source = getattr(predictor, "source", "local")
    db = Database(database_url or os.getenv("DATABASE_URL") or f"sqlite:///{ROOT / 'visionguard.db'}")

    @app.context_processor
    def _globals():
        return {"max_mb": max_mb, "mode": mode, "source": source}

    @app.template_filter("localtime")
    def _localtime(dt):
        if dt.tzinfo is None:                # SQLite returns naive datetimes; we stored UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(display_tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    def error_page(title, message, status):
        return render_template("error.html", title=title, message=message, status=status), status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        file = request.files.get("image")
        if file is None or not file.filename:
            return error_page("No file selected", "Choose an image file before pressing Analyse.", 400)
        name = clean_filename(file.filename)
        data = file.read()
        try:
            check_extension(name)
            check_decodable(data)
        except UploadError as e:
            return error_page(e.title, e.message, e.status)
        try:
            pred = predictor.predict(data, name) if mode == "remote" else predictor.predict(data)
        except Exception as exc:  # noqa: BLE001  (remote API down, etc.)
            app.logger.exception("prediction failed")
            return error_page("Prediction service unavailable",
                              f"The model could not be reached. Please try again later. ({exc})", 503)
        row = db.log(pred, name, source)
        return render_template("result.html", p=pred, row=row, filename=name,
                               preview=_thumbnail_data_uri(data))

    @app.get("/history")
    def history():
        return render_template("history.html", rows=db.recent(200), stats=db.stats())

    @app.get("/health")
    def health():
        db_ok = db.ping()
        body = {"status": "ok" if db_ok else "degraded", "database": "ok" if db_ok else "unreachable",
                "inference_mode": mode, "inference_source": source,
                "model_version": getattr(predictor, "model_version", "unknown"),
                "max_upload_mb": max_mb}
        if mode == "remote":
            body["model_api"] = predictor.health()
            if not body["model_api"].get("reachable"):
                body["status"] = "degraded"
        return jsonify(body), (200 if body["status"] == "ok" else 503)

    @app.errorhandler(413)
    def too_large(_):
        return error_page("File too large", f"The upload limit is {max_mb:g} MB. "
                          "Please resize or compress the image.", 413)

    @app.errorhandler(404)
    def not_found(_):
        return error_page("Page not found", "That page does not exist.", 404)

    @app.errorhandler(405)
    def bad_method(_):
        return error_page("Not allowed", "Use the upload form on the home page.", 405)

    @app.errorhandler(500)
    def server_error(_):
        return error_page("Something went wrong", "An unexpected error occurred. It has been logged.", 500)

    return app
