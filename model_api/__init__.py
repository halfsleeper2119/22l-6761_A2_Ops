"""Task 7 - the model API that is deployed to the cloud.

  GET  /health    -> {"status": "ok", "model_version": ..., ...}     (no key needed)
  POST /predict   multipart field "image"; header X-API-Key if API_KEY is set
                  -> {"prediction": {label, label_text, p_ai, confidence, threshold, ...}}

Environment: MODEL_PATH, THRESHOLD, MODEL_VERSION, API_KEY, MAX_UPLOAD_MB.
The champion model file is loaded once at start-up from MODEL_PATH (baked into the image),
so the API never contacts DagsHub while serving.
"""
import hmac
import os

from flask import Flask, jsonify, request

from visionguard.validation import UploadError, check_decodable, check_extension, clean_filename


def create_api(predictor=None) -> Flask:
    app = Flask(__name__)
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "5"))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    api_key = os.getenv("API_KEY", "")
    if predictor is None:
        from visionguard.inference import Predictor
        predictor = Predictor()

    def err(message, status):
        return jsonify({"error": message}), status

    @app.get("/")
    def root():
        return jsonify({"service": "VisionGuard model API", "endpoints": ["GET /health", "POST /predict"]})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_name": predictor.model_name,
                        "model_version": predictor.model_version, "threshold": predictor.threshold,
                        "input_size": getattr(predictor, "img_size", None), "auth_required": bool(api_key)})

    @app.post("/predict")
    def predict():
        # Constant-time comparison so the key cannot be guessed from response timing.
        if api_key and not hmac.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            return err("missing or invalid API key", 401)
        file = request.files.get("image")
        if file is None or not file.filename:
            return err("send the image as multipart form field 'image'", 400)
        name = clean_filename(file.filename)
        data = file.read()
        try:
            check_extension(name)
            check_decodable(data)
        except UploadError as e:
            return err(e.message, e.status)
        pred = predictor.predict(data)
        return jsonify({"filename": name, "prediction": pred.to_dict()})

    @app.errorhandler(413)
    def too_large(_):
        return err(f"file larger than {max_mb:g} MB", 413)

    @app.errorhandler(404)
    def not_found(_):
        return err("not found", 404)

    @app.errorhandler(405)
    def bad_method(_):
        return err("method not allowed", 405)

    return app
