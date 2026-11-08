"""Start the model API with waitress (debug OFF).

  python run_api.py            -> http://127.0.0.1:8001   (in Docker / cloud: HOST=0.0.0.0)
"""
import os

from dotenv import load_dotenv

load_dotenv()

from waitress import serve  # noqa: E402

from model_api import create_api  # noqa: E402

if __name__ == "__main__":
    app = create_api()
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("API_PORT", os.getenv("PORT", "8001")))
    print(f"VisionGuard model API on http://{host}:{port}  (debug={app.debug})")
    serve(app, host=host, port=port, threads=4)
