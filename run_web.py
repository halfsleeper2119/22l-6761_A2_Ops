"""Start the VisionGuard web app with a production WSGI server (waitress), debug OFF.

  python run_web.py            -> http://127.0.0.1:8000

Why not `flask run --debug`? The debug server is single-purpose for development: it reloads
code, shows full tracebacks to anyone who triggers an error, and its interactive debugger
allows running arbitrary Python from the browser. Waitress is a multi-threaded production
server and our error pages show only friendly messages.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from waitress import serve  # noqa: E402

from webapp import create_app  # noqa: E402

if __name__ == "__main__":
    app = create_app()
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    print(f"VisionGuard web on http://{host}:{port}  (debug={app.debug})")
    serve(app, host=host, port=port, threads=4)
