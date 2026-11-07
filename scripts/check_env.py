"""Task 1.1 - Record exact library versions and whether TensorFlow sees a GPU.

Run:  python scripts/check_env.py
Screenshot the terminal output for the report. A full `pip freeze` is also saved to
reports/env/pip_freeze.txt (reused later as the package-list artifact in MLflow).
"""
import getpass
import importlib
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import ENV_DIR  # noqa: E402

PACKAGES = [
    ("TensorFlow", "tensorflow"),
    ("Keras", "keras"),
    ("Keras Tuner", "keras_tuner"),
    ("MLflow", "mlflow"),
    ("DagsHub", "dagshub"),
    ("DVC", "dvc"),
    ("NumPy", "numpy"),
    ("pandas", "pandas"),
    ("Pillow", "PIL"),
    ("Matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("Flask", "flask"),
]


def version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # report, don't crash
        return f"NOT INSTALLED ({type(exc).__name__})"
    return getattr(module, "__version__", "installed (no __version__)")


def main():
    print("=" * 60)
    print(f"VisionGuard environment check  |  user: {getpass.getuser()}  |  {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 60)
    print(f"{'Python':<14}: {sys.version.split()[0]}  ({sys.executable})")
    print(f"{'OS':<14}: {platform.system()} {platform.release()} ({platform.machine()})")
    for label, module in PACKAGES:
        print(f"{label:<14}: {version_of(module)}")

    print("-" * 60)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        print(f"TF built with CUDA : {tf.test.is_built_with_cuda()}")
        print(f"GPUs visible to TF : {gpus if gpus else 'none -> training runs on CPU'}")
    except ImportError:
        print("TensorFlow not importable; cannot check GPU.")

    ENV_DIR.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    out = ENV_DIR / "pip_freeze.txt"
    out.write_text(freeze)
    print(f"Full package list saved to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
