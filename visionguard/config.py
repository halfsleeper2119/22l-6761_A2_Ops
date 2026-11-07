"""Central project settings. Every script imports paths and constants from here,
so a path or seed is defined in exactly one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"        # original Kaggle images (tracked by DVC)
SPLITS_DIR = ROOT / "data" / "splits"   # train/val/test lists (small CSVs, tracked by Git)
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
ENV_DIR = REPORTS / "env"

SEED = 42

# Label encoding. "ai" is the positive class (label 1): the thing we want to detect.
CLASSES = ["real", "ai"]
LABEL_OF = {name: i for i, name in enumerate(CLASSES)}

# Images smaller than this on either side carry too little signal and are dropped.
MIN_SIDE = 32

# Training defaults (each can be overridden on the command line of scripts/train.py)
IMG_SIZE = 160          # 160x160: MobileNetV2 has ImageNet weights for it; ~2x faster than 224 on CPU
BATCH_SIZE = 32
MODELS_DIR = ROOT / "models"
RUNS_DIR = REPORTS / "runs"            # local copy of every run's figures and metrics
REGISTERED_MODEL_NAME = "22l-6761_A2_Ops_visionguard"
MLFLOW_EXPERIMENT = "22l-6761_A2_Ops"

# OS junk files that are not part of the dataset.
JUNK_FILES = {".ds_store", "thumbs.db", "desktop.ini"}


def folder_to_class(folder_name: str):
    """Map a top-level dataset folder name to a class.
    The Kaggle download uses names like 'RealArt' and 'AiArtData'."""
    n = folder_name.lower()
    if "real" in n:
        return "real"
    if n.startswith("ai") or "generated" in n or "fake" in n:
        return "ai"
    return None
