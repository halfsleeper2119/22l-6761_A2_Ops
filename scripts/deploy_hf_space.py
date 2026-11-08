"""Task 7 - deploy the model API to Hugging Face Spaces (Docker) in one command.

  pip install huggingface_hub
  python scripts/deploy_hf_space.py --hf-user <your-hf-username> --token <hf-write-token>

What it does:
  1. stages exactly the files the API image needs into deploy/_build/ (code, Dockerfile,
     README with the Space settings, and the exported champion model)
  2. creates the public Space <hf-user>/22l-6761_A2_Ops (Docker SDK) if it does not exist
  3. stores API_KEY (from .env) as a Space SECRET - never in the uploaded files
  4. uploads the folder; Hugging Face then builds the Docker image in the cloud
The API URL is printed at the end: https://<hf-user>-22l-6761-a2-ops.hf.space
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_NAME = "22l-6761_A2_Ops"


def stage() -> Path:
    out = ROOT / "deploy" / "_build"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in ["Dockerfile", "README.md"]:
        shutil.copy(ROOT / "deploy" / "hf_space" / f, out / f)
    shutil.copy(ROOT / "requirements-docker.txt", out)
    shutil.copy(ROOT / "run_api.py", out)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "visionguard", out / "visionguard", ignore=ignore)
    shutil.copytree(ROOT / "model_api", out / "model_api", ignore=ignore)
    (out / "models").mkdir()
    for f in ["champion.keras", "champion.txt"]:
        src = ROOT / "models" / f
        if not src.exists():
            sys.exit(f"Missing {src}. Run: python scripts/registry.py export")
        shutil.copy(src, out / "models" / f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-user", required=True)
    p.add_argument("--token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--dry-run", action="store_true", help="only stage the files")
    a = p.parse_args()

    out = stage()
    files = sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file())
    print(f"Staged {len(files)} files in {out}:")
    for f in files:
        print("  ", f)
    if a.dry_run:
        return

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("API_KEY")
    if not a.token:
        sys.exit("Pass --token (a Hugging Face WRITE token) or set HF_TOKEN.")
    if not api_key:
        sys.exit("API_KEY is missing from .env")

    api = HfApi(token=a.token)
    repo_id = f"{a.hf_user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_secret(repo_id, "API_KEY", api_key)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(out),
                      commit_message="Deploy VisionGuard model API")
    host = f"{a.hf_user}-{SPACE_NAME}".lower().replace("_", "-")
    print(f"\nSpace page : https://huggingface.co/spaces/{repo_id}")
    print(f"API URL    : https://{host}.hf.space")
    print("The image now builds in the cloud (5-10 min). Watch the 'Logs' tab on the Space page.")


if __name__ == "__main__":
    main()
