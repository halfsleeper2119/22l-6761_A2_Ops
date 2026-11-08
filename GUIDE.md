# VisionGuard: step-by-step guide

Replace these placeholders everywhere:

- `<ROLL>`: your roll number, e.g. `22L-1234`. The project name is then `<ROLL>_A2_Ops`.
- `<gh-user>`: your GitHub username.
- `<dh-user>`: your DagsHub username.
- `<token>`: your DagsHub access token (from DagsHub > Settings > Tokens).

Commands are given for Windows PowerShell. Where Linux or macOS differs, it is shown after "Linux:".

Screenshot rule: the terminal prompt or browser address bar must be visible in every screenshot. Each screenshot step is marked 📸. Next to each one, write a single line in your notes saying what it shows.

---

## Step 0. Accounts and repository (15 min)

1. On GitHub, create an empty repository named `<ROLL>_A2_Ops`, with no README.
2. On DagsHub, choose Create + > New Repository > **Connect a repository** > GitHub, and pick `<ROLL>_A2_Ops`. DagsHub now mirrors your GitHub repo and adds DVC storage and an MLflow server to it.
3. On DagsHub, create an access token under Settings > Tokens. Keep it private.
4. Clone the GitHub repo, copy all files from this folder into it, then:

   ```
   cd <ROLL>_A2_Ops
   copy .env.example .env        # Linux: cp .env.example .env
   ```

   Fill in `.env` with your DagsHub username and token. It is git-ignored, so it never reaches GitHub.

---

## Task 1.1 Environment (15 min)

**What we are doing.** We create a virtual environment: a private folder of Python packages for this project only. Its versions then can't clash with other projects, and anyone can rebuild it from `requirements.txt`.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` first. Use Python 3.10 to 3.12.

📸 Screenshot the output of `check_env.py`.

**Report notes.**
- The script prints exact versions and saves a full `pip freeze` to `reports/env/pip_freeze.txt`. We reuse that file in Task 3.4.
- On native Windows, TensorFlow 2.11 and later cannot use an NVIDIA GPU (that needs WSL2). "GPUs visible: none" is therefore expected. CPU is enough for this dataset size.

---

## Download the dataset (10 min)

Download the zip from https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images and unzip it into `data/raw/`. You should end up with two class folders, roughly:

```
data/raw/AiArtData/...
data/raw/RealArt/...
```

Extra nesting inside those folders is fine. The class is taken from the top-level folder name (see `folder_to_class` in `visionguard/config.py`). The audit prints which folder mapped to which class, so check that line.

---

## Task 1.2 Dataset audit (10 min)

```
python scripts/audit_dataset.py
```

**What it does.** It visits every file under `data/raw`, records class, size, width, height, colour mode, format and SHA-256 hash, and saves the full table to `reports/tables/audit.csv`.

- **Readability** is checked in two passes. `verify()` checks the file structure. `load()` then decodes every pixel, which catches half-downloaded (truncated) files that `verify()` misses.
- **SHA-256** is a fingerprint of the file's bytes. Two files with the same hash are exact duplicates. The script also flags duplicates that sit in *both* class folders, since those carry a contradictory label.
- **Extension mismatch** flags files whose extension disagrees with their real format, for example a PNG saved with a `.jpg` name.

📸 Screenshot the console output. The tables in `reports/tables/audit_summary.md` can be pasted straight into the report.

Then build the splits:

```
python scripts/make_splits.py
```

This drops unreadable files, contradictory duplicates, extra duplicate copies and images under 32 px. It then makes a stratified 70/15/15 split with seed 42.

- **Why remove duplicates before splitting:** if the same image lands in both train and test, the test score is inflated. The script asserts that no hash appears in two splits.
- **Why stratified:** each split keeps the same real/AI ratio.

📸 Screenshot the cleaning steps and split table.

---

## Task 1.3 Visual EDA (10 min)

```
python scripts/eda.py
```

This produces, in `reports/figures/`:

| Figure | Shows |
|---|---|
| `samples_real.png`, `samples_ai.png` | 18 random images per class, with size and format |
| `distributions.png` | Width, height, file size (log scale) and aspect ratio, per class |
| `formats.png` | Share of each file format within each class |

The script also prints medians and format counts. Send me that output and we'll write the 2 to 3 sentence interpretation for each figure from your real numbers.

**What to look for.** In this kind of dataset the two classes often differ in *metadata*: AI images tend to be PNG, square, and of a fixed resolution (512, 768, 1024), while real photos are JPEG and of varied size. A model could learn those "shortcuts" instead of the image content. This is a strong point for your report. Our pipeline helps by decoding every file to pixels and resizing to one size, which removes format and resolution. JPEG compression traces still survive in the pixels, so we come back to this in error analysis (Task 2.6).

---

## Task 1.4 DVC (30 min)

**What DVC is.** Git is built for small text files. A dataset of hundreds of MB would bloat the repository forever, because every version stays in Git history, and GitHub rejects files over 100 MB.

DVC fixes this by storing the real data in a separate *remote storage*, here DagsHub. Git only keeps a tiny *pointer file* (`.dvc`) that holds the data's hash, size and path. Checking out an old Git commit and running `dvc checkout` restores exactly the data that commit pointed to.

### Toy part

1. On DagsHub, create a plain new repository (not connected to GitHub) named `<ROLL>_A2_Ops_DVC_Toy`.
2. Create the toy project in a folder *outside* the main project:

```
mkdir dvc-toy; cd dvc-toy
git init; git branch -M main
dvc init
git commit -m "Initialise DVC"

python -c "open('scores.csv','w').write('id,score\n1,70\n2,85\n3,90\n')"
dvc add scores.csv
git add scores.csv.dvc .gitignore
git commit -m "v1: scores.csv with 3 rows"
type scores.csv.dvc                      # Linux: cat scores.csv.dvc
```
📸 Screenshot the pointer file contents.

```
python -c "open('scores.csv','a').write('4,60\n5,95\n')"
dvc add scores.csv
git add scores.csv.dvc
git commit -m "v2: scores.csv with 5 rows"
git log --oneline
```
📸 Screenshot both commits.

Go back to v1, then return to v2:
```
git checkout HEAD~1 -- scores.csv.dvc
dvc checkout
type scores.csv                          # shows 3 rows

git checkout main -- scores.csv.dvc
dvc checkout
type scores.csv                          # shows 5 rows again
```
📸 Screenshot this.

Set up the remote, push, delete the local data and pull it back:
```
dvc remote add -d origin https://dagshub.com/<dh-user>/<ROLL>_A2_Ops_DVC_Toy.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dh-user>
dvc remote modify origin --local password <token>
git add .dvc/config
git commit -m "Add DagsHub DVC remote"

git remote add origin https://dagshub.com/<dh-user>/<ROLL>_A2_Ops_DVC_Toy.git
git push -u origin main                  # username: <dh-user>, password: <token>
dvc push --all-commits                   # uploads both versions

Remove-Item scores.csv; Remove-Item -Recurse -Force .dvc\cache    # Linux: rm scores.csv; rm -rf .dvc/cache
dvc pull
type scores.csv
```
📸 Screenshot the push, the deletion, the pull, and the DagsHub repo page showing both commits.

**Why `--local`:** it writes your credentials to `.dvc/config.local`, which DVC git-ignores automatically. The remote URL goes into the shared `.dvc/config`, but the token never does.

### Apply to the real dataset

Back in the main project folder:
```
dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc .dvcignore
git commit -m "Track raw dataset with DVC"

dvc remote add -d origin https://dagshub.com/<dh-user>/<ROLL>_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dh-user>
dvc remote modify origin --local password <token>
git add .dvc/config
git commit -m "Add DagsHub DVC remote"
dvc push

git add .
git commit -m "Task 1: environment, audit, EDA, splits"
git push
```
📸 Screenshot `data/raw.dvc`, the `dvc push` output, and the DagsHub page showing the data folder.

**Report notes.**
- `data/raw.dvc` holds an `md5` hash ending in `.dir`, the total `size` and `nfiles`. The `.dir` suffix means "this is a folder": the hash points to a manifest listing every file's hash.
- `dvc add` also writes `data/.gitignore`, so Git ignores the actual images.
- The split lists stay in Git because they are small text files that define the experiment.

---

---

# Tasks 2 and 3: models and experiment tracking

## Step A. Switch MLflow to 2.x (2 min)

`requirements.txt` now pins `mlflow>=2.19,<3`. MLflow 3 logs models through new server endpoints that hosted servers such as DagsHub may not support, and 2.x is the safe choice. With `.venv` active:

```
pip install -r requirements.txt
```

This downgrades MLflow, and pandas with it. That is expected.

## Step B. Quick evidence scripts (15 min)

```
python scripts\show_batch.py                      # Task 2.1 figure: reports\figures\augmented_batch.png
python scripts\toy_keras_tuner.py                 # Task 2.7 toy: Fashion-MNIST tuner table
python scripts\toy_mlflow.py --lr 0.001 --local   # Task 3.1: two local runs
python scripts\toy_mlflow.py --lr 0.01 --local
mlflow ui --backend-store-uri ./mlruns            # open http://127.0.0.1:5000, Ctrl+C to stop
```

📸 In the MLflow UI, tick both runs and click **Compare**. Screenshot the run list, one run page, and the comparison view.

Next, the same toy run on DagsHub (Task 3.2). Fill in `.env` first:

```
python scripts\toy_mlflow.py --lr 0.001
git check-ignore -v .env                          # 📸 proof that .env is git-ignored
```

📸 Screenshot the toy run on DagsHub: Experiments tab, then the run.

## Step C. All training runs (40 to 80 min, can run unattended)

```
.\scripts\run_experiments.ps1
```

It runs 8 trainings in a row and logs each one to DagsHub. Three of them are registered in the model registry. At the end it prints a comparison table, also saved to `reports\tables\runs_comparison.md`. Every run's figures go to `reports\runs\<run-name>\`:

| File | Used for |
|---|---|
| `curves.png` | 2.3 / 2.4: loss and accuracy curves, with the best epoch and the fine-tuning start marked |
| `model_summary.txt` | 2.2: output shape and parameter count per layer |
| `trainable_layers*.txt` | 2.3: trainable layers in stage 1 and stage 2 |
| `confusion_matrix.png`, `roc_curve.png`, `metrics.json` | 2.5 |
| `misclassified.png` | 2.6: the 10 most confident mistakes |

What each run changes, against its neighbour:

| Run | Change |
|---|---|
| cnn_lr1e-3 | Baseline CNN |
| cnn_lr3e-4 | Learning rate |
| cnn_noaug | No augmentation |
| cnn_autolog | Same as baseline, but logged with MLflow autolog (Task 3.3) |
| mnv2_160 | MobileNetV2 transfer learning, frozen then last block fine-tuned |
| mnv2_160_frozen | Backbone never unfrozen |
| mnv2_128 | Image size 128 |
| effb0_160 | EfficientNetB0 backbone |

Then the tuner (Task 2.7, 20 to 40 min):
```
python scripts\tune.py
```

## Step D. Registry (10 min)

```
python scripts\registry.py list                   # versions with their test scores
python scripts\registry.py promote --version <N>  # N = the best run (by validation AUC)
python scripts\predict_champion.py data\raw\<some folder>\<some image>
python scripts\registry.py export                 # writes models\champion.keras for serving
```

📸 Screenshot the DagsHub Models page showing the versions and the alias, and the terminal output of `predict_champion.py`.

## Design decisions to explain in the report

- **Image size 160:** MobileNetV2 has ImageNet weights at 160, and it is about twice as fast as 224 on CPU.
- **Aspect ratio:** every image is squashed to 160x160, so aspect ratio is not kept. We deliberately avoid padding. AI images here are mostly square and real photos are not, so padding bars would appear almost only on real photos and give the model a shortcut.
- **Pillow decoding inside tf.data:** we decode with Pillow through `tf.numpy_function`. It handles WebP, animated GIF, transparency and EXIF rotation, and it is the same function the API will use (Task 4.1).
- **Pipeline stages:** decode → resize → cache → shuffle → batch → augment (training only) → prefetch.
- **Augmentation:** flip, ±18° rotation, 10% zoom and 10% contrast. We avoid blur, noise and re-compression, because those change exactly the traces that separate AI images from real ones.
- **Scaling lives inside the model:** pixel scaling is the model's first layer (`Rescaling`). For MobileNetV2 that layer is exactly `preprocess_input`, mapping pixels to [-1, 1], because the backbone was trained on that range. EfficientNet does its own scaling internally. The `.keras` file is therefore self-contained.
- **Loss and output:** binary cross-entropy with one sigmoid output gives the probability that an image is AI. We use Adam and track accuracy, AUC, precision and recall.
- **Fine-tuning:** stage 2 unfreezes only the last block, at a learning rate about 100x smaller. BatchNorm layers stay frozen, so roughly 700 images can't overwrite the statistics they learned on ImageNet.
- **Callbacks:** EarlyStopping (patience 5, restores the best weights), ModelCheckpoint (best `val_loss`), and ReduceLROnPlateau (halves the learning rate after 2 flat epochs). Seeds are fixed with `keras.utils.set_random_seed(42)`.
- **Threshold:** chosen on **validation** data, never on test. Calling a real photo fake is the worse error, because it wrongly accuses a genuine photo. So we pick the threshold that catches the most AI images while flagging at most 10% of real photos (`--max-fpr`). The test set is evaluated once, with that threshold.
- **Manual logging vs autolog (Task 3.3):**
  - Our manual callback logs exactly what we choose: our parameters, per-epoch metrics including the learning rate, and our tags.
  - Autolog captures more without any code: every `fit()` and optimizer setting, EarlyStopping details, model summary, TensorBoard logs and checkpoints.
  - Autolog misses our own parameters (augmentation, image size, backbone), the git and DVC tags, and anything that happens after `fit()`. Test metrics and the threshold only appear because our code logs them.

## When Tasks 2 and 3 are done

Send me `reports\tables\runs_comparison.md`, the tuner table and the champion run's `metrics.json` (it lists the misclassified images). We'll write the evaluation, error analysis and comparison paragraphs from them, then build the Flask app.

---

# Tasks 4 and 5: Flask app and database

## Step E. Install and configure (5 min)

```
pip install -r requirements.txt
python -m pytest -q
```
14 tests should pass. They use a fake model, so they will also run in CI later.

Add these lines to `.env`. Leave `THRESHOLD` and `MODEL_VERSION` empty, so the app reads them from `models\champion.txt`, which `registry.py export` wrote:
```
INFERENCE_MODE=local
MODEL_PATH=models/champion.keras
THRESHOLD=
MODEL_VERSION=
DATABASE_URL=sqlite:///visionguard.db
MAX_UPLOAD_MB=5
PORT=8000
```

## Step F. Run it

```
python run_web.py
```
Open http://127.0.0.1:8000.

📸 Screenshots to take:
- **4.1:** the console line `[inference] loaded champion.keras ... in X s`. It appears once at start-up, not per request. Also take two results: one AI image and one real image from the test split (see `data\splits\test.csv`).
- **4.2:** the three failure pages: a `.txt` file renamed to `.png` gives "Image could not be read"; a `.pdf` or `.txt` gives "Unsupported file type"; an image over 5 MB gives "File too large".
- **4.3:** the upload page, a result page, and the History page.
- **4.4:** http://127.0.0.1:8000/health, plus the console line showing `debug=False`.
- **5:** the History page after 5 or more uploads.

## Design points for the report

- **One shared inference module** (`visionguard/inference.py`). It loads the model **once** at start-up. Loading per request would re-read the file and rebuild the TensorFlow graph on every upload, which takes about 1 second each time and wastes memory.
- **Same preprocessing as training.** Both use `preprocess.load_for_model`, and pixel scaling is a layer inside the model, so serving can't drift from training.
- **Three validation checks**, because each one alone can be fooled:
  - the file extension
  - the size limit, enforced by `MAX_CONTENT_LENGTH` before the body is read
  - a real decode with Pillow, which catches renamed or truncated files
- **Filename sanitizing:** `secure_filename` turns a name like `../../etc/passwd.png` into `etc_passwd.png`.
- **Nothing on disk:** a custom `InMemoryRequest` stops Werkzeug from spooling uploads larger than 500 KB to temporary files. The preview is embedded in the result page as base64.
- **Configuration and production serving:** every setting comes from environment variables. The app runs under **waitress** with debug off, because Flask's debug server exposes an interactive debugger and full tracebacks to anyone.
- **Database:** SQLAlchemy, with `DATABASE_URL` choosing SQLite locally or PostgreSQL in Docker. Each row stores the time (UTC), sanitized filename, verdict, confidence, P(AI), threshold, model name and version, which server answered, and latency.
- **`INFERENCE_MODE=remote`** makes the same app call the cloud model API instead (Task 8). The model API is `run_api.py` / `model_api/`.

---

# Task 6: Docker, Compose and CI

Install **Docker Desktop**, start it, and wait for "Engine running". Check with `docker version`.

## 6.1 Warm-up with a ready-made image (15 min)

📸 Take a screenshot after each block.
```
docker run -d --name web1 -p 8080:80 nginx:alpine           # run + port mapping; open http://localhost:8080
mkdir site; "<h1>Hello from a volume - 22l-6761</h1>" | Set-Content site\index.html
docker run -d --name web2 -p 8081:80 -v ${PWD}\site:/usr/share/nginx/html:ro nginx:alpine   # volume; open :8081
docker logs web1                                             # logs (refresh :8080 first)
docker exec -it web1 sh                                      # shell inside; try: ls /etc/nginx ; whoami ; exit
docker ps -a                                                 # list containers
docker stop web1 web2; docker rm web1 web2                   # remove containers
docker images; docker rmi nginx:alpine                       # list and remove images
Remove-Item -Recurse site
```

Definitions, in your own words:
- **Image:** a read-only template (OS files, libraries, app).
- **Container:** a running instance of an image.
- **Layer:** one cached filesystem step of an image, one per Dockerfile instruction.
- **Volume:** storage that lives outside the container and survives its removal.
- **Port mapping:** host port → container port, so the outside world can reach the app.

## 6.2 to 6.5 The project stack

Add the Postgres settings and an API key to `.env`. Choose your own password; it's only for your local database:
```
@"

POSTGRES_USER=visionguard
POSTGRES_PASSWORD=ChooseAStrongPassword123
POSTGRES_DB=visionguard
API_KEY=PasteALongRandomStringHere
"@ | Add-Content -Encoding ascii .env
```
For a random API key, run: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

Build and start (the first build downloads TensorFlow, so it takes 5 to 10 min):
```
docker compose up -d --build
docker compose ps                                   # 📸 web "healthy" and db "healthy" (wait ~1 min)
```
Open http://localhost:8000 and upload a few images. The History page now reads from **PostgreSQL**.

📸 Evidence:
```
docker compose exec web whoami                      # -> app   (non-root user)
docker compose exec web id
Test-NetConnection localhost -Port 5432             # TcpTestSucceeded : False  -> DB port closed to the outside
docker compose exec db psql -U visionguard -c "SELECT id, created_at, label, confidence, model_version FROM predictions;"
```

**6.4 Image size:**
```
docker image ls visionguard                          # 📸 AFTER size
docker history visionguard:latest                    # 📸 which layers are largest (the pip install layer)
docker build -f docker/Dockerfile.naive -t visionguard:naive .
docker image ls visionguard                          # 📸 naive (BEFORE) next to latest (AFTER)
docker rmi visionguard:naive
```
For the report: the BEFORE image uses the full `python:3.12` base, the full dev requirements (with full `tensorflow`, mlflow, dvc, matplotlib) and keeps the pip cache. The AFTER image uses `python:3.12-slim`, serving-only packages with `tensorflow-cpu`, a multi-stage build, no pip cache and a `.dockerignore`.

**6.5 Operations:**
```
docker compose stop; docker compose start; docker compose restart web
docker compose logs --tail 30 web
docker compose logs --tail 30 db
docker compose exec web sh          # look around: ls /app ; exit
docker compose exec db sh           # exit
```
Break something on purpose (missing model file), then diagnose it from the logs only:
```
Rename-Item models\champion.keras champion.keras.bak
docker compose restart web
docker compose ps                   # web keeps restarting / unhealthy
docker compose logs --tail 20 web   # 📸 FileNotFoundError: Model file not found: /app/models/champion.keras
Rename-Item models\champion.keras.bak champion.keras
docker compose restart web
docker compose ps                   # 📸 healthy again
```

Design points for the report:
- **Instruction order:** dependencies are installed before the source code is copied. That `pip install` layer (about 600 MB) is reused from cache whenever only code changes, so a rebuild takes seconds.
- **No secrets in the image:** `.dockerignore` excludes `.env`, and passwords reach the containers only through Compose `environment:` at run time. Each service gets only the variables it needs, so the DagsHub token never enters a container.
- **Model mounted, not built in:** it is mounted read-only from `./models`. The model is versioned in the MLflow registry rather than in Git, and swapping champions needs only `registry.py export` plus a restart, with no rebuild. CI can also build the image without the model file.
- **Database wiring:** `db` publishes no port. `web` reaches it by its service name (`@db:5432`) on Compose's internal network, and `depends_on: condition: service_healthy` waits for `pg_isready`.
- **`tensorflow-cpu`:** the same API without the CUDA/GPU libraries, because the servers we deploy to have no GPU.

## 6.6 GitHub Actions

**Toy:**
1. Create a new empty GitHub repo, e.g. `22l-6761_A2_Ops_ci_toy`.
2. Copy the *contents* of the `ci-toy` folder into it (`calc.py`, `test_calc.py`, `.github/workflows/ci.yml`), commit and push. The repo's **Actions** tab shows a green run for both Python versions. 📸
3. Break it: change `return a + b` to `return a - b`, push. 📸 Red run.
4. Fix it back and push. 📸 Green run.

**Project:** `.github/workflows/ci.yml` is already in your repo. Every push runs ruff, then pytest, then `docker build`. 📸 Screenshot the green run, and the badge at the top of `README.md` on GitHub.

**Secrets:** workflow secrets are stored in the repo under **Settings → Secrets and variables → Actions** and used as `${{ secrets.NAME }}`. Writing them in the workflow file would publish them to anyone who can read the repo, and they would stay in Git history forever. Our CI needs no secrets.

---

# Task 7: deploy the model API (Hugging Face Spaces, Pathway C)

**Why this platform:** it's free and needs no card. It builds our Dockerfile in the cloud and gives a public HTTPS URL. Our model API is one container, which a Docker Space runs directly.

| Compose element | Spaces equivalent |
|---|---|
| `api` service | the Space's container, built from `deploy/hf_space/Dockerfile` |
| `ports: 8001` | `app_port: 8001` in the Space README; Hugging Face puts HTTPS on port 443 in front |
| `.env` API_KEY | a Space **secret** (Settings → Variables and secrets) |
| `./models` volume | the model is copied into the image, because a Space has no host folder to mount |
| `db` | not needed in the cloud: the local Flask app (Task 8) logs to its own database |

1. Create an account at https://huggingface.co, then a **Write** token at https://huggingface.co/settings/tokens.
2. Deploy:
   ```
   pip install huggingface_hub
   python scripts\deploy_hf_space.py --hf-user <your-hf-username> --token <hf-write-token>
   ```
   📸 The script output shows the staged files, the Space page and the API URL.
3. Open the Space page and its **Logs** tab. Wait until the build finishes and the log shows `VisionGuard model API on http://0.0.0.0:8001`, and the status shows **Running** (5–10 min). 📸 the build log, the Running status, and Settings → Variables and secrets showing `API_KEY` as a secret, with its value hidden.
4. Open `https://<api-url>/health` in a browser. 📸

**Postman tests** (📸 each response, with the status code and time visible):

| Test | Request | Expected |
|---|---|---|
| Health | `GET {{url}}/health` | 200, model version 4, threshold 0.632 |
| Real image | `POST {{url}}/predict`; Headers `X-API-Key: <API_KEY>`; Body → form-data, key `image` (type **File**) | 200, a prediction JSON |
| AI image | same, with an AI test image | 200 |
| No key | same, without the header | 401 `missing or invalid API key` |
| Wrong type | form-data `image` = a `.txt` file | 415 |

For a mobile-data test (a different network): turn on your phone hotspot, connect the laptop, and repeat one request. Or open `/health` in your phone's browser.

# Task 8: local Flask calls the cloud API

In `.env`, change these lines (`code .env`):
```
INFERENCE_MODE=remote
MODEL_API_URL=https://<your-api-url>
MODEL_API_KEY=<same value as API_KEY>
```
Then run `python run_web.py`. The startup log no longer loads a model, and the page footer shows `inference: remote (<your-api-host>)`.

Upload at least 5 images, a mix of real and AI. 📸 each result: "Served by" shows the cloud host. Then 📸 the History page, whose "Served by" column shows the cloud host, and `/health`, which now includes a `model_api` block showing `reachable: true`.

When you're done, set `INFERENCE_MODE=local` again if you want the local model back.
