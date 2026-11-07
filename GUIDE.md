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
