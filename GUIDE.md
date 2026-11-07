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

## When Task 1 is done

Send me:

1. The output of `check_env.py`, so the Docker image can use matching versions.
2. The audit, split and EDA console output, so we can write the interpretations.

Task 2 (tf.data pipeline and models) comes next. Start it while you write up Task 1.
