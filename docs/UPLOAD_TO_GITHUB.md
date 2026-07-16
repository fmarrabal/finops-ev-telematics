# Upload or replace the existing GitHub repository

Target repository: `fmarrabal/finops-ev-telematics`.

## Recommended branch and pull-request workflow

Clone the current repository and create a dedicated branch:

```bash
git clone https://github.com/fmarrabal/finops-ev-telematics.git
cd finops-ev-telematics
git checkout -b r2-reproducibility
```

Copy the contents of this package into the clone, preserving the clone's `.git`
directory. On Linux or macOS, from the repository directory:

```bash
rsync -av --delete --exclude .git /path/to/finops-ev-telematics-reproducible-r2/ ./
```

On Windows PowerShell, copy the extracted package contents into the clone and
then run:

```powershell
Copy-Item -Path "C:\path\to\finops-ev-telematics-reproducible-r2\*" `
          -Destination "." -Recurse -Force
```

Validate before committing:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
python reproduce.py
pytest -q
```

Commit and push:

```bash
git add -A
git commit -m "Add R2 reproducibility package: common-fold benchmark, ITS, and uncertainty"
git push -u origin r2-reproducibility
```

Open a pull request and merge only after the `reproduce-paper-results` GitHub
Actions workflow has passed. The workflow uploads all generated tables and
figures as a downloadable Actions artifact.

## Direct replacement of `main`

Use this only after creating a local backup or tag:

```bash
git checkout main
git pull --ff-only
git tag pre-r2-reproducibility
git push origin pre-r2-reproducibility
rsync -av --delete --exclude .git /path/to/finops-ev-telematics-reproducible-r2/ ./
git add -A
git commit -m "Release R2 reproducibility package"
git push origin main
```
