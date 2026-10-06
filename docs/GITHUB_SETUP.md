# Safe GitHub setup (Windows PowerShell)

This repository has not been committed or pushed by the project workflow. These
commands initialize it locally and show the exact review steps before publishing.
The checked-in training CSV is explicitly synthetic; private pilot data, local
SQLite files, `.env` files, and model checkpoints must remain untracked.

Run these commands from the repository root:

```powershell
Set-Location C:\GrieveAI\GrieveAI

# Initialize once. If this repository already has a .git directory, skip this.
git init

# Inspect the complete working tree and confirm .env, databases, raw data,
# generated checkpoints, and caches do not appear as addable files.
git status --short --branch
git status --short --ignored

# Stage the intended project files. Review what was staged before committing.
git add -A
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached

# Commit only after the staged diff has been reviewed.
git commit -m "Prepare GrieveAI prototype for reproducible development"

# Select the default branch name and verify it.
git branch -M main
git branch --show-current

# Connect a GitHub repository that you created; replace the placeholder.
git remote add origin "https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git"
git remote -v

# Push only after confirming the remote and branch are correct.
git push -u origin main
```

Before `git add -A`, confirm `.env.example` contains placeholders only. Use
`git check-ignore -v .env app/instance/grieveai.db checkpoints/baseline/category_clf.joblib data/raw/example.csv`
to verify the ignore rules. A path that does not exist can still be checked by
Git. Never use `git add -f` for ignored environment files, databases, raw data,
or checkpoints. If a secret was ever committed, removing it from a later commit
does not revoke it; rotate the credential and clean repository history before
publishing.

GitHub authentication should use the supported Git Credential Manager or SSH
keys. Do not place credentials in the remote URL or commit them to this repository.
