# GrieveAI
Explainable Near-Real-Time Hindi-English Code-Mixed Grievance Triage and Department Routing
A Case Study of VCET Student Grievance Redressal

## Repo layout

```
GrieveAI/
├── config/
│   └── taxonomy.json          # 7 categories x 28 sub-categories, priority defaults
├── data/
│   ├── raw/                   # real VCET data goes here when available (git-ignored)
│   └── processed/             # train/val/test splits after preprocessing
├── generate_synthetic_grievances.py   # bootstrap dataset generator
├── notebooks/
│   └── 01_train_classifier.ipynb      # run this in Colab (GPU required)
├── src/
│   ├── model.py                # GrieveAIClassifier: MuRIL + LoRA + 3 heads
│   ├── train.py                # CLI training script (used by the notebook)
│   ├── language_id.py          # fastText language/script detection (Week 1)
│   ├── priority_dedup.py       # priority head inference + duplicate detection (Week 3)
│   ├── explainability.py       # SHAP wrapper (Week 3)
│   └── llm_report.py           # Claude API + OpenRouter fallback (Week 4)
├── app/
│   ├── app.py                  # Flask entry point
│   ├── routes.py
│   ├── models_db.py            # SQLAlchemy models (PostgreSQL/SQLite)
│   └── templates/
├── tests/
└── requirements.txt
```

## Quick start — working prototype (local, no GPU needed)

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install flask flask_sqlalchemy joblib scikit-learn pandas numpy
python generate_synthetic_grievances.py --per_combo 15   # regenerates the dataset (optional, already in data/processed/)
python train_baseline.py                                  # trains the working baseline classifier (~seconds)
python app/app.py                                          # starts the Flask app on http://127.0.0.1:5000
```

Then, in a second terminal, submit a grievance:

```bash
curl -X POST http://127.0.0.1:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The library wifi has not worked for 3 days."}'
```

Open http://127.0.0.1:5000/ to see it appear on the dashboard.

This uses `src/baseline_classifier.py` (TF-IDF + Logistic Regression) as
the classifier — a real, working pipeline today with no GPU and no
model download. It is NOT the final model for the evaluation chapter;
it's a stand-in until the MuRIL+LoRA checkpoint from Colab training
(below) is ready to swap in. `app/routes.py` is written so that swap is
a small change, not a rewrite.

## Training the classifier (Colab, GPU required)

1. Push this repo to GitHub (see "Git setup" below).
2. Open `notebooks/01_train_classifier.ipynb` in Google Colab.
3. Runtime -> Change runtime type -> T4 GPU.
4. Run all cells. The notebook clones the repo, installs dependencies,
   mounts Google Drive for checkpoints, and calls `src/train.py`.
5. Trained adapter + head weights are saved to
   `/content/drive/MyDrive/GrieveAI_checkpoints/`.

## Git setup (do this first)

```bash
cd GrieveAI
git init
git add .
git commit -m "Initial scaffold: taxonomy, synthetic data generator, MuRIL+LoRA model/train scripts"
git branch -M main
git remote add origin https://github.com/<your-username>/GrieveAI.git
git push -u origin main
```

Create the empty repo on github.com first (no README/license, so there's
no merge conflict on first push), then run the commands above.

## Status

See the project timeline in the panel-facing plan document for the
week-by-week build schedule through the October milestone.
