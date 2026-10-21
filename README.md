# Chronic Disease AI

An educational Streamlit prototype for estimating a diabetes risk category from eight measurements. It is not a diagnostic device and does not provide treatment advice.

## Features

- Home page with project workflow and limitations.
- Risk estimate from a saved scikit-learn preprocessing/classification pipeline.
- Model-derived global feature explanation when the classifier supports it.
- General health information, with no personalized treatment instructions.
- Holdout performance display when a matching evaluation artifact exists.

## Project structure

```text
app/app.py                  Streamlit entry point
src/preprocessing.py        Dataset validation and shared preprocessing
src/train_model.py          Reproducible model training
src/evaluate_model.py       Holdout evaluation
src/explainability.py       Model-derived feature importance
src/recommendations.py      General health guidance
data/raw/diabetes.csv       Required original dataset (not included if empty)
models/                     Trained pipeline and metadata
results/                    Evaluation JSON
tests/                      Automated tests
```

## Prerequisites and environment

Python 3.11 is the intended runtime. Create an environment and install pinned direct dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in `data/raw/diabetes.csv` is currently a zero-byte file. Supply the project's legitimate source dataset in that location before training. The expected schema is `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`; the code validates it rather than making a replacement dataset.

## Train, evaluate, and test

```bash
python -m src.train_model
python -m src.evaluate_model
pytest -q
```

Training compares logistic regression and random forest using a stratified validation subset drawn from the development data, then refits the chosen pipeline on all development rows and records the untouched 20% test ROC-AUC in metadata. It writes `models/best_model.joblib` and `models/metadata.json`. Evaluation separately writes test metrics to `results/evaluation.json`. No real metrics are available until the authentic dataset is restored and these commands run successfully.

## Run locally

```bash
streamlit run app/app.py
```

The app reports a missing model rather than fabricating an estimate. Paths are based on the project files, so launch from the repository root or another working directory.

## GitHub upload

Create a repository on GitHub, then from this project directory:

```bash
git init
git add .
git commit -m "Prepare Chronic Disease AI app"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Include the authentic dataset only when you have the right to redistribute it. Train locally and include the generated compatible model and metadata files if deploying without training on the cloud. Keep secrets and local virtual environments out of Git.

## Streamlit Community Cloud deployment

1. Push the reviewed project to GitHub.
2. In Streamlit Community Cloud, create an app from that repository and branch.
3. Set the main file path to `app/app.py`.
4. Deploy with Python 3.11 and the repository's `requirements.txt`.
5. Confirm the authentic `models/best_model.joblib` and `models/metadata.json` artifacts are committed and load on the cloud app.
6. Test each page and a known valid input in the deployed app. Local verification does not confirm cloud deployment.

## Limitations and medical use

This app depends on the source dataset and serialized model, neither of which is presently usable in this checkout. The input ranges are guardrails based on the commonly used dataset's documented ranges and are not clinical thresholds. Model estimates are educational outputs and can be biased or inaccurate. They are not diagnoses, should not replace professional care, and must not be used to make medication or treatment decisions. Consult a qualified healthcare professional for health concerns.
