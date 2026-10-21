"""Evaluate the saved model on the reproducible stratified holdout split."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.preprocessing import DATA_PATH, PROJECT_ROOT, load_data, prepare_data, split_data

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    result = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    if hasattr(model, "predict_proba") and 1 in model.classes_:
        positive = list(model.classes_).index(1)
        result["ROC-AUC"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, positive]))
    return result


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model is missing: {MODEL_PATH}. Run `python -m src.train_model` first.")
    X, y = prepare_data(load_data(DATA_PATH))
    _, X_test, _, y_test = split_data(X, y)
    model = joblib.load(MODEL_PATH)
    metrics = evaluate_model(model, X_test, y_test)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in metrics.items()}
    (RESULTS_DIR / "evaluation.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(pd.Series({key: value for key, value in metrics.items() if key != "Confusion Matrix"}).to_string())
    print("Confusion Matrix:\n", np.array(metrics["Confusion Matrix"]))
    return metrics


if __name__ == "__main__":
    main()
