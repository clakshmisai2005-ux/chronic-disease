"""Train reproducible sklearn pipelines from the repository's diabetes CSV."""
import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.preprocessing import DATA_PATH, FEATURE_COLUMNS, PROJECT_ROOT, load_data, prepare_data, split_data, create_preprocessing_pipeline

MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"


def train_models():
    X, y = prepare_data(load_data(DATA_PATH))
    X_development, X_test, y_development, y_test = split_data(X, y)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development, y_development, test_size=0.25, random_state=43, stratify=y_development
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocessing", create_preprocessing_pipeline()), ("model", classifier)])
        pipeline.fit(X_train, y_train)
        scores[name] = float(roc_auc_score(y_validation, pipeline.predict_proba(X_validation)[:, list(pipeline.classes_).index(1)]))
    best_name = max(scores, key=scores.get)
    final_model = Pipeline([("preprocessing", create_preprocessing_pipeline()), ("model", candidates[best_name])])
    final_model.fit(X_development, y_development)
    test_score = float(roc_auc_score(y_test, final_model.predict_proba(X_test)[:, list(final_model.classes_).index(1)]))
    joblib.dump(final_model, BEST_MODEL_PATH)
    metadata = {
        "model_name": best_name,
        "feature_names": FEATURE_COLUMNS,
        "target": "Outcome",
        "selection_metric": "validation ROC-AUC",
        "validation_scores": scores,
        "test_roc_auc": test_score,
        "random_state": 42,
        "test_size": 0.2,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {best_name: final_model}, (X_development, X_test, y_development, y_test), metadata


if __name__ == "__main__":
    _, _, metadata = train_models()
    print(f"Selected model: {metadata['model_name']}")
    print(f"Test ROC-AUC: {metadata['test_roc_auc']:.3f}")
