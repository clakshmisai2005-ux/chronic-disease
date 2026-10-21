import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS, create_preprocessing_pipeline, load_data, prepare_data
from src.recommendations import get_recommendations
from src.explainability import get_feature_importance
from src.evaluate_model import evaluate_model
from src.train_model import train_models
from app.app import validate_inputs, FEATURE_INFO


def fixture_data(rows=40):
    values = np.arange(rows * len(FEATURE_COLUMNS), dtype=float).reshape(rows, len(FEATURE_COLUMNS))
    frame = pd.DataFrame(values + 1, columns=FEATURE_COLUMNS)
    frame["Outcome"] = np.array([0, 1] * (rows // 2))
    return frame


def test_load_data_missing_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        load_data(empty)


def test_prepare_data_feature_order_and_validation():
    frame = fixture_data()
    X, y = prepare_data(frame)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(frame)
    with pytest.raises(ValueError, match="Required feature"):
        prepare_data(frame.drop(columns=[FEATURE_COLUMNS[0]]))


def test_pipeline_training_prediction_and_serialization(tmp_path):
    frame = fixture_data()
    X, y = prepare_data(frame)
    pipeline = Pipeline([("preprocessing", create_preprocessing_pipeline()), ("model", LogisticRegression())])
    pipeline.fit(X, y)
    output = tmp_path / "model.joblib"
    joblib.dump(pipeline, output)
    loaded = joblib.load(output)
    assert loaded.predict(X.iloc[:3]).shape == (3,)
    probabilities = loaded.predict_proba(X.iloc[:3])
    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    metrics = evaluate_model(loaded, X.iloc[:8], y.iloc[:8])
    assert 0 <= metrics["Accuracy"] <= 1


def test_input_validation_and_recommendations():
    values = {name: FEATURE_INFO[name][3] for name in FEATURE_COLUMNS}
    validate_inputs(values)
    values["Age"] = 20
    with pytest.raises(ValueError):
        validate_inputs(values)
    assert get_recommendations(1)
    assert get_recommendations(0)


def test_explanation_from_fitted_pipeline():
    frame = fixture_data()
    X, y = prepare_data(frame)
    pipeline = Pipeline([("preprocessing", create_preprocessing_pipeline()), ("model", LogisticRegression())]).fit(X, y)
    rows = get_feature_importance(pipeline, FEATURE_COLUMNS)
    assert rows is not None and len(rows) == len(FEATURE_COLUMNS)


def test_repository_dataset_schema():
    X, y = prepare_data(load_data())
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 768
    assert set(y.unique()) == {0, 1}
