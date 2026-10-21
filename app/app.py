"""Streamlit entry point for the Chronic Disease AI educational prototype."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability import get_feature_importance
from src.preprocessing import DATA_PATH, FEATURE_COLUMNS, PROJECT_ROOT
from src.recommendations import get_recommendations

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "results" / "evaluation.json"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
FEATURE_INFO = {
    "Pregnancies": (0, 17, 1, 0, "Number of pregnancies"),
    "Glucose": (0, 200, 1, 100, "Glucose concentration (dataset measurement units)"),
    "BloodPressure": (0, 122, 1, 70, "Diastolic blood pressure (dataset measurement units)"),
    "SkinThickness": (0, 99, 1, 20, "Triceps skin-fold thickness (dataset measurement units)"),
    "Insulin": (0, 846, 1, 80, "Insulin concentration (dataset measurement units)"),
    "BMI": (0.0, 67.1, 0.1, 25.0, "Body mass index"),
    "DiabetesPedigreeFunction": (0.0, 2.5, 0.01, 0.5, "Diabetes pedigree function value"),
    "Age": (21, 100, 1, 35, "Age in years"),
}


@st.cache_resource(show_spinner="Loading trained model…")
def load_model(path_string):
    import joblib
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model is missing at {path}. Add the original dataset, then run `python -m src.train_model`.")
    return joblib.load(path)


def validate_inputs(values):
    for name, value in values.items():
        lower, upper = FEATURE_INFO[name][:2]
        if pd.isna(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}.")
    if values["Age"] < 21:
        raise ValueError("This model's source dataset represents adults aged 21 and older.")


def prediction_page():
    st.title("Risk Prediction")
    st.caption("Enter measurements as recorded. A value of zero is accepted as an input but does not have special medical interpretation here.")
    with st.form("prediction_form"):
        values = {}
        left, right = st.columns(2)
        for index, name in enumerate(FEATURE_COLUMNS):
            lo, hi, step, default, help_text = FEATURE_INFO[name]
            with (left if index % 2 == 0 else right):
                values[name] = st.number_input(name, min_value=lo, max_value=hi, value=default, step=step, help=help_text)
        submitted = st.form_submit_button("Estimate risk")
    if submitted:
        try:
            validate_inputs(values)
            model = load_model(str(MODEL_PATH))
            sample = pd.DataFrame([[values[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            predicted = model.predict(sample)[0]
            st.subheader("Model estimate")
            st.write(f"Predicted class: **{predicted}** ({'higher-risk category' if predicted == 1 else 'lower-risk category'})")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(sample)[0]
                if not all(0 <= float(probability) <= 1 for probability in probabilities):
                    raise ValueError("The model returned a probability outside [0, 1].")
                st.markdown("#### Class probabilities")
                probability_columns = st.columns(len(model.classes_))
                for column, class_label, probability in zip(probability_columns, model.classes_, probabilities):
                    label = "Higher-risk category" if class_label == 1 else "Lower-risk category"
                    column.metric(f"Class {class_label} · {label}", f"{float(probability):.1%}")
                st.caption("Model-estimated class probabilities are not diagnoses or guaranteed medical outcomes.")
            st.markdown("#### General health information")
            for item in get_recommendations(int(predicted)):
                st.write(f"• {item}")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")


def explainability_page():
    st.subheader("Model Explainability")
    try:
        model = load_model(str(MODEL_PATH))
        rows = get_feature_importance(model, FEATURE_COLUMNS)
        if rows is None:
            st.info("The selected classifier does not expose compatible feature importance or coefficients.")
        else:
            importance = pd.DataFrame(rows, columns=["Feature", "Importance"]).set_index("Feature")
            st.bar_chart(importance)
            st.dataframe(importance.style.format("{:.6f}"), use_container_width=True)
    except FileNotFoundError as exc:
        st.info(str(exc))


def performance_page():
    st.subheader("Model Performance")
    if not METRICS_PATH.is_file():
        st.info("No verified evaluation artifact is available. Restore the real dataset, train the pipeline, then run `python -m src.evaluate_model`.")
        return
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        st.warning("Evaluation results cannot be verified because the model or its metadata is missing.")
        return
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.caption(f"Model: {metadata['model_name']} · scikit-learn {metadata['scikit_learn_version']} · evaluation split recorded by the project pipeline")
        metric_names = [key for key in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC") if key in metrics]
        cols = st.columns(len(metric_names))
        for col, key in zip(cols, metric_names):
            col.metric(key, f"{metrics[key]:.3f}")
        if "Confusion Matrix" in metrics:
            st.write("Confusion Matrix")
            matrix = pd.DataFrame(metrics["Confusion Matrix"], index=["Actual: Class 0", "Actual: Class 1"], columns=["Predicted: Class 0", "Predicted: Class 1"])
            st.dataframe(matrix.style.format("{:,.0f}"), use_container_width=True)
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Evaluation artifact is invalid: {exc}")


def main():
    st.set_page_config(page_title="Chronic Disease AI", page_icon="🩺", layout="wide")
    page = st.sidebar.radio("Navigate", ["Risk Prediction", "Model Insights"], index=0)
    if page == "Risk Prediction":
        prediction_page()
    else:
        st.title("Model Insights")
        explainability_page()
        st.divider()
        performance_page()


if __name__ == "__main__":
    main()
