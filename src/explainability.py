"""Model-derived global explanations; these describe model behavior, not causation."""
import numpy as np


def get_feature_importance(model, feature_names):
    classifier = model.named_steps.get("model", model)
    if hasattr(classifier, "feature_importances_"):
        scores = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        scores = np.abs(np.asarray(classifier.coef_, dtype=float)).mean(axis=0)
    else:
        return None
    if len(scores) != len(feature_names):
        raise ValueError("Feature importance count does not match the model's input features.")
    return sorted(zip(feature_names, scores.tolist()), key=lambda item: item[1], reverse=True)
