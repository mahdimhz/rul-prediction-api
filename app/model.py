from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_rul_femto.pkl"

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
scaler = model_bundle["scaler"]
max_rul = float(model_bundle["max_rul"])
feature_cols = model_bundle["feature_cols"]


def predict_rul(features: dict) -> dict:
    """Predict remaining useful life from engineered bearing features."""
    features_array = np.array(
        [[features[feature_name] for feature_name in feature_cols]],
        dtype=float,
    )
    scaled_input = scaler.transform(features_array)
    pred_norm = model.predict(scaled_input)
    pred_seconds = float(pred_norm[0]) * max_rul
    rul_seconds = max(0.0, pred_seconds)

    return {
        "predicted_rul_seconds": rul_seconds,
        "predicted_rul_minutes": rul_seconds / 60.0,
        "n_features_received": len(features),
    }
