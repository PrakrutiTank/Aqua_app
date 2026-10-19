
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


# --------------------------------------------------
# 1. Application and project paths
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "water_quality_model.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "metrics.json"

app = Flask(
    __name__,
    template_folder=str(PROJECT_DIR / "templates"),
    static_folder=str(PROJECT_DIR / "static"),
    static_url_path="/static",
)


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Run your training script first."
    )

artifact = joblib.load(MODEL_PATH)

# Support either a saved model directly or a dictionary
# containing the model and its feature schema.
if isinstance(artifact, dict):
    model = (
        artifact.get("model")
        or artifact.get("pipeline")
        or artifact.get("best_model")
    )

    feature_names = (
        artifact.get("feature_names")
        or artifact.get("features")
    )
else:
    model = artifact
    feature_names = None

if model is None:
    raise ValueError(
        "Could not find the trained model in the saved artifact. "
        "Check the structure saved by app/train.py."
    )

# Get feature names from the trained model if necessary.
if not feature_names:
    feature_names = getattr(model, "feature_names_in_", None)

if feature_names is None:
    raise ValueError(
        "Could not determine model feature names. "
        "Save feature_names in your model artifact."
    )

feature_names = list(feature_names)


# --------------------------------------------------
# 3. Frontend homepage
# --------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html"), 200


# --------------------------------------------------
# 4. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "project": "AquaGuard"
    }), 200


@app.get("/api")
def api_home():
    return jsonify({
        "project": "AquaGuard",
        "status": "running"
    }), 200

# --------------------------------------------------
# 5. Return the required model features
# --------------------------------------------------

@app.get("/features")
def features():
    return jsonify({
        "features": feature_names,
        "required_features": feature_names
    }), 200


# --------------------------------------------------
# 6. Return training metrics
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({
            "error": "Training metrics file not found."
        }), 404

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics_data = json.load(file)

        return jsonify(metrics_data), 200

    except (json.JSONDecodeError, OSError):
        return jsonify({
            "error": "Could not read training metrics."
        }), 500


# --------------------------------------------------
# 7. Make a water-quality prediction
# --------------------------------------------------

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    # Validate the request body.
    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    # Check that all required features are present.
    missing_features = [
        name for name in feature_names
        if name not in data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features
        }), 400

    # Convert input values to numbers.
    values = []

    for name in feature_names:
        try:
            value = float(data[name])
        except (TypeError, ValueError, OverflowError):
            return jsonify({
                "error": f"Feature '{name}' must be numeric."
            }), 400

        if not np.isfinite(value):
            return jsonify({
                "error": f"Feature '{name}' must be a finite number."
            }), 400

        values.append(value)

    # Preserve the exact feature order used during training.
    input_df = pd.DataFrame(
        [values],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_df)
        predicted_value = float(prediction[0])

        if not np.isfinite(predicted_value):
            return jsonify({
                "error": "The model produced an invalid prediction."
            }), 500

        return jsonify({
            "prediction": predicted_value,
            "target": "next_day_pH",
            "message": "Prediction generated successfully."
        }), 200

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Check the model and input schema."
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
