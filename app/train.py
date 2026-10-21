
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "water_quality_train.csv"
TEST_PATH = DATA_DIR / "water_quality_test.csv"
MODEL_DIR = ROOT / "models"

TARGET_COLUMN = "Target"


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run download_data.py first."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {path.name}."
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


def main():
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    feature_names = [
        column for column in train_df.columns
        if column != TARGET_COLUMN
    ]

    feature_names = [
        column for column in feature_names
        if train_df[column].notna().any()
        and train_df[column].nunique(dropna=True) > 1
    ]

    if not feature_names:
        raise ValueError("No usable input features remain.")

    missing_test_features = [
        column for column in feature_names
        if column not in test_df.columns
    ]

    if missing_test_features:
        raise ValueError(
            f"Test dataset is missing features: {missing_test_features}"
        )

    X_train = train_df[feature_names]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COLUMN]

    if len(X_train) < 20:
        raise ValueError("Too few training rows.")

    if len(X_test) == 0:
        raise ValueError("No valid testing rows remain.")

    print("Training dataset shape:", train_df.shape)
    print("Testing dataset shape:", test_df.shape)
    print("Number of features:", len(feature_names))
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    models = {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", ExtraTreesRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )),
        ]),
    }

    results = []
    best_model = None
    best_mae = float("inf")
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        })

        print(
            f"{name}: MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, R2={r2:.4f}"
        )

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "water_quality_model.joblib"

    joblib.dump({
        "model": best_model,
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "model_name": best_name,
    }, model_path)

    metrics_path = MODEL_DIR / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "target_column": TARGET_COLUMN,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "metrics": results,
        }, f, indent=2)

    print(f"Best model by test MAE: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
