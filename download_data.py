
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io

ROOT = Path(__file__).resolve().parent
MAT_PATH = ROOT / "water_dataset.mat"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

mat_data = scipy.io.loadmat(MAT_PATH)

feature_names = [
    "specific_conductance_max",
    "ph_max",
    "ph_min",
    "specific_conductance_min",
    "specific_conductance_mean",
    "dissolved_oxygen_max",
    "dissolved_oxygen_mean",
    "dissolved_oxygen_min",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
]

def prepare_data(X_raw, Y_raw):
    X_samples = [
        np.asarray(sample, dtype=float)
        for sample in X_raw.ravel()
    ]

    X = np.vstack(X_samples)
    y = np.asarray(Y_raw, dtype=float).T.reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature-target mismatch: {X.shape[0]} rows "
            f"and {y.shape[0]} targets."
        )

    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y[:, 0]

    return df.replace([np.inf, -np.inf], np.nan)

train_df = prepare_data(mat_data["X_tr"], mat_data["Y_tr"])
test_df = prepare_data(mat_data["X_te"], mat_data["Y_te"])

train_df.to_csv(DATA_DIR / "water_quality_train.csv", index=False)
test_df.to_csv(DATA_DIR / "water_quality_test.csv", index=False)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)
print("\nTraining missing values:")
print(train_df.isnull().sum())
print("\nTesting missing values:")
print(test_df.isnull().sum())
print("\nTraining sample:")
print(train_df.head())
print("\nSaved both datasets to:", DATA_DIR)
