"""
train_models.py
-----------------
Trains and compares two regression models for next-day temperature
prediction:
  - Linear Regression (simple baseline)
  - Random Forest Regressor (non-linear ensemble model)

Uses a chronological (not random) train/test split, since shuffling a time
series would let the model "see the future" during training.
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

TEST_SIZE_FRACTION = 0.2
RANDOM_STATE = 42


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_SIZE_FRACTION):
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)}


def train_and_evaluate(df: pd.DataFrame):
    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE
        ),
    }

    results = {}
    predictions = {"date": test_df["date"].values, "actual": y_test.values}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = evaluate(y_test, preds)
        predictions[name] = preds
        joblib.dump(model, f"outputs/{name}.joblib")

    return results, pd.DataFrame(predictions), models


if __name__ == "__main__":
    features = pd.read_csv("data/weather_features.csv", parse_dates=["date"])
    results, preds_df, _ = train_and_evaluate(features)

    print(json.dumps(results, indent=2))
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    preds_df.to_csv("outputs/test_predictions.csv", index=False)
