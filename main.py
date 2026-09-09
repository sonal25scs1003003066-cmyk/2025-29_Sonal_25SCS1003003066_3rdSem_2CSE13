"""
main.py
--------
Runs the full weather analysis and prediction pipeline end to end:
  raw data -> cleaning -> feature engineering -> model training ->
  evaluation -> visualization

Usage:
    python src/main.py
"""

import json

from data_preprocessing import clean_data, load_data, summarize_cleaning
from feature_engineering import build_feature_table
from train_models import train_and_evaluate
from visualize import plot_actual_vs_predicted, plot_correlation_heatmap, plot_seasonal_trends


def run_pipeline():
    print("Step 1/5 - Loading raw data...")
    raw = load_data("data/historical_weather.csv")

    print("Step 2/5 - Cleaning data...")
    clean = clean_data(raw)
    print("  ", summarize_cleaning(raw, clean))
    clean.to_csv("data/historical_weather_clean.csv", index=False)

    print("Step 3/5 - Engineering features...")
    features = build_feature_table(clean)
    features.to_csv("data/weather_features.csv", index=False)
    print(f"   Feature table shape: {features.shape}")

    print("Step 4/5 - Training and evaluating models...")
    results, preds_df, _ = train_and_evaluate(features)
    print("  ", json.dumps(results, indent=2))
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    preds_df.to_csv("outputs/test_predictions.csv", index=False)

    print("Step 5/5 - Generating plots...")
    plot_seasonal_trends(clean)
    plot_correlation_heatmap(clean)
    plot_actual_vs_predicted(preds_df)

    print("\nDone. See outputs/metrics.json and outputs/figures/ for results.")


if __name__ == "__main__":
    run_pipeline()
