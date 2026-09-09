"""
visualize.py
-------------
Generates the exploratory and results plots for the project:
  1. seasonal_trends.png    - daily temperature over time with a 30-day rolling mean
  2. correlation_heatmap.png - correlation between the core weather variables
  3. actual_vs_predicted.png - Random Forest predictions vs actual on the test period
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

CORE_COLUMNS = ["temperature_c", "humidity_pct", "wind_speed_kmh", "precipitation_mm"]


def plot_seasonal_trends(df: pd.DataFrame, out_path: str = "outputs/figures/seasonal_trends.png"):
    df = df.copy()
    df["temp_rolling_30d"] = df["temperature_c"].rolling(30).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df["temperature_c"], color="#C9A0B5", linewidth=0.6, alpha=0.6, label="Daily temperature")
    ax.plot(df["date"], df["temp_rolling_30d"], color="#8E3A5C", linewidth=2, label="30-day rolling mean")
    ax.set_title("Daily Temperature and Seasonal Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: str = "outputs/figures/correlation_heatmap.png"):
    corr = df[CORE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Between Weather Variables", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(preds_df: pd.DataFrame, out_path: str = "outputs/figures/actual_vs_predicted.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(preds_df["date"], preds_df["actual"], label="Actual", color="#3A3138", linewidth=1.6)
    ax.plot(preds_df["date"], preds_df["random_forest"], label="Random Forest prediction", color="#C75B85", linewidth=1.4, linestyle="--")
    ax.plot(preds_df["date"], preds_df["linear_regression"], label="Linear Regression prediction", color="#8AA6C7", linewidth=1.0, linestyle=":")
    ax.set_title("Next-Day Temperature: Actual vs Predicted (Test Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    clean = pd.read_csv("data/historical_weather_clean.csv", parse_dates=["date"])
    preds = pd.read_csv("outputs/test_predictions.csv", parse_dates=["date"])

    plot_seasonal_trends(clean)
    plot_correlation_heatmap(clean)
    plot_actual_vs_predicted(preds)
    print("Saved figures to outputs/figures/")
