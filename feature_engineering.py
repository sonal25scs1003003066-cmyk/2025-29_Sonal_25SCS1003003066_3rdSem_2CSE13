"""
feature_engineering.py
------------------------
Builds the time-aware features used to predict next-day temperature:
  - calendar features (day of year, month, seasonality via sin/cos)
  - lag features (yesterday's / last-3-days' readings)
  - rolling averages (7-day rolling mean of temperature & humidity)
  - the prediction target: next day's temperature
"""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3]:
        df[f"temp_lag_{lag}"] = df["temperature_c"].shift(lag)
        df[f"humidity_lag_{lag}"] = df["humidity_pct"].shift(lag)

    df["temp_roll_mean_7"] = df["temperature_c"].rolling(window=7).mean()
    df["humidity_roll_mean_7"] = df["humidity_pct"].rolling(window=7).mean()
    df["wind_roll_mean_7"] = df["wind_speed_kmh"].rolling(window=7).mean()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_next_day_temp"] = df["temperature_c"].shift(-1)
    return df


FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "season_sin",
    "season_cos",
    "temp_lag_1",
    "temp_lag_2",
    "temp_lag_3",
    "humidity_lag_1",
    "humidity_lag_2",
    "humidity_lag_3",
    "temp_roll_mean_7",
    "humidity_roll_mean_7",
    "wind_roll_mean_7",
]
TARGET_COLUMN = "target_next_day_temp"


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_target(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    clean = pd.read_csv("data/historical_weather_clean.csv", parse_dates=["date"])
    features = build_feature_table(clean)
    features.to_csv("data/weather_features.csv", index=False)
    print(f"Feature table shape: {features.shape}")
    print(features[FEATURE_COLUMNS + [TARGET_COLUMN]].head())
