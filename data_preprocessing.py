"""
data_preprocessing.py
----------------------
Loading and cleaning steps for the historical weather dataset:
  1. Parse dates and sort chronologically.
  2. Flag and fix physically impossible / inconsistent readings.
  3. Fill missing values using time-aware interpolation.
"""

import numpy as np
import pandas as pd

VALID_RANGES = {
    "temperature_c": (-10, 55),
    "humidity_pct": (0, 100),
    "wind_speed_kmh": (0, 120),
    "precipitation_mm": (0, 300),
}


def load_data(path: str = "data/historical_weather.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Mark out-of-range / inconsistent sensor readings as missing
    for col, (low, high) in VALID_RANGES.items():
        out_of_range = ~df[col].between(low, high)
        df.loc[out_of_range, col] = np.nan

    # 2. Time-aware interpolation to fill gaps, then fill any edge NaNs
    df = df.set_index("date")
    numeric_cols = list(VALID_RANGES.keys())
    df[numeric_cols] = df[numeric_cols].interpolate(method="time").bfill().ffill()
    df = df.reset_index()

    # 3. Drop exact duplicate calendar days, keep the first
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    return df


def summarize_cleaning(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    return {
        "rows_raw": len(raw),
        "rows_clean": len(clean),
        "missing_before": int(raw[list(VALID_RANGES.keys())].isna().sum().sum()),
        "missing_after": int(clean[list(VALID_RANGES.keys())].isna().sum().sum()),
    }


if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)
    print(summarize_cleaning(raw, clean))
    clean.to_csv("data/historical_weather_clean.csv", index=False)
