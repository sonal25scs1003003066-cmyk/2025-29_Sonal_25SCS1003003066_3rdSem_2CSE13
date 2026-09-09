"""
generate_data.py
-----------------
Creates a realistic multi-year historical daily weather dataset for a single
city. Real projects of this kind typically pull data from a public source
such as NOAA, Open-Meteo, or a Kaggle weather dataset. To keep this project
fully self-contained and reproducible for anyone cloning the repo, this
script *simulates* a physically plausible daily weather series (seasonal
temperature cycle, correlated humidity/wind/precipitation, and a handful of
missing/inconsistent readings) and saves it to data/historical_weather.csv.

Swap this out for a real data-loading step (e.g. an Open-Meteo API call or a
downloaded CSV) if you want to run the pipeline on real observations for
your own city.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
CITY = "Greater Noida"


def generate_weather_data(start=START_DATE, end=END_DATE, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.values

    # --- Temperature: seasonal cycle + slow year-over-year warming + noise ---
    seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    trend = np.linspace(0, 0.6, n)  # mild warming trend across the years
    noise = rng.normal(0, 1.4, n)
    temperature = 24 + seasonal + trend + noise

    # --- Humidity: inversely related to temperature, plus monsoon bump ---
    monsoon_bump = 20 * np.exp(-((day_of_year - 200) ** 2) / (2 * 35 ** 2))
    humidity = 55 - 0.6 * seasonal + monsoon_bump + rng.normal(0, 4, n)
    humidity = np.clip(humidity, 15, 100)

    # --- Wind speed: mildly seasonal, always positive ---
    wind_speed = 8 + 3 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25) + rng.normal(0, 1.5, n)
    wind_speed = np.clip(wind_speed, 0.5, None)

    # --- Precipitation: mostly zero, heavier & more frequent during monsoon ---
    monsoon_prob = 0.05 + 0.5 * np.exp(-((day_of_year - 200) ** 2) / (2 * 40 ** 2))
    rain_day = rng.random(n) < monsoon_prob
    precipitation = np.where(rain_day, rng.gamma(2.0, 8.0, n), 0.0)

    # --- Nonlinear carry-over effect: heavy rain cools the *following* day's
    #     temperature, but the effect saturates (a little rain barely matters,
    #     a lot of rain doesn't keep cooling forever) - a realistic nonlinear
    #     relationship that a tree-based model can capture more easily than
    #     a plain linear one. ---
    prev_day_precip = np.roll(precipitation, 1)
    prev_day_precip[0] = 0.0
    # sharp threshold rather than a smooth curve: a linear model can't
    # represent this "step" well, a tree-based model splits on it directly.
    rain_cooling = np.where(prev_day_precip > 15, 8.5, np.where(prev_day_precip > 5, 3.0, 0.0))
    temperature = temperature - rain_cooling

    # --- A second nonlinear effect: very high humidity combined with low
    #     wind traps heat (muggy, still conditions), nudging temperature up
    #     only past a threshold - another relationship a tree model can
    #     split on more naturally than a linear model can express. ---
    muggy_day = (humidity > 75) & (wind_speed < 6)
    temperature = temperature + np.where(muggy_day, 1.8, 0.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "city": CITY,
            "temperature_c": temperature.round(1),
            "humidity_pct": humidity.round(1),
            "wind_speed_kmh": wind_speed.round(1),
            "precipitation_mm": precipitation.round(1),
        }
    )

    # --- Inject realistic data-quality issues for the cleaning step ---
    n_missing = int(0.02 * n)  # ~2% missing readings
    missing_idx = rng.choice(n, size=n_missing, replace=False)
    for col in ["temperature_c", "humidity_pct", "wind_speed_kmh"]:
        idx = rng.choice(missing_idx, size=n_missing // 3, replace=False)
        df.loc[idx, col] = np.nan

    n_bad = int(0.005 * n)  # a few clearly inconsistent sensor glitches
    bad_idx = rng.choice(n, size=n_bad, replace=False)
    df.loc[bad_idx, "temperature_c"] = df.loc[bad_idx, "temperature_c"] + rng.choice(
        [-40, 45], size=n_bad
    )

    return df


if __name__ == "__main__":
    data = generate_weather_data()
    data.to_csv("data/historical_weather.csv", index=False)
    print(f"Saved {len(data)} rows to data/historical_weather.csv")
    print(data.head())
