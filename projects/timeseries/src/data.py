import numpy as np
import pandas as pd


def generate_series(n: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    trend = np.linspace(0, 10, n)
    season = 5 * np.sin(2 * np.pi * np.arange(n) / 365 * 3)
    noise = rng.normal(0, 2, n)
    values = 50 + trend + season + noise
    return pd.DataFrame({"date": dates, "value": values.round(2)})


def train_test_split_series(df: pd.DataFrame, test_days: int = 60):
    return df.iloc[:-test_days], df.iloc[-test_days:]
