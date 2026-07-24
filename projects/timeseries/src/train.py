import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from projects.timeseries.src.data import generate_series, train_test_split_series

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = np.arange(len(df))
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def train() -> dict:
    df = generate_series()
    df = make_features(df)
    train_df, test_df = train_test_split_series(df, test_days=60)

    features = ["day", "sin_month", "cos_month", "dayofweek"]
    X_train, y_train = train_df[features], train_df["value"]
    X_test, y_test = test_df[features], test_df["value"]

    models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models["gradient_boosting"], os.path.join(MODEL_DIR, "ts_model.joblib"))
    return results


if __name__ == "__main__":
    results = train()
    for name, m in results.items():
        print(f"{name}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}")
