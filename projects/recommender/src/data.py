import numpy as np
import pandas as pd


def generate_ratings(n_users: int = 50, n_items: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        min_rated = min(5, n_items - 1)
        n_rated = rng.integers(min_rated, n_items) if n_items > 1 else 1
        items = rng.choice(n_items, size=n_rated, replace=False)
        for i in items:
            rating = int(np.clip(rng.normal(3.5, 1.2), 1, 5))
            rows.append({"user_id": u, "item_id": i, "rating": rating})
    return pd.DataFrame(rows)


def generate_item_profiles(n_items: int = 20, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"feature_{f}": rng.uniform(0, 1, n_items) for f in range(n_features)}
    data["item_id"] = range(n_items)
    return pd.DataFrame(data)
