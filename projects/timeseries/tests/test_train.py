import pytest
from projects.timeseries.src.train import train


def test_train_returns_results():
    results = train()
    assert "linear" in results
    assert "random_forest" in results
    assert "gradient_boosting" in results


def test_train_rmse_finite():
    results = train()
    for m in results.values():
        assert m["rmse"] > 0
        assert m["mae"] > 0
