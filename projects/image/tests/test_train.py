import pytest
from projects.image.src.train import train


def test_train_returns_results():
    results = train()
    assert "random_forest" in results
    assert "mlp" in results


def test_train_accuracy_above_baseline():
    results = train()
    for m in results.values():
        assert m["accuracy"] > 0.3
