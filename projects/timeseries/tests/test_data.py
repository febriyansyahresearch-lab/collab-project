import pytest
from projects.timeseries.src.data import generate_series, train_test_split_series


def test_generate_series_shape():
    df = generate_series(200)
    assert df.shape == (200, 2)


def test_generate_series_columns():
    df = generate_series(100)
    assert list(df.columns) == ["date", "value"]


def test_train_test_split():
    df = generate_series(365)
    train, test = train_test_split_series(df, test_days=60)
    assert len(train) == 305
    assert len(test) == 60
