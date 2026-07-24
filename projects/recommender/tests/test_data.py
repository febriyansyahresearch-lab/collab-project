import pytest
from projects.recommender.src.data import generate_ratings, generate_item_profiles


def test_generate_ratings_columns():
    df = generate_ratings(20, 10)
    assert list(df.columns) == ["user_id", "item_id", "rating"]


def test_generate_ratings_rating_range():
    df = generate_ratings(50, 20)
    assert df["rating"].between(1, 5).all()


def test_generate_item_profiles():
    df = generate_item_profiles(10, 5)
    assert df.shape == (10, 6)
