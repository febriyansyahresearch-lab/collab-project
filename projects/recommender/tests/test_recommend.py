import pytest
from projects.recommender.src.recommend import recommend, build_user_item_matrix
from projects.recommender.src.data import generate_ratings


def test_recommend_returns_recommendations():
    result = recommend(user_id=0, n_recommend=3)
    assert "recommendations" in result
    assert len(result["recommendations"]) <= 3


def test_build_user_item_matrix_shape():
    ratings = generate_ratings(10, 8)
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    matrix = build_user_item_matrix(ratings, n_users, n_items)
    assert matrix.shape == (10, 8)
