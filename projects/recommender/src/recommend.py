import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from projects.recommender.src.data import generate_ratings, generate_item_profiles

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_user_item_matrix(ratings: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    matrix = np.zeros((n_users, n_items))
    for _, row in ratings.iterrows():
        matrix[int(row["user_id"]), int(row["item_id"])] = row["rating"]
    return matrix


def recommend(user_id: int, n_recommend: int = 5) -> dict:
    ratings = generate_ratings()
    profiles = generate_item_profiles()
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()

    matrix = build_user_item_matrix(ratings, n_users, n_items)
    sim = cosine_similarity(matrix)
    sim_users = sim[user_id]
    similar_users = np.argsort(sim_users)[-4:-1]

    user_ratings = matrix[user_id]
    unrated = np.where(user_ratings == 0)[0]

    scores = []
    for item in unrated:
        pred = np.mean([matrix[u, item] for u in similar_users if matrix[u, item] > 0])
        if pred > 0:
            scores.append((int(item), float(pred)))

    scores.sort(key=lambda x: x[1], reverse=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(matrix, os.path.join(MODEL_DIR, "user_item_matrix.joblib"))

    return {
        "user_id": user_id,
        "recommendations": [
            {"item_id": item, "predicted_rating": round(rating, 2)}
            for item, rating in scores[:n_recommend]
        ],
    }


if __name__ == "__main__":
    result = recommend(user_id=0)
    print(f"Recommendations for user {result['user_id']}:")
    for r in result["recommendations"]:
        print(f"  Item {r['item_id']}: predicted {r['predicted_rating']}")
