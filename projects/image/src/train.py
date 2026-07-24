import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from projects.image.src.data import generate_images

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train() -> dict:
    X, y, classes = generate_images()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models["random_forest"], os.path.join(MODEL_DIR, "image_model.joblib"))
    return results


if __name__ == "__main__":
    results = train()
    for name, m in results.items():
        print(f"{name}: accuracy={m['accuracy']:.4f}")
