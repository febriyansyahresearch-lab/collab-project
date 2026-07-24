import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from projects.nlp.src.data import generate_texts

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train() -> dict:
    df = generate_texts()
    X, y = df["text"], df["label"]
    vec = TfidfVectorizer(max_features=1000, stop_words="english")
    X_vec = vec.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(kernel="linear", random_state=42),
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
    joblib.dump(vec, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(models["svm"], os.path.join(MODEL_DIR, "svm_model.joblib"))
    return results


if __name__ == "__main__":
    results = train()
    for name, m in results.items():
        print(f"{name}: accuracy={m['accuracy']:.4f}")
