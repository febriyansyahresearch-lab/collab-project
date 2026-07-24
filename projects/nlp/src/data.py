import numpy as np
import pandas as pd


TOPICS = {
    "tech": ["python", "cloud", "api", "docker", "kubernetes", "microservice", "devops", "git", "database", "server"],
    "sports": ["football", "basketball", "tennis", "champion", "league", "player", "coach", "stadium", "goal", "match"],
    "finance": ["stock", "market", "investment", "bank", "interest", "dividend", "portfolio", "trading", "bond", "asset"],
}


def generate_texts(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, words in TOPICS.items():
        for _ in range(n // len(TOPICS)):
            text = " ".join(rng.choice(words, size=rng.integers(3, 8))).lower()
            rows.append({"text": text, "label": label})
    rng.shuffle(rows)
    return pd.DataFrame(rows)
