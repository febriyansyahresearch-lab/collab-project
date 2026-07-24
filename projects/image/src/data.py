import numpy as np
import pandas as pd


def generate_images(n: int = 200, img_size: int = 16, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X, y = [], []
    classes = {"circle": 0, "square": 1, "triangle": 2}

    for label, code in classes.items():
        for _ in range(n // len(classes)):
            img = np.zeros((img_size, img_size))
            cx, cy = rng.integers(4, img_size - 4), rng.integers(4, img_size - 4)
            r = rng.integers(3, 6)

            for i in range(img_size):
                for j in range(img_size):
                    if label == "circle":
                        if (i - cx) ** 2 + (j - cy) ** 2 <= r ** 2:
                            img[i, j] = 1.0
                    elif label == "square":
                        if abs(i - cx) <= r and abs(j - cy) <= r:
                            img[i, j] = 1.0
                    elif label == "triangle":
                        if (j >= cy - r and j <= cy + r and i >= cx - r and
                            i <= cx + r and (i - cx) <= (r - abs(j - cy))):
                            img[i, j] = 1.0

            noise = rng.normal(0, 0.05, (img_size, img_size))
            img = np.clip(img + noise, 0, 1)
            X.append(img.flatten())
            y.append(code)

    return np.array(X), np.array(y), list(classes.keys())
